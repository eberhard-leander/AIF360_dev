# based on adversarial_debiasing.py
# v1: train the base classifier and adjuster internally

import numpy as np
from scipy.special import expit

try:
    import tensorflow.compat.v1 as tf
except ImportError as error:
    from logging import warning

    warning(
        "{}: AdversarialDebiasing will be unavailable. To install, run:\n"
        "pip install 'aif360[AdversarialDebiasing]'".format(error)
    )

from aif360.algorithms import Transformer


def logit(p):
    return tf.math.log(p / (1 - p))


class FairnessAdjuster(Transformer):
    """
    Fairness adjuster using Adversarial Debiasing.
    """

    def __init__(
        self,
        unprivileged_groups,
        privileged_groups,
        scope_name,
        sess,
        seed=None,
        adversary_loss_weight=0.1,
        num_epochs=50,
        batch_size=128,
        classifier_num_hidden_units=200,
        debias=True,
    ):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged groups
            privileged_groups (tuple): Representation for privileged groups
            scope_name (str): scope name for the tenforflow variables
            sess (tf.Session): tensorflow session
            seed (int, optional): Seed to make `predict` repeatable.
            adversary_loss_weight (float, optional): Hyperparameter that chooses
                the strength of the adversarial loss.
            num_epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier model.
            debias (bool, optional): Learn a classifier with or without
                debiasing.
        """
        super(FairnessAdjuster, self).__init__(
            unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
        )

        self.scope_name = scope_name
        self.seed = seed

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]

        # separate session for the base model and the adjuster since the base model should be fixed
        # during the adjuster training
        self.base_sess = tf.Session()
        self.adjuster_sess = sess

        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias

        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable."""
        with tf.variable_scope("classifier_model"):
            W1 = tf.get_variable(
                "W1",
                [features_dim, self.classifier_num_hidden_units],
                initializer=tf.initializers.glorot_uniform(seed=self.seeds[0]),
            )
            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name="b1")

            h1 = tf.nn.relu(tf.matmul(features, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob, seed=self.seeds[1])

            W2 = tf.get_variable(
                "W2",
                [self.classifier_num_hidden_units, 1],
                initializer=tf.initializers.glorot_uniform(seed=self.seeds[2]),
            )
            b2 = tf.Variable(tf.zeros(shape=[1]), name="b2")

            pred_logit = tf.matmul(h1, W2) + b2
            pred_label = tf.sigmoid(pred_logit)

        return pred_label, pred_logit

    def _adjuster_model(self, features, features_dim, keep_prob):
        # same model architecture as the classifier for now, but this does not need to be the case
        with tf.variable_scope("adjuster_model"):
            W1 = tf.get_variable(
                "W1",
                [features_dim, self.classifier_num_hidden_units],
                initializer=tf.initializers.glorot_uniform(seed=self.seeds[3]),
            )
            # W1 = tf.Variable(
            #     tf.zeros(shape=[features_dim, self.classifier_num_hidden_units], name="W1")
            # )

            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name="b1")

            h1 = tf.nn.relu(tf.matmul(features, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob, seed=self.seeds[4])

            W2 = tf.get_variable(
                "W2",
                [self.classifier_num_hidden_units, 1],
                initializer=tf.initializers.glorot_uniform(seed=self.seeds[5]),
            )
            # W2 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units, 1], name="W2"))

            b2 = tf.Variable(tf.zeros(shape=[1]), name="b2")

            # the only difference is we don't apply the sigmoid here
            pred = tf.matmul(h1, W2) + b2

        return pred

    def _adversary_model(self, pred_logits, true_labels):
        """Compute the adversary predictions for the protected attribute."""
        with tf.variable_scope("adversary_model"):
            c = tf.get_variable("c", initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

            W2 = tf.get_variable(
                "W2", [3, 1], initializer=tf.initializers.glorot_uniform(seed=self.seeds[6])
            )
            b2 = tf.Variable(tf.zeros(shape=[1]), name="b2")

            pred_protected_attribute_logit = (
                tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
            )
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

        return pred_protected_attribute_label, pred_protected_attribute_logit

    def fit(self, dataset):
        """Compute the model parameters of the fair classifier using gradient
        descent.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        if tf.executing_eagerly():
            raise RuntimeError(
                "AdversarialDebiasing does not work in eager "
                "execution mode. To fix, add `tf.disable_eager_execution()`"
                " to the top of the calling script."
            )

        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seeds = list(np.random.randint(ii32.min, ii32.max, size=7))

        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(), 0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(), 0] = 0.0

        #################################################################################
        # train the base classifier whose predictions will be adjusted afterwards
        #################################################################################
        with tf.variable_scope(self.scope_name):
            # Setup placeholders
            self.features_ph1 = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph1 = tf.placeholder(tf.float32, shape=[None, 1])
            self.true_labels_ph1 = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob1 = tf.placeholder(tf.float32)

            num_train_samples, self.features_dim = np.shape(dataset.features)

            # Obtain classifier predictions and classifier loss
            self.base_pred_labels, self.base_pred_logits = self._classifier_model(
                self.features_ph1, self.features_dim, self.keep_prob1
            )
            pred_labels_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.true_labels_ph1, logits=self.base_pred_logits
                )
            )

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 1000, 0.96, staircase=True
            )
            classifier_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [
                var
                for var in tf.trainable_variables(scope=self.scope_name)
                if "classifier_model" in var.name
            ]

            classifier_grads = []
            # compute the classifier gradients
            for grad, var in classifier_opt.compute_gradients(
                pred_labels_loss, var_list=classifier_vars
            ):
                classifier_grads.append((grad, var))

            classifier_minimizer = classifier_opt.apply_gradients(
                classifier_grads, global_step=global_step
            )

            self.base_sess.run(tf.global_variables_initializer())
            self.base_sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i : self.batch_size * (i + 1)]
                    batch_features = dataset.features[batch_ids]
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1, 1])
                    batch_protected_attributes = np.reshape(
                        dataset.protected_attributes[batch_ids][
                            :,
                            dataset.protected_attribute_names.index(self.protected_attribute_name),
                        ],
                        [-1, 1],
                    )

                    batch_feed_dict = {
                        self.features_ph1: batch_features,
                        self.true_labels_ph1: batch_labels,
                        self.protected_attributes_ph1: batch_protected_attributes,
                        self.keep_prob1: 0.8,
                    }

                    _, pred_labels_loss_value = self.base_sess.run(
                        [classifier_minimizer, pred_labels_loss], feed_dict=batch_feed_dict
                    )
                    if i % 200 == 0:
                        print(
                            "epoch %d; iter: %d; batch classifier loss: %f"
                            % (epoch, i, pred_labels_loss_value)
                        )

        # get the scores from the base classifier. This is a numpy array
        self._base_classifier_scores = self.predict(dataset).scores.astype(np.float32).copy()

        #################################################################################
        # adjust the predictions of the base classifier with the fairness adjuster
        # code largely copied over from the adversarial debiasing implementation
        #################################################################################
        with tf.variable_scope(self.scope_name):
            # Setup placeholders
            self.features_ph2 = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph2 = tf.placeholder(tf.float32, shape=[None, 1])
            self.true_labels_ph2 = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob2 = tf.placeholder(tf.float32)
            self.base_pred_ph2 = tf.placeholder(tf.float32, shape=[None, 1])

            num_train_samples, self.features_dim = np.shape(dataset.features)

            # Obtain adjusted predictions and adjuster loss
            self.adjuster_preds = self._adjuster_model(
                self.features_ph2, self.features_dim, self.keep_prob2
            )

            # note: base predictions should not be updated during prediction
            pred_logits = logit(self.base_pred_ph2) + self.adjuster_preds

            # mean of the squared adjuster predictions
            adjuster_loss = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    tf.zeros_like(self.adjuster_preds), self.adjuster_preds
                )
            )

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = (
                    self._adversary_model(pred_logits, self.true_labels_ph2)
                )
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.protected_attributes_ph2,
                        logits=pred_protected_attributes_logits,
                    )
                )

            pred_labels_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.true_labels_ph2, logits=pred_logits
                )
            )

            # Setup optimizers with learning rates
            global_step2 = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step2, 1000, 0.96, staircase=True
            )
            adjuster_opt = tf.train.AdamOptimizer(learning_rate)
            # adjuster_opt = tf.train.AdagradOptimizer(learning_rate)
            # adjuster_opt = tf.train.GradientDescentOptimizer(starter_learning_rate * 0.1)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)
                # adversary_opt = tf.train.AdagradOptimizer(learning_rate)

            adjuster_vars = [
                var
                for var in tf.trainable_variables(scope=self.scope_name)
                if "adjuster_model" in var.name
            ]
            if self.debias:
                adversary_vars = [
                    var
                    for var in tf.trainable_variables(scope=self.scope_name)
                    if "adversary_model" in var.name
                ]
                # Update adjuster parameters
                adversary_grads = {
                    var: grad
                    for (grad, var) in adversary_opt.compute_gradients(
                        pred_protected_attributes_loss, var_list=adjuster_vars
                    )
                }
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            adjuster_grads = []
            # compute the adjuster gradients
            for grad, var in adjuster_opt.compute_gradients(adjuster_loss, var_list=adjuster_vars):
                adjuster_grads.append((grad, var))

                if self.debias:
                    # Subtract of the component of the gradient that aligns with the adversary
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad

                    grad -= self.adversary_loss_weight * adversary_grads[var]

            adjuster_minimizer = adjuster_opt.apply_gradients(
                adjuster_grads, global_step=global_step2
            )

            if self.debias:
                # Update adversary parameters
                with tf.control_dependencies([adjuster_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(
                        pred_protected_attributes_loss, var_list=adversary_vars
                    )  # , global_step=global_step2)

            self.adjuster_sess.run(tf.global_variables_initializer())
            self.adjuster_sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i : self.batch_size * (i + 1)]
                    batch_features = dataset.features[batch_ids]
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1, 1])
                    batch_protected_attributes = np.reshape(
                        dataset.protected_attributes[batch_ids][
                            :,
                            dataset.protected_attribute_names.index(self.protected_attribute_name),
                        ],
                        [-1, 1],
                    )

                    batch_base_predictions = self._base_classifier_scores[batch_ids]

                    batch_feed_dict = {
                        self.features_ph2: batch_features,
                        self.true_labels_ph2: batch_labels,
                        self.protected_attributes_ph2: batch_protected_attributes,
                        self.keep_prob2: 0.8,
                        self.base_pred_ph2: batch_base_predictions,
                    }
                    if self.debias:
                        (
                            _,
                            _,
                            adjuster_norm_loss_value,
                            pred_labels_loss_value,
                            pred_protected_attributes_loss_vale,
                        ) = self.adjuster_sess.run(
                            [
                                adjuster_minimizer,
                                adversary_minimizer,
                                adjuster_loss,
                                pred_labels_loss,
                                pred_protected_attributes_loss,
                            ],
                            feed_dict=batch_feed_dict,
                        )
                        if i % 200 == 0:
                            print(
                                "epoch %d; iter: %d; batch adjuster loss: %f; batch classifier loss; %f; batch adversarial loss: %f"
                                % (
                                    epoch,
                                    i,
                                    adjuster_norm_loss_value,
                                    pred_labels_loss_value,
                                    pred_protected_attributes_loss_vale,
                                )
                            )
                    else:
                        _, adjuster_norm_loss_value = self.adjuster_sess.run(
                            [adjuster_minimizer, adjuster_loss], feed_dict=batch_feed_dict
                        )
                        if i % 200 == 0:
                            print(
                                "epoch %d; iter: %d; batch adjuster loss: %f"
                                % (epoch, i, adjuster_norm_loss_value)
                            )

        writer = tf.summary.FileWriter("output", self.adjuster_sess.graph)
        writer.close()
        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
            sess: Tensorflow session containing the trained model. Defaults to the adversary session

        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples, _ = np.shape(dataset.features)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = dataset.features[batch_ids]
            batch_labels = np.reshape(dataset.labels[batch_ids], [-1, 1])
            batch_protected_attributes = np.reshape(
                dataset.protected_attributes[batch_ids][
                    :, dataset.protected_attribute_names.index(self.protected_attribute_name)
                ],
                [-1, 1],
            )

            batch_feed_dict = {
                self.features_ph1: batch_features,
                self.true_labels_ph1: batch_labels,
                self.protected_attributes_ph1: batch_protected_attributes,
                self.keep_prob1: 1.0,
            }

            batch_base_pred_logits = self.base_sess.run(
                self.base_pred_logits, feed_dict=batch_feed_dict
            )[:, 0]

            # get the adjuster predictions
            if hasattr(self, "adjuster_preds"):
                batch_feed_dict = {
                    self.features_ph2: batch_features,
                    self.true_labels_ph2: batch_labels,
                    self.protected_attributes_ph2: batch_protected_attributes,
                    self.keep_prob2: 1.0,
                    self.base_pred_ph2: batch_base_pred_logits.reshape(-1, 1),
                }

                batch_adjuster_preds = self.adjuster_sess.run(
                    self.adjuster_preds, feed_dict=batch_feed_dict
                )[:, 0]
            else:
                batch_adjuster_preds = 0.0

            # apply the adjuster predictions to the predicted logits
            batch_pred_logits = expit(batch_base_pred_logits + batch_adjuster_preds).tolist()

            pred_labels += batch_pred_logits

            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels) > 0.5).astype(np.float64).reshape(-1, 1)

        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new
