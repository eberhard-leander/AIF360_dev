import numpy as np

try:
    from xgboost import XGBClassifier
    from .adversary_utils import AdversaryLoss

except ImportError as error:
    from logging import warning

    warning(
        "{}: AdversarialDebiasing will be unavailable. To install, run:\n"
        "pip install 'aif360[AdversarialDebiasing]'".format(error)
    )

from aif360.algorithms import Transformer


class XGBAdversarialDebiasing(Transformer):
    """Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [5]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [5] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

    def __init__(
        self,
        unprivileged_groups,
        privileged_groups,
        seed=None,
        debias=True,
        adversary_loss_weight=0.1,
        protected_group_vector=None,
        debug=False,
        **kwargs,
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
        super(XGBAdversarialDebiasing, self).__init__(
            unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
        )
        self.seed = seed

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError(
                "Only one unprivileged_group or privileged_group supported."
            )
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]

        # Check if objective is set correctly
        # if (objective is not None) ^ (debias):
        #     raise ValueError("objective and debias cannot be set independently")

        self.debias = debias
        if self.debias:
            self.objective = AdversaryLoss(
                protected_attr=protected_group_vector,
                adversary_weight=adversary_loss_weight,
                seed=seed,
                debug=debug,
            )
            self.estimator = XGBClassifier(objective=self.objective, **kwargs)
        else:
            self.estimator = XGBClassifier(**kwargs)

    def fit(self, dataset):
        """Compute the model parameters of the fair classifier using gradient
        descent.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        X = dataset.features
        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(), 0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(), 0] = 0.0
        Y = temp_labels
        self.estimator.fit(X, Y)

        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """

        preds = self.estimator.predict_proba(dataset.features)[:, 1]

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.scores = np.array(preds, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(preds) > 0.5).astype(np.float64).reshape(-1, 1)

        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new

    def predict_proba(self, dataset):
        return self.estimator.predict_proba(dataset.features)
