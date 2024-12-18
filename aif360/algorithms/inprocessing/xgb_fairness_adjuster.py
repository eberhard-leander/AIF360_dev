# based on adversarial_debiasing.py
# v1: train the base classifier and adjuster internally

import numpy as np
from scipy.special import expit, logit

try:
    from xgboost import XGBClassifier, XGBRegressor
    from .adversary_utils import AdjusterAdversaryLoss
except ImportError as error:
    from logging import warning

    warning(
        "{}: AdversarialDebiasing will be unavailable. To install, run:\n"
        "pip install 'aif360[AdversarialDebiasing]'".format(error)
    )

from aif360.algorithms import Transformer


class XGBFairnessAdjuster(Transformer):
    """
    Fairness adjuster using Adversarial Debiasing.
    """

    def __init__(
        self,
        unprivileged_groups,
        privileged_groups,
        seed=None,
        debias=True,
        adversary_loss_weight=0.1,
        protected_group_vector=None,
        **kwargs
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
        super(XGBFairnessAdjuster, self).__init__(
            unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
        )
        self.seed = seed
        self.adversary_loss_weight = adversary_loss_weight
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
        self.base_estimator = XGBClassifier(**kwargs)

    def fit(self, dataset, **kwargs):
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
        self.base_estimator.fit(X, Y)
        self.base_probs = self.base_estimator.predict_proba(X)[:, 1]
        if self.debias:
            Z = dataset.protected_attributes[
                :,
                dataset.protected_attribute_names.index(self.protected_attribute_name),
            ]
            self.adjuster_loss = AdjusterAdversaryLoss(
                base_preds=self.base_probs,
                protected_attr=Z,
                adversary_weight=self.adversary_loss_weight,
                seed=self.seed,
            )
            self.model_adjuster = XGBRegressor(
                objective=self.adjuster_loss,
                **kwargs,
            )
            self.model_adjuster.fit(X, Y)

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
        dataset_new = dataset.copy(deepcopy=True)
        preds = self.base_estimator.predict_proba(dataset.features)[:, 1]
        if self.debias:
            adjuster_preds = self.model_adjuster.predict(dataset.features)
            preds = expit(logit(preds) + adjuster_preds)

        dataset_new.scores = np.array(preds, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(preds) > 0.5).astype(np.float64).reshape(-1, 1)
        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()
        return dataset_new
