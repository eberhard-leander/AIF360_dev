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
from flaml import AutoML, tune
from flaml.automl.model import XGBoostSklearnEstimator
import logging
logger = logging.getLogger(__name__) 

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
        debug=False,
<<<<<<< HEAD
        tune_hyperparameters_base=False,
        tuning_settings_base=None,
        tune_hyperparameters_adjuster=False,
        tuning_settings_adjuster=None,
        task="regression",
        use_target=False,
=======
        **kwargs,
>>>>>>> origin/main
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
        
        self.tune_hyperparameters_base = tune_hyperparameters_base
        self.tune_hyperparameters_adjuster = tune_hyperparameters_adjuster

        verbose = 2
        self.base_settings = {
            "time_budget": 60,  # total running time in seconds
            "estimator_list": [
                "xgboost"
            ],  # list of ML learners; we tune XGBoost in this example
            "task": "classification",  # task type
            "log_file_name": "XGBAdversarialDebiasing-Base.log",  # flaml log file
            "seed": self.seed,  # random seed
            "verbose": verbose,
        }
        self.adjuster_settings = {
            "time_budget": 60,  # total running time in seconds
            "estimator_list": [
                "xgboost"
            ],  # list of ML learners; we tune XGBoost in this example
            "task": "regression",  # task type
            "log_file_name": "XGBAdversarialDebiasing-Adjuster.log",  # flaml log file
            "seed": self.seed,  # random seed
            "verbose": verbose,
            # "eval": "cv",
        }
        self.base_settings.update(
            tuning_settings_base
        ) if tuning_settings_base else None
        self.adjuster_settings.update(
            tuning_settings_adjuster
        ) if tuning_settings_adjuster else None

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
        self.debug = debug
        self.debias = debias
        
        self.task = task
        self.use_target = use_target

    def get_X_Y(self, dataset):
        X = dataset.features
        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(), 0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(), 0] = 0.0
        Y = temp_labels
        return X, Y

    def fit(self, dataset, test_dataset=None, **kwargs):
        """Compute the model parameters of the fair classifier using gradient
        descent.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        if self.tune_hyperparameters_base:
            self.base_estimator = AutoML()
        else:
            self.base_estimator = XGBClassifier(**kwargs)
        
        train_X, train_Y = self.get_X_Y(dataset=dataset)
        if test_dataset:
            val_X, val_Y = self.get_X_Y(dataset=test_dataset)
        else:
            val_X = val_Y = None

        if self.tune_hyperparameters_base:
            self.base_estimator.fit(
                X_train=train_X,
                y_train=train_Y,
                X_val=val_X,
                y_val=val_Y,
                **self.base_settings,
            )
            best_params = self.base_estimator.best_config
        else:
            self.base_estimator.fit(train_X, train_Y)
            best_params = {}

        self.base_probs = self.base_estimator.predict_proba(train_X)[:, 1]
        if self.debias:
            Z = dataset.protected_attributes[
                :,
                dataset.protected_attribute_names.index(self.protected_attribute_name),
            ]
            adjuster_loss = AdjusterAdversaryLoss(
                base_preds=self.base_probs,
                protected_attr=Z,
                adversary_weight=self.adversary_loss_weight,
                seed=self.seed,
                debug=self.debug,
<<<<<<< HEAD
                task=self.task,
                use_target=self.use_target,
=======
>>>>>>> origin/main
            )
            kwargs.update(best_params)
            self.model_adjuster = XGBRegressor(
                objective=adjuster_loss,
                **kwargs,
            )
            self.model_adjuster.fit(train_X, train_Y)

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
        preds = self.predict_proba(dataset_new)
        dataset_new.scores = np.array(preds, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(preds) > 0.5).astype(np.float64).reshape(-1, 1)
        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()
        return dataset_new

    def predict_proba(self, dataset):
        preds = self.base_estimator.predict_proba(dataset.features)[:, 1]
        if self.debias:
            adjuster_preds = self.model_adjuster.predict(dataset.features)
            preds = expit(logit(preds) + adjuster_preds)
        return preds
