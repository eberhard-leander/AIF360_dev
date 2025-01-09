import numpy as np
import torch
from scipy.special import logit, expit
import xgboost

bce_loss = torch.nn.BCELoss(reduction="sum")
norm_loss = torch.nn.MSELoss(reduction="sum")


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim=1):
        """
        input_dim: number of features for the adversary
        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(
            input_dim, 1
        )  # Single output for binary classification

    def forward(self, x):
        return torch.sigmoid(
            self.linear(x)
        )  # Sigmoid activation for binary classification


def logistic_regression_generator(seed=None, lr=0.1):
    if seed:
        torch.manual_seed(seed)

    # fit the adversary model using pytorch
    model = LogisticRegressionModel()
    # log loss, aka binary cross entropy loss. We use mean here so that the learning rate is independent of the
    # data size
    criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    while True:
        X, y = yield

        for _ in range(1):
            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = criterion(y_pred, y)

            # Backward pass
            loss.backward(retain_graph=True)

            # clip the gradients to improve stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

        yield model


class OptimizedBCELoss:
    """
    Similar performance to the native XGBoost loss function
    """

    def __call__(self, y_true, y_pred):
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        # Direct sigmoid computation (vectorized)
        prob = 1 / (1 + np.exp(-y_pred))

        # Compute gradients and Hessians
        grad = prob - y_true
        hess = prob * (1 - prob)

        return grad, hess


class AdversaryLoss:
    def __init__(self, protected_attr, adversary_weight, seed=None, debug=False):
        self.protected_attr = protected_attr
        self.protected_attr_tensor = torch.tensor(
            protected_attr.astype(np.float32).reshape(-1, 1), requires_grad=False
        )
        self.adversary_weight = adversary_weight
        self.adversary_generator = logistic_regression_generator(seed=seed)
        next(self.adversary_generator)

        self.seed = seed
        self.debug = debug

        self.iter = 0

    def __call__(self, y_true, y_pred):
        self.iter += 1
        # print(f"Type y_pred {type(y_pred)}")
        # print(f"Type dtrain {type(dtrain)}")
        # print(f"Shape of preds is {y_pred.shape}")
        # print(f"Shape of dtrain is {dtrain.shape}")

        # y_true = dtrain.get_label()
        # classifier accuracy
 
        # if isinstance(y_pred, xgboost.core.DMatrix):
        #     y_pred = y_pred.get_data().toarray()
        # print(f"Shape of preds is {y_pred.shape}")
        # print(f"Shape of true is {y_true.shape}")

        prob = 1 / (1 + np.exp(-y_pred))
        classifier_grad = prob - y_true
        classifier_hess = prob * (1 - prob)

        # Compute the adversary hessian and gradient
        # xgboost predicts the margin, so we need to convert this back into a probability
        pred_margin_tensor = torch.tensor(y_pred.astype(np.float32), requires_grad=True)
        pred_prob_tensor = torch.sigmoid(pred_margin_tensor)

        # fit the adversary model
        detached_pred_prob_tensor = pred_prob_tensor.detach()

        to_send = (
            detached_pred_prob_tensor.reshape(-1, 1),
            self.protected_attr_tensor,
        )
        adversary_model = self.adversary_generator.send(to_send)
        next(self.adversary_generator)

        # predict using the model
        adversary_pred = adversary_model(pred_prob_tensor.reshape(-1, 1))

        # multiply by negative adversary weight
        adversary_loss = bce_loss(adversary_pred, self.protected_attr_tensor)

        first_derivative = torch.autograd.grad(
            adversary_loss, pred_margin_tensor, create_graph=True
        )[0]

        second_derivative = torch.autograd.grad(
            first_derivative,
            pred_margin_tensor,
            grad_outputs=torch.ones_like(first_derivative),
            create_graph=False,
        )[0]

        adversary_grad = first_derivative.detach().numpy()
        adversary_hess = second_derivative.detach().numpy()

        if self.debug:
            print(f"{adversary_loss=}")
            print(adversary_grad[:10])
            print(adversary_hess[:10])

        if self.iter >= 10:
            grad = classifier_grad - self.adversary_weight * adversary_grad
            hess = classifier_hess - self.adversary_weight * adversary_hess
        else:
            grad = classifier_grad
            hess = classifier_hess

        return grad, hess


class AdjusterAdversaryLoss:
    def __init__(
        self, base_preds, protected_attr, adversary_weight, seed=None, debug=False, task="regression", use_target=False,
    ):
        self.base_preds = base_preds
        self.base_pred_logit_tensor = torch.tensor(
            logit(base_preds).astype(np.float32), requires_grad=False
        )
        self.protected_attr_tensor = torch.tensor(
            protected_attr.astype(np.float32).reshape(-1, 1), requires_grad=False
        )
        self.adversary_weight = adversary_weight
        self.adversary_generator = logistic_regression_generator(seed=seed, lr=0.1)
        next(self.adversary_generator)

        self.seed = seed
        self.debug = debug

        self.task = task
        self.use_target = use_target

        self.iter = 0

    def __call__(self, y_true, y_pred):
        eps = 10e-6

        self.iter += 1
        
        if self.task == "regression":
            if self.use_target:
                norm_grad = 2 * (y_pred + self.base_preds - y_true)
                norm_hess = 2 * np.ones(y_pred.shape)
            else:
                # norm loss (equivalent to MSE with y_true = y_hat)
                norm_grad = 2 * y_pred
                norm_hess = 2 * np.ones(y_pred.shape)
        elif self.task == "classification":
            if self.use_target:
                adjusted_pred = expit(y_pred + logit(self.base_preds))
                norm_grad = adjusted_pred - y_true
                norm_hess = adjusted_pred * (1 - adjusted_pred)
            else:
                # log loss with y_hat = as the true probabilities
                adjusted_pred = np.clip(expit(y_pred + logit(self.base_preds)), eps, 1-eps)
                norm_grad = adjusted_pred - self.base_preds
                norm_hess = adjusted_pred * (1 - adjusted_pred)
        else:
            raise ValueError(f"Task {self.task} not supported!")

        # Compute the adversary hessian and gradient
        adjuster_pred_tensor = torch.tensor(
            y_pred.astype(np.float32), requires_grad=True
        )

        pred_prob_tensor = torch.sigmoid(
            adjuster_pred_tensor + self.base_pred_logit_tensor
        )

        # fit the adversary model
        detached_pred_prob_tensor = pred_prob_tensor.detach()

        to_send = (
            detached_pred_prob_tensor.reshape(-1, 1),
            self.protected_attr_tensor,
        )
        adversary_model = self.adversary_generator.send(to_send)
        next(self.adversary_generator)

        # predict using the model
        adversary_pred = adversary_model(pred_prob_tensor.reshape(-1, 1))

        # multiply by negative adversary weight
        adversary_loss = bce_loss(adversary_pred, self.protected_attr_tensor)

        first_derivative = torch.autograd.grad(
            adversary_loss, adjuster_pred_tensor, create_graph=True
        )[0]

        second_derivative = torch.autograd.grad(
            first_derivative,
            adjuster_pred_tensor,
            grad_outputs=torch.ones_like(first_derivative),
            create_graph=False,
        )[0]

        adversary_grad = first_derivative.detach().numpy()
        adversary_hess = second_derivative.detach().numpy()

        if self.debug:
            print(f"{adversary_loss=}")
            print(adversary_grad[:10])
            print(adversary_hess[:10])

        if self.iter >= 10:
            grad = norm_grad - self.adversary_weight * adversary_grad
            hess = norm_hess - self.adversary_weight * adversary_hess
        else:
            grad = norm_grad
            hess = norm_hess

        return grad, hess
