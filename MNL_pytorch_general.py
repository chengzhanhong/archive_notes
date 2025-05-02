"""
Author: Zhanhong Cheng
Date: 2025-05-02
Description: A PyTorch implementation of the Multinomial Logit (MNL) model, with methods for parameter unpacking, robust standard error calculation, and significance testing.
Reference: https://chengzhanhong.github.io/2025-05-01/pytorch-mnl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t
import pandas as pd


class MNL_Choice(nn.Module):
    """
    Multinomial Logit model
    """
    def __init__(self, num_features, num_comm):
        super(MNL_Choice, self).__init__()
        self.beta = nn.Parameter(torch.randn(num_features), requires_grad=True)
        self.asc = nn.Parameter(torch.zeros(num_comm - 1), requires_grad=True)

    def forward(self, data):
        # data: (batch_size, num_choice, num_features)
        x = data @ self.beta  # (batch_size, num_choice)
        x[:, :-1] += self.asc
        return F.log_softmax(x, dim=-1)  # (batch_size, num_choice)

    def unpack_param(self, param_vector):
        """
        Unpack the model parameters from a vector.
        Args:
            param_vector: A 1D tensor containing the model parameters.
        Returns: A dictionary mapping parameter names to their values.
        """
        param_dict = {}
        idx = 0
        for name, p in self.named_parameters():
            num_param = p.numel()
            param_dict[name] = param_vector[idx : idx + num_param].view_as(p)
            idx += num_param
        return param_dict

    def get_param_std(self, x, y):
        """
        Calculate the parameter robust standard errors, using hessian and jacobian.

        Returns: a vector of standard errors for each parameter.
        """

        def loss(param, reduction="none"):
            """
            Calculate the negative log-likelihood loss.
            Args:
                param: A vector of model parameters.
                reduction: Specifies the reduction to apply to the output:
                    'none' | 'mean' | 'sum'.
            """
            param_dict = self.unpack_param(param)
            log_probs = torch.func.functional_call(self, param_dict, x)
            nll = F.nll_loss(log_probs, y, reduction=reduction)
            return nll

        def loss_each(param):
            # Calculate the negative log-likelihood loss for each sample.
            return loss(param, reduction="none")

        def loss_total(param):
            # Calculate the total negative log-likelihood loss.
            return loss(param, reduction="sum")

        # Get the model parameters as a vector
        param = torch.cat([p.view(-1) for p in self.parameters()])

        # Compute the Jacobian and Hessian
        jacobian = torch.autograd.functional.jacobian(loss_each, param)
        hessian = torch.autograd.functional.hessian(loss_total, param)

        # Compute the standard errors
        B = jacobian.T @ jacobian  # (n_params, n_params)
        Hinv = torch.linalg.pinv(hessian)
        COV = Hinv @ B @ Hinv
        std_err = torch.sqrt(torch.diag(COV))

        return std_err

    def significance_test(self, x, y):
        """
        Perform a significance test on the model parameters.
        Args:
            x: Input features. (must be the same as used in training, otherwise the output is not meaningful)
            y: Target labels.  (must be the same as used in training, otherwise the output is not meaningful)
        Returns: A pandas DataFrame containing the parameter estimates, standard errors, t-values, and p-values.
        """
        std_err = self.get_param_std(x, y).detach().cpu().numpy()
        param_vector = (
            torch.cat([p.view(-1) for p in self.parameters()]).detach().cpu().numpy()
        )
        t_values = param_vector / std_err
        p_values = 2 * (
            1 - t.cdf(abs(t_values), df=len(x) - len(param_vector))
        )  # two-tailed test

        # organize the results in a pandas DataFrame, columns are params, std_err, t_values, p_values
        # Row names are the parameter names
        results = {
            "params": param_vector,
            "std_err": std_err,
            "t_values": t_values,
            "p_values": p_values,
        }

        param_names = []
        for name, value in self.named_parameters():
            param_names.extend([f"{name}_{i}" for i in range(value.numel())])

        results_df = pd.DataFrame(results, index=param_names)
        return results_df
