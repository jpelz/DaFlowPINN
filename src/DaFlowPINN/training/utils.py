import torch
from typing import Literal

# Originally from https://github.com/tum-pbs/ConFIG/ (© 2024 TUM Physics-based Simulation, MIT License)
# Modifications: `get_gradient_vector()` modified to return a flat vector on demand → used by LBFGS-optimizer.

def get_para_vector(network: torch.nn.Module) -> torch.Tensor:
    """
    Returns the parameter vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.

    Returns:
        torch.Tensor: The parameter vector of the network.
    """
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed = par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec


def get_gradient_vector(
    network: torch.nn.Module, none_grad_mode: Literal["raise", "zero", "skip"] = "zero", flat = False
) -> torch.Tensor:
    """
    Returns the gradient vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.
        none_grad_mode (Literal['raise', 'zero', 'skip']): The mode to handle None gradients. default: 'skip'
            - 'raise': Raise an error when the gradient of a parameter is None.
            - 'zero': Replace the None gradient with a zero tensor.
            - 'skip': Skip the None gradient.
                        The None gradient usually occurs when part of the network is not trainable (e.g., fine-tuning)
            or the weight is not used to calculate the current loss (e.g., different parts of the network calculate different losses).
            If all of your losses are calculated using the same part of the network, you should set none_grad_mode to 'skip'.
            If your losses are calculated using different parts of the network, you should set none_grad_mode to 'zero' to ensure the gradients have the same shape.

    Returns:
        torch.Tensor: The gradient vector of the network.
    """
    with torch.no_grad():
      if not flat:
        grad_vec = None
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "raise":
                    raise RuntimeError("None gradient detected.")
                elif none_grad_mode == "zero":
                    viewed = torch.zeros_like(par.data.view(-1)).float()
                elif none_grad_mode == "skip":
                    continue
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                viewed = par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec
      else:
        views = []
        for p in network.parameters():
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_().float()
                #continue
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)


def apply_gradient_vector(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
    none_grad_mode: Literal["zero", "skip"] = "zero",
    zero_grad_mode: Literal["skip", "pad_zero", "pad_value"] = "pad_value",
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    This function requires the network contains the some gradient information in order to apply the gradient vector.
    If your network does not contain the gradient information, you should consider using `apply_gradient_vector_para_based` function.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
        none_grad_mode (Literal['zero', 'skip']): The mode to handle None gradients.
            You should set this parameter to the same value as the one used in `get_gradient_vector` method.
        zero_grad_mode (Literal['padding', 'skip']): How to set the value of the gradient if your `none_grad_mode` is "zero". default: 'skip'
            - 'skip': Skip the None gradient.
            - 'padding': Replace the None gradient with a zero tensor.
            - 'pad_value': Replace the None gradient using the value in the gradient.
            If you set `none_grad_mode` to 'zero', that means you padded zero to your `grad_vec` if the gradient of the parameter is None when getting the gradient vector.
            When you apply the gradient vector back to the network, the value in the `grad_vec` corresponding to the previous None gradient may not be zero due to the applied gradient operation.
                        Thus, you need to determine whether to recover the original None value, set it to zero, or set the value according to the value in `grad_vec`.
            If you are not sure what you are doing, it is safer to set it to 'pad_value'.

    """
    if none_grad_mode == "zero" and zero_grad_mode == "pad_value":
        apply_gradient_vector_para_based(network, grad_vec)
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "skip":
                    continue
                elif none_grad_mode == "zero":
                    start = start + par.data.view(-1).shape[0]
                    if zero_grad_mode == "pad_zero":
                        par.grad = torch.zeros_like(par.data)
                    elif zero_grad_mode == "skip":
                        continue
                    else:
                        raise ValueError(f"Invalid zero_grad_mode '{zero_grad_mode}'.")
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                end = start + par.data.view(-1).shape[0]
                par.grad.data = grad_vec[start:end].view(par.data.shape)
                start = end


def apply_gradient_vector_para_based(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    Please only use this function when you are sure that the length of `grad_vec` is the same of your network's parameters.
    This happens when you use `get_gradient_vector` with `none_grad_mode` set to 'zero'.
    Or, the 'none_grad_mode' is 'skip' but all of the parameters in your network is involved in the loss calculation.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.grad = grad_vec[start:end].view(par.data.shape)
            start = end


def apply_para_vector(network: torch.nn.Module, para_vec: torch.Tensor) -> None:
    """
    Applies a parameter vector to the network's parameters.

    Args:
        network (torch.nn.Module): The network to apply the parameter vector to.
        para_vec (torch.Tensor): The parameter vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.data = para_vec[start:end].view(par.data.shape)
            start = end