import torch
from typing import List, Optional
  


class WeightUpdater:
  """
  Class for updating loss weights in PINN training using various auto-weighting schemes.
  """

  def __init__(self, scheme: Optional[int] = 3, alpha: float = 0.9, device: str = "cuda") -> None:
    """
    Initialize the WeightUpdater.

    Args:
      scheme (Optional[int]): The weighting scheme to use.
        - 1: ConflictFree
        - 2: Wang 2023
        - 3: Wang 2021 (Data loss weight is set to 1)
        - 4: Wang 2021 (PDE weight is set to 1)
        - None: No Auto Weights
      alpha (float): Smoothing parameter for weight updates.
      device (str): Device for tensor operations.
    """
    self.device = device
    self.scheme = scheme
    self.alpha = alpha

  def update(
    self,
    current_weights: List[float],
    grads: List[torch.Tensor]
  ) -> List[torch.Tensor]:
    """
    Update the weights according to the selected scheme.

    Args:
      current_weights (List[float]): Current weights for each loss term.
      grads (List[torch.Tensor]): Gradients for each loss term.

    Returns:
      List[torch.Tensor]: Updated weights.
    """
    if len(current_weights) != len(grads):
      raise ValueError("Weights and gradient must have the same length.")

    new_weights: List[torch.Tensor] = []

    if self.scheme == 1 or self.scheme is None:
      # ConflictFree or no Auto Weights
      new_weights = current_weights

    elif self.scheme == 2:
      # Source: Wang et al. - 2023 arXiv:2308.08468v1
      grad_norms: List[torch.Tensor] = [g.norm() for g in grads]
      for i in range(len(current_weights)):
        new_weights.append(sum(grad_norms) / grad_norms[i])

      for i in range(len(current_weights)):
        new_weights[i] = self.alpha * current_weights[i] + (1 - self.alpha) * new_weights[i]

    elif self.scheme == 3:
      # Source: Wang et al. 2021 - https://epubs.siam.org/doi/epdf/10.1137/20M1318043
      # Modified so Data loss weight is set to 1
      new_weights.append(torch.tensor(1, device=self.device).float())

      if len(current_weights) == 3:
        # BC weight
        lambda_bc = torch.mean(torch.abs(grads[0])) / torch.mean(torch.abs(grads[1]))
        new_weights.append(self.alpha * current_weights[1] + (1 - self.alpha) * lambda_bc)

      # NS weight
      lambda_ns = torch.mean(torch.abs(grads[0])) / torch.abs(grads[-1]).max()
      new_weights.append(self.alpha * current_weights[-1] + (1 - self.alpha) * lambda_ns)

    elif self.scheme == 4:
      # Source: Wang et al. 2021 (original) - PDE weight is 1
      lambda_data = torch.abs(grads[-1]).max() / torch.mean(torch.abs(grads[0]))
      new_weights.append(self.alpha * current_weights[0] + (1 - self.alpha) * lambda_data)

      if len(current_weights) == 3:
        # BC weight
        lambda_bc = torch.abs(grads[-1]).max() / torch.mean(torch.abs(grads[1]))
        new_weights.append(self.alpha * current_weights[1] + (1 - self.alpha) * lambda_bc)

      # NS weight is set to 1
      new_weights.append(torch.tensor(1, device=self.device).float())

    elif self.scheme == 5:
      # Source: Wang et al. - 2023 arXiv:2308.08468v1 - modified to let sum of weights be 1
      grad_norms: List[torch.Tensor] = [g.norm() for g in grads]
      for i in range(len(current_weights)):
        new_weights.append(sum(grad_norms) / grad_norms[i])

      # Normalize weights to sum to 1
      total_weight = sum(new_weights)
      for i in range(len(new_weights)):
        new_weights[i] /= total_weight

      for i in range(len(current_weights)):
        new_weights[i] = self.alpha * current_weights[i] + (1 - self.alpha) * new_weights[i]

      

    else:
      raise ValueError("Invalid auto weight type.")

    return new_weights

  def new_grads(
    self,
    weights: List[float],
    grads: List[torch.Tensor]
  ) -> torch.Tensor:
    """
    Compute the new weighted gradients.

    Args:
      weights (List[float]): Weights for each loss term.
      grads (List[torch.Tensor]): Gradients for each loss term.

    Returns:
      torch.Tensor: Weighted sum of gradients.
    """
    if self.scheme == 1:
      # Use ConFIG for ConflictFree gradient aggregation
      # Requires: pip install conflictfree
      from conflictfree.grad_operator import ConFIG_update
      new_grads = ConFIG_update(grads)
    else:
      grads_stacked = torch.stack(grads)
      weight_mat = torch.tensor(weights, device=self.device).float()
      new_grads = grads_stacked.T @ weight_mat

    return new_grads



#### OLD SCHEMES:
"""
 def auto_update_weights2(self, weights: List[float], losses: List[float], self.alpha = 0.9) -> List[float]:
    #own - equal losses
    if len(weights) != len(losses):
      raise ValueError("Weights and losses must have the same length.")

    new_weights = []
    
    for i in range(len(weights)):
      new_weights.append(self.alpha * weights[i] + (1-self.alpha) * losses[i]/sum(losses)) 

    return new_weights

  def auto_update_weights4(self, weights: List[float], gradient_norms: List[float], losses: List[float], self.alpha = 0.9) -> List[float]:
    #Modified Wang 2023 - Weights Multiplied by loss ratio
    if len(weights) != len(gradient_norms):
      raise ValueError("Weights and gradient norms must have the same length.")

    new_weights = []
    
    for i in range(len(weights)):
      new_weights.append(self.alpha * weights[i] + (1-self.alpha) * losses[i]/sum(losses) * sum(gradient_norms) / gradient_norms[i]) 

    return new_weights
  
  def auto_update_weights5(self, weights: List[float], gradient_norms: List[float], self.alpha = 0.9, epochs = 20) -> List[float]:
    #Modified Wang 2023 - update weights only on negative gradients
    if len(weights) != len(gradient_norms):
      raise ValueError("Weights and gradient norms must have the same length.")

    new_weights = []

    if self.epoch > epochs:
      calc_grads = []
      calc_grads.append((self.hist_data[-1]-self.hist_data[-epochs])/epochs)
      calc_grads.append((self.hist_bc[-1]-self.hist_bc[-epochs])/epochs)
      calc_grads.append((self.hist_ns[-1]-self.hist_ns[-epochs])/epochs)

    #if any(calc_grads < 0)
      

      calc_grads = torch.tensor(calc_grads).to(self.device)

      calc_grads = calc_grads - calc_grads.min() + 1
      calc_grads = calc_grads.sum() / calc_grads / 3
      
      for i in range(len(weights)):
        new_weights.append(self.alpha * weights[i] + (1-self.alpha) * calc_grads[i]) 
    else:
      new_weights = weights

    return new_weights

  def auto_update_weights6(self, weights: List[float], gradient_norms: List[float], self.alpha = 0.9, epochs = 20) -> List[float]:
    #Modified Wang 2023 - update weights only on negative gradients
    if len(weights) != len(gradient_norms):
      raise ValueError("Weights and gradient norms must have the same length.")

    new_weights = []

    if self.epoch > epochs:
      calc_grads = []
      calc_grads.append((self.hist_data[-1]-self.hist_data[-epochs])/epochs)
      calc_grads.append((self.hist_bc[-1]-self.hist_bc[-epochs])/epochs)
      calc_grads.append((self.hist_ns[-1]-self.hist_ns[-epochs])/epochs)
      
      print(f"Calc_Grads: {calc_grads}")

      for i in range(len(calc_grads)):
        if calc_grads[i] > 0:
          gradient_norms[i] = -gradient_norms[i]

      gradient_norms = torch.tensor(gradient_norms).to(self.device)

      print(f"New Norms: {gradient_norms}")

      gradient_norms = gradient_norms - gradient_norms.min() + 1
      gradient_norms = gradient_norms.sum() / gradient_norms /3
      
      for i in range(len(weights)):
        new_weights.append(self.alpha * weights[i] + (1-self.alpha) * gradient_norms[i]) 

    else:
      new_weights = weights

    return new_weights
"""