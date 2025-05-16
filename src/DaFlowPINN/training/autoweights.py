import torch
from typing import List, Literal, Optional, Sequence, Union
  


class WeightUpdater:
    def __init__(self, scheme = 3, alpha = 0.9, device = "cuda"):
        self.device = device
        self.scheme = scheme
        self.alpha = alpha

    def update(self, current_weights: List[float], grads: List[float]):

        if len(current_weights) != len(grads):
            raise ValueError("Weights and gradient must have the same length.")

        new_weights = []
        

        

        if self.scheme == 1 or self.scheme == None:
            new_weights = current_weights

        elif self.scheme == 2:
            # Source:  Wang - 2023 arXiv:2308.08468v1

            grad_norms = []
            for i in range(len(grads)):
              grad_norms.append(grads[i].norm())
              
            for i in range(len(current_weights)):
                new_weights.append(sum(grad_norms)/grad_norms[i])

            max_weight = max(new_weights)

            for i in range(len(current_weights)):
              new_weights[i] = self.alpha * current_weights[i] + (1-self.alpha) * new_weights[i] #/ max_weight

        elif self.scheme == 3 or self.scheme == 4:
            # Source: Wang 2021 - https://epubs.siam.org/doi/epdf/10.1137/20M1318043 - PDE weight is 1"

            # --> Modified so data loss is 1
            # --> 4 is for combination with conflictfree
            
            #Data weight
            new_weights.append(torch.tensor(1, device=self.device).float())

            if len(current_weights) == 3:
              #BC weight
              lambda_bc = torch.mean(torch.abs(grads[0])) / torch.mean(torch.abs(current_weights[1]*grads[1]))
              new_weights.append(self.alpha * current_weights[1] + (1-self.alpha) * lambda_bc)

            #NS weight
            lambda_ns = torch.mean(torch.abs(grads[0])) / torch.abs(current_weights[-1]*grads[-1]).max()
            new_weights.append(self.alpha * current_weights[-1] + (1-self.alpha) * lambda_ns)


            #Original:
            # for i in range(len(current_weights)-1):
            #     lambda_i = torch.abs(pde_weight).max() / torch.mean(torch.abs(current_weights[i]*grads[i]))
            #     new_weights.append(self.alpha * current_weights[i] + (1-self.alpha) * lambda_i)

            #new_weights.append(torch.tensor(1, device=self.device).float())


        elif self.scheme == 4:
          grad_norms = []
          for i in range(len(grads)):
            grad_norms.append(grads[i].norm())

          min_norm = min(grad_norms)

          for i in range(len(current_weights)):
            new_weights.append(min_norm/grad_norms[i])

          max_weight = max(new_weights)

          for i in range(len(current_weights)):
            new_weights[i] = self.alpha * current_weights[i] + (1-self.alpha) * new_weights[i]


        else:
            raise ValueError("Invalid auto weight type.")

        # for i in range(len(new_weights)):
        #   new_weights[i] /= max(new_weights)

        return new_weights
    
    def new_grads(self, weights: List[float], grads: List[float]):
        if self.scheme == 0:
            #See https://tum-pbs.github.io/ConFIG/
            #needs pip install conflictfree

            from conflictfree.grad_operator import ConFIG_update
            new_grads = ConFIG_update(grads)
        elif self.scheme == 4:
          from conflictfree.grad_operator import ConFIG_update
          new_grads = ConFIG_update(grads, weight_model=self.configModel)

        else:
            grads = torch.stack(grads)
            weight_mat = torch.tensor(weights, device=self.device).float()
            new_grads = grads.T @ weight_mat

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