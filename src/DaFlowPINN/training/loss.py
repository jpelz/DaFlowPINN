import torch
from typing import List, Union

def mse_loss(
  model: callable, 
  X: torch.Tensor, 
  Y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error (MSE) loss between the true values and the model predictions.

    Args:
      model (callable): The model to generate predictions. Should accept X as input and return a tensor.
      X (torch.Tensor): Input tensor to the model.
      Y_true (torch.Tensor): Ground truth tensor with target values.
      *args: Additional positional arguments (unused).
      **kwargs: Additional keyword arguments (unused).

    Returns:
      torch.Tensor: The mean squared error loss as a scalar tensor.
    """
    Y_pred=model(X)
    return torch.nanmean((Y_true-Y_pred[:,0:3])**2)

def scaled_mse_loss(
  model: torch.nn.Module,
  X: torch.Tensor,
  Y_true: torch.Tensor,
  weight: Union[torch.Tensor, List[float]]
) -> torch.Tensor:
    """
    Computes a scaled mean squared error (MSE) loss between the model predictions and true values,
    applying a per-output scaling factor.

    Args:
      model (torch.nn.Module): The neural network model to generate predictions.
      X (torch.Tensor): Input tensor to the model.
      Y_true (torch.Tensor): Ground truth tensor with shape (batch_size, 3).
      weight (torch.Tensor or array-like): Scaling weights for each output dimension (length 3).

    Returns:
      torch.Tensor: The computed scaled MSE loss as a scalar tensor.

    Notes:
      - The weights are normalized such that their sum equals 3.
      - The loss is computed as the mean of the sum of squared errors for each output,
        each scaled by its corresponding normalized weight.
      - Handles both tensor and array-like weights, moving them to the correct device.
    """
    Y_pred=model(X)
    if not isinstance(weight, torch.Tensor):
      weight = torch.tensor(weight, device=Y_true.device).float()
    else:
      weight = weight.to(Y_true.device)
    weight = weight/(weight.sum()/3)
    return torch.nanmean((weight[0]*(Y_true[:,0]-Y_pred[:,0]))**2 + (weight[1]*(Y_true[:,1]-Y_pred[:,1]))**2 + (weight[2]*(Y_true[:,2]-Y_pred[:,2]))**2)

def boundary_loss(model: callable, X: torch.Tensor) -> torch.Tensor:
  """
  Computes the mean squared error loss enforcing zero boundary conditions
  on the first three outputs of the model (e.g., velocity components).

  Args:
    model (callable): The model to generate predictions. Should accept X as input and return a tensor.
    X (torch.Tensor): Input tensor to the model.

  Returns:
    torch.Tensor: The mean squared error loss as a scalar tensor.
  """
  Y_pred = model(X)
  return torch.nanmean((Y_pred[:, 0:3] - 0) ** 2)



def physics_loss(
  model: torch.nn.Module,
  X: torch.Tensor,
  Re: float,
  normalize: callable,
  denormalize: callable,
  return_min_max_p: bool = False #,for normalied equations: norm_scales: List[float] = [1, 1, 1, 1, 1, 1, 1, 1], norm_offsets: List[float] = [0, 0, 0, 0, 0, 0, 0, 0]):
) -> torch.Tensor:
    """
    Computes the physics-informed loss for the Navier-Stokes equations using a PINN model.
    Args:
      model (torch.nn.Module): The neural network model that predicts velocity and pressure fields.
      X (torch.Tensor): Input tensor of shape (N, 4), where columns represent (x, y, z, t) coordinates.
      Re (float): Reynolds number for the dimensionless Navier-Stokes equations.
      normalize (callable): Function to normalize input data.
      denormalize (callable): Function to denormalize model outputs.
      return_min_max_p (bool, optional): If True, also returns the minimum and maximum predicted pressure values. Defaults to False.
    Returns:
      torch.Tensor: The computed physics-informed loss (scalar).
      If return_min_max_p is True, returns a tuple:
        (loss, max_pressure, min_pressure)
        where max_pressure and min_pressure are detached torch.Tensors.
    """
    # With Navier-Stokes equation

    X.requires_grad = True

    def compute_gradients(outputs, inputs):
      return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]



    if denormalize is not None and normalize is not None:
        X_denorm = denormalize(X)
        X_norm = normalize(X_denorm)
    else:
        X_norm = X

    result = model(X_norm)

    if denormalize is not None and normalize is not None:
        result = denormalize(y=result)

    u, v, w, p = result[:, 0], result[:, 1], result[:, 2], result[:, 3]

    u_X=compute_gradients(u, X_denorm)
    v_X=compute_gradients(v, X_denorm)
    w_X=compute_gradients(w, X_denorm)
    p_X=compute_gradients(p, X_denorm)

    u_x, u_y, u_z, u_t = u_X[:, 0], u_X[:, 1], u_X[:, 2], u_X[:, 3]
    v_x, v_y, v_z, v_t = v_X[:, 0], v_X[:, 1], v_X[:, 2], v_X[:, 3]
    w_x, w_y, w_z, w_t = w_X[:, 0], w_X[:, 1], w_X[:, 2], w_X[:, 3]
    p_x, p_y, p_z = p_X[:, 0], p_X[:, 1], p_X[:, 2]


    # Extract velocity components' gradients
    u_x, u_y, u_z, u_t = u_X[:, 0], u_X[:, 1], u_X[:, 2], u_X[:, 3]
    v_x, v_y, v_z, v_t = v_X[:, 0], v_X[:, 1], v_X[:, 2], v_X[:, 3]
    w_x, w_y, w_z, w_t = w_X[:, 0], w_X[:, 1], w_X[:, 2], w_X[:, 3]
    p_x, p_y, p_z = p_X[:, 0], p_X[:, 1], p_X[:, 2]

    # Compute second derivatives
    u_xx = compute_gradients(u_X[:, 0], X_denorm)[:, 0]
    u_yy = compute_gradients(u_X[:, 1], X_denorm)[:, 1]
    u_zz = compute_gradients(u_X[:, 2], X_denorm)[:, 2]

    v_xx = compute_gradients(v_X[:, 0], X_denorm)[:, 0]
    v_yy = compute_gradients(v_X[:, 1], X_denorm)[:, 1]
    v_zz = compute_gradients(v_X[:, 2], X_denorm)[:, 2]

    w_xx = compute_gradients(w_X[:, 0], X_denorm)[:, 0]
    w_yy = compute_gradients(w_X[:, 1], X_denorm)[:, 1]
    w_zz = compute_gradients(w_X[:, 2], X_denorm)[:, 2]

    # Dimensionless Equations
    f_mass = u_x + v_y + w_z
    f_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1/Re * (u_xx + u_yy + u_zz)
    f_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1/Re * (v_xx + v_yy + v_zz)
    f_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1/Re * (w_xx + w_yy + w_zz)


    # Dimensional Equations:
    # rho=1000
    # nu=1E-6

    # f_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x/rho - nu * (u_xx + u_yy + u_zz)
    # f_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y/rho - nu * (v_xx + v_yy + v_zz)
    # f_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z/rho - nu * (w_xx + w_yy + w_zz)


    # Normalized Equations:
    # https://arxiv.org/html/2403.19923v2

    # sx, sy, sz, st, su, sv, sw, sp = norm_scales

    # U = su*u + norm_offsets[4]
    # V = sv*v + norm_offsets[5]
    # W = sw*w + norm_offsets[6]

    # # Dimensionless:

    # f_mass = su/sx*u_x + sv/sy*v_y + sw/sz*w_z
    # # f_x = su/st*u_t + su*(U/sx*u_x + V/sy*u_y + W/sz*u_z) + sp/sx*p_x - 1/Re * su*(u_xx/sx**2 + u_yy/sy**2 + u_zz/sz**2)
    # # f_y = sv/st*v_t + sv*(U/sx*v_x + V/sy*v_y + W/sz*v_z) + sp/sy*p_y - 1/Re * sv*(v_xx/sx**2 + v_yy/sy**2 + v_zz/sz**2)
    # # f_z = sw/st*w_t + sw*(U/sx*w_x + V/sy*w_y + W/sz*w_z) + sp/sz*p_z - 1/Re * sw*(w_xx/sx**2 + w_yy/sy**2 + w_zz/sz**2)

    # # Dimensional:
    # rho=1000
    # nu=1E-6

    # f_x = su/st*u_t + su*(U/sx*u_x + V/sy*u_y + W/sz*u_z) + sp/sx*p_x/rho - nu * su*(u_xx/sx**2 + u_yy/sy**2 + u_zz/sz**2)
    # f_y = sv/st*v_t + sv*(U/sx*v_x + V/sy*v_y + W/sz*v_z) + sp/sy*p_y/rho - nu * sv*(v_xx/sx**2 + v_yy/sy**2 + v_zz/sz**2)
    # f_z = sw/st*w_t + sw*(U/sx*w_x + V/sy*w_y + W/sz*w_z) + sp/sz*p_z/rho - nu * sw*(w_xx/sx**2 + w_yy/sy**2 + w_zz/sz**2)

    loss = torch.mean(f_mass**2) + torch.mean(f_x**2) + torch.mean(f_y**2) + torch.mean(f_z**2)
    
    if not return_min_max_p:
      return loss
    else:
      return loss, max(result[:,3]).detach(), min(result[:,3]).detach()

def numerical_physics_loss(model, X, Re, dx, dy, dz, dt, denormalize: callable = None):
    # n-PINNs as in Chiu et al. 2022 - https://doi.org/10.1016/j.cma.2022.114909
    # Coupling with AD is missing yet

    pred = model(X)
    
    if denormalize is not None:
        X, pred = denormalize(X, pred)

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    t = X[:, 3]
    u = pred[:, 0]
    v = pred[:, 1]
    w = pred[:, 2]
    p = pred[:, 3]
    

    #x
    x1 = torch.stack((x + dx, y, z, t), dim=-1)
    pred = model(x1)
    u_x1 = pred[:, 0]
    v_x1 = pred[:, 1]
    w_x1 = pred[:, 2]
    p_x1 = pred[:, 3]

    x0 = torch.stack((x - dx, y, z, t), dim=-1)
    pred = model(x0)
    u_x0 = pred[:, 0]
    v_x0 = pred[:, 1]
    w_x0 = pred[:, 2]
    p_x0 = pred[:, 3]

    #y
    y1 = torch.stack((x, y + dy, z, t), dim=-1)
    pred = model(y1)
    u_y1 = pred[:, 0]
    v_y1 = pred[:, 1]
    w_y1 = pred[:, 2]
    p_y1 = pred[:, 3]

    y0 = torch.stack((x, y - dy, z, t), dim=-1)
    pred = model(y0)
    u_y0 = pred[:, 0]
    v_y0 = pred[:, 1]
    w_y0 = pred[:, 2]
    p_y0 = pred[:, 3]

    #z
    z1 = torch.stack((x, y, z + dz, t), dim=-1)
    pred = model(z1)
    u_z1 = pred[:, 0]
    v_z1 = pred[:, 1]
    w_z1 = pred[:, 2]
    p_z1 = pred[:, 3]

    z0 = torch.stack((x, y, z - dz, t), dim=-1)
    pred = model(z0)
    u_z0 = pred[:, 0]
    v_z0 = pred[:, 1]
    w_z0 = pred[:, 2]
    p_z0 = pred[:, 3]

    #t
    t1 = torch.stack((x, y, z, t + dt), dim=-1)
    pred = model(t1)
    u_t1 = pred[:, 0]
    v_t1 = pred[:, 1]
    w_t1 = pred[:, 2]
    p_t1 = pred[:, 3]

    t0 = torch.stack((x, y, z, t - dt), dim=-1)
    pred = model(t0)
    u_t0 = pred[:, 0]
    v_t0 = pred[:, 1]
    w_t0 = pred[:, 2]
    p_t0 = pred[:, 3]

    #central difference
    u_x = (u_x1 - u_x0) / (2 * dx)
    v_x = (v_x1 - v_x0) / (2 * dx)
    w_x = (w_x1 - w_x0) / (2 * dx)
    p_x = (p_x1 - p_x0) / (2 * dx)

    u_y = (u_y1 - u_y0) / (2 * dy)
    v_y = (v_y1 - v_y0) / (2 * dy)
    w_y = (w_y1 - w_y0) / (2 * dy)
    p_y = (p_y1 - p_y0) / (2 * dy)

    u_z = (u_z1 - u_z0) / (2 * dz)
    v_z = (v_z1 - v_z0) / (2 * dz)
    w_z = (w_z1 - w_z0) / (2 * dz)
    p_z = (p_z1 - p_z0) / (2 * dz)

    u_t = (u_t1 - u_t0) / (2 * dt)
    v_t = (v_t1 - v_t0) / (2 * dt)
    w_t = (w_t1 - w_t0) / (2 * dt)
    p_t = (p_t1 - p_t0) / (2 * dt)

    #2nd order
    u_xx = (u_x1 - 2 * u + u_x0) / (dx ** 2)
    v_xx = (v_x1 - 2 * v + v_x0) / (dx ** 2)
    w_xx = (w_x1 - 2 * w + w_x0) / (dx ** 2)

    u_yy = (u_y1 - 2 * u + u_y0) / (dy ** 2)
    v_yy = (v_y1 - 2 * v + v_y0) / (dy ** 2)
    w_yy = (w_y1 - 2 * w + w_y0) / (dy ** 2)

    u_zz = (u_z1 - 2 * u + u_z0) / (dz ** 2)
    v_zz = (v_z1 - 2 * v + v_z0) / (dz ** 2)
    w_zz = (w_z1 - 2 * w + w_z0) / (dz ** 2)

    #continuity
    f_mass = u_x + v_y + w_z

    #momentum
    f_x = u_t + u * u_x + v * u_y + w * u_z + p_x - 1 / Re * (u_xx + u_yy + u_zz)
    f_y = v_t + u * v_x + v * v_y + w * v_z + p_y - 1 / Re * (v_xx + v_yy + v_zz)
    f_z = w_t + u * w_x + v * w_y + w * w_z + p_z - 1 / Re * (w_xx + w_yy + w_zz)

    loss = torch.nanmean(f_mass ** 2) + torch.nanmean(f_x ** 2) + torch.nanmean(f_y ** 2) + torch.nanmean(f_z ** 2)
    return loss



  
