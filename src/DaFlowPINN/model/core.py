import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from typing import List, Tuple, Union, Callable

import os
import sys

import copy


from .architectures import HardBC, RFF, FCN

from ..training.loss import mse_loss, scaled_mse_loss, boundary_loss, physics_loss, numerical_physics_loss
from ..training.optim import LBFGS, SOAP
from ..training.optim.scheduler import ReduceLROnPlateau_custom, WarmupScheduler
from ..training.autoweights import WeightUpdater
from ..training.utils import get_gradient_vector, apply_gradient_vector

from ..post.plot import plotPINN_2D
from ..post.evaluation import detailed_data_err

class Network(nn.Module):
  """
  Wrapper network for PINN models supporting optional Fourier features and hard boundary conditions.
  """
  def __init__(
    self,
    model: nn.Module,
    N_LAYERS: int = 5,
    N_NEURONS: int = 512,
    hardBC_sdf: callable = None,
    fourier_feature: bool = False,
    **kwargs
  ):
    """
    Initializes the core model for DaFlowPINN.
    Args:
      model (nn.Module): The base neural network architecture to be used.
      N_LAYERS (int, optional): Number of layers in the neural network. Default is 5.
      N_NEURONS (int, optional): Number of neurons per layer. Default is 512.
      hardBC_sdf (callable, optional): Signed or approximate distance function (SDF/ADF) for enforcing hard boundary conditions. If provided, hard boundary conditions are applied.
      fourier_feature (bool, optional): If True, applies random Fourier feature mapping to the input. Default is False.
      **kwargs: Additional keyword arguments for feature mapping and scaling, used only if `fourier_feature` is True:
        - mapping_size (int, optional): Number of random Fourier features (default: 512).
        - scale_x (float, optional): Scaling factor for the x input dimension (default: 1).
        - scale_y (float, optional): Scaling factor for the y input dimension (default: 1).
        - scale_z (float, optional): Scaling factor for the z input dimension (default: 1).
        - scale_t (float, optional): Scaling factor for the t input dimension (default: 1).
    Notes:
      - If `fourier_feature` is True, the model input is mapped to a higher-dimensional space using random Fourier features, and the above scaling factors and mapping size can be set via kwargs.
      - If `hardBC_sdf` is provided, hard boundary conditions are enforced using the given SDF/ADF function.

    """
    super().__init__()
    
    if fourier_feature:
      self.fourier_feature: bool = True
      mapping_size: int = kwargs.get('mapping_size', 512)
      self.mapping_size: int = mapping_size if mapping_size is not None else 512
      scale_x: float = kwargs.get('scale_x', 1.0)
      scale_y: float = kwargs.get('scale_y', 1.0)
      scale_z: float = kwargs.get('scale_z', 1.0)
      scale_t: float = kwargs.get('scale_t', 1.0)
      # Input dim: 4, output dim: 2*mapping_size after RFF
      self.network: nn.Module = model(2 * self.mapping_size, 4, N_NEURONS, N_LAYERS)
      self.rff: RFF = RFF(n_freqs=self.mapping_size, scales=[scale_x, scale_y, scale_z, scale_t])
    else:
      self.fourier_feature: bool = False
      self.network: nn.Module = model(4, 4, N_NEURONS, N_LAYERS)

    if hardBC_sdf is not None:
      self.hardBC: bool = True
      self.hbc: HardBC = HardBC(hardBC_sdf)
    else:
      self.hardBC: bool = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the network, applying optional Fourier features and hard boundary conditions.

    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, 4).

    Returns:
      torch.Tensor: Output tensor of shape (batch_size, 4).
    """
    x = x.float()
    if self.hardBC:
      x0 = x.detach()
    if self.fourier_feature:
      x = self.rff(x)
    x = self.network(x)
    if self.hardBC:
      x = self.hbc(x, x0)
    return x

#MARK: --- Begin of PINN ---
# 'MARK' comments are used for marking sections in VSCode

class PINN_3D(nn.Module):
  """
  PINN_3D: Physics-Informed Neural Network for 3D PDEs.

  This class implements a PINN for solving 3D partial differential equations (PDEs) using neural networks.
  It supports:
    - Data-driven supervised learning (with data points)
    - Physics-based loss via collocation points (enforcing PDEs)
    - Boundary condition enforcement (soft or hard)
    - Optional Fourier feature mapping and hard boundary conditions
    - Automatic mixed precision (AMP) training
    - Collocation point growth and dynamic point updates
    - Learning rate scheduling and advanced optimizers (Adam, LBFGS, SOAP)
    - Loss weighting and auto-weighting schemes
    - Training/validation split, normalization, and non-dimensionalization
    - Plotting and evaluation utilities

  Attributes:
    model (nn.Module): The wrapped neural network (with optional Fourier features and hard BC).
    NAME (str): Model identifier.
    Re (float): Reynolds number for non-dimensionalization.
    N_LAYERS (int): Number of layers in the neural network.
    N_NEURONS (int): Number of neurons per layer.
    amp_enabled (bool): Enable AMP training.
    device (torch.device): Device for computation.
    N_COLLOCATION (int): Number of collocation (physics) points.
    N_BC (int): Number of boundary condition points.
    N_DATA (int): Number of data points.
    point_update_freq (int): Frequency for updating points.
    collocation_growth (bool): Enable collocation growth.
    epoch (int): Current training epoch.
    model_cpu (nn.Module): Model copy for CPU inference.
    hist_total, hist_data, hist_ns, hist_bc, hist_lr (list): Loss and LR histories.
    plot_setups (list): Plot configuration setups.
    l_scale, u_scale, t_scale, p_scale (float): Non-dimensionalization scales.
    norm_scales, norm_offsets (torch.Tensor): Normalization parameters.
    lambda_data, lambda_bc, lambda_ns (float): Loss weights.
    scheduler: Learning rate scheduler.
    optimizer, post_optimizer: Optimizer(s).
    autoweight (bool): Enable auto-weighting of losses.
    grad_acc (bool): Enable gradient accumulation.
    ... (many other training and bookkeeping attributes)
  """
  def __init__(self,
               model: nn.Module,
               NAME: str,
               Re: float,
               N_LAYERS: int = 4,
               N_NEURONS: int = 256,
               amp_enabled: bool = False,
               hardBC_sdf: callable = None,
               fourier_feature: bool = False,
               **kwargs):
    """
    Initializes the PINN_3D class with the given parameters.

    Args:
      model (nn.Module): The base neural network architecture to be used (e.g., FCN).
      NAME (str): Identifier for the model instance.
      Re (float): Reynolds number for non-dimensionalization.
      N_LAYERS (int, optional): Number of layers in the neural network. Defaults to 4.
      N_NEURONS (int, optional): Number of neurons per layer in the neural network. Defaults to 512.
      amp_enabled (bool, optional): Enable automatic mixed precision (AMP) training. Defaults to False.
      hardBC_sdf (callable, optional): Signed or approximate distance function for hard boundary conditions. If provided, hard BCs are enforced.
      fourier_feature (bool, optional): If True, applies random Fourier feature mapping to the input. Defaults to False.
      **kwargs: Additional keyword arguments for feature mapping and scaling, used only if `fourier_feature` is True:
      - mapping_size (int, optional): Number of random Fourier features (default: 512).
      - scale_x (float, optional): Scaling factor for the x input dimension (default: 1).
      - scale_y (float, optional): Scaling factor for the y input dimension (default: 1).
      - scale_z (float, optional): Scaling factor for the z input dimension (default: 1).
      - scale_t (float, optional): Scaling factor for the t input dimension (default: 1).
    """
    

    # Set default data type to float
    torch.set_default_dtype(torch.float)

    # Set random seed for reproducibility
    torch.manual_seed(123)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(123)
    super().__init__()
    
    # Initialize the model with optional Fourier features and hard boundary conditions
    self.model = Network(model, N_LAYERS, N_NEURONS, hardBC_sdf, fourier_feature, **kwargs)

    # Set model parameters for saving/loading
    self.fourier_feature = fourier_feature
    if self.fourier_feature:
      self.mapping_size = kwargs.get('mapping_size', 512)
    self.NAME = NAME
    self.N_LAYERS = N_LAYERS
    self.N_NEURONS = N_NEURONS

    # Set device to GPU if available, otherwise CPU
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
      torch.set_num_threads(os.cpu_count())


    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
      self.model = DataParallel(self.model)

    # Move the model to the specified device
    self.model = self.model.to(self.device)
    

    # Initialize collocation, boundary, and data points
    self.N_COLLOCATION = 0
    self.N_BC = 0
    self.N_DATA = 0

    # Initialize point update frequency and collocation growth flag
    self.point_update_freq = None
    self.collocation_growth = False

    # Initialize epoch and loss tensors
    self.epoch = 0
    self.inner_iter = 0
    self.loss_bc = torch.tensor(0, device=self.device)
    self.loss_data = torch.tensor(0, device=self.device)
    self.loss_ns = torch.tensor(0, device=self.device)
    self.loss_ns_sum = torch.tensor(0, device=self.device)
    self.loss_total = torch.tensor(0, device=self.device)

    # Initialize data, boundary, and collocation loss weights
    self.lambda_data = 1.0
    self.lambda_bc = 1.0
    self.lambda_ns = 1.0

    # Initialize scheduler and AMP scaler
    self.scheduler = None
    self.amp_enabled = amp_enabled
    self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    self.post_optimizer = None

    # Initialize model for CPU inference
    self.model_cpu = model

    # Initialize history lists for losses, learning rate, RMSEs, and loss weights
    self.hist_total = []
    self.hist_data = []
    self.hist_ns = []
    self.hist_bc = []
    self.hist_lr = []

    self.p_hist = []

    self.hist_RMSE_test_u = []
    self.hist_RMSE_test_v = []
    self.hist_RMSE_test_w = []

    self.hist_l1 = []
    self.hist_l2 = []
    self.hist_l3 = []

    # Initialize plot setups
    self.plot_setups = []

    # Initialize non-dimensionalization scales
    self.Re = Re

    self.l_scale = 1
    self.u_scale = 1
    self.t_scale = 1
    self.p_scale = 1


    # Initialize normalization scales and offsets
    self.norm_scales = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], device=self.device)
    self.norm_offsets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], device=self.device)

    self.autoweight = False

    self.detailed_err = True

    self.NaN_counter = 0

#MARK: Forwarding
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the PINN model.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor after passing through the model and redimensionalizing.
    """
    # Apply dimensionless transformation to the input
    #print(min(x[:, 1]), max(x[:, 1]))

    x_dimless = self.dimensionless(x)

    #print(min(x_dimless[:, 1]), max(x_dimless[:, 1]))
    
    # Pass the dimensionless input through the model
    #print(min(self.normalize(x_dimless)[:, 1]), max(self.normalize(x_dimless)[:, 1]))
    y_dimless = self.model(self.normalize(x_dimless))
    
    # Redimensionalize the model output
    y = self.redimension(y=self.denormalize(y=y_dimless))
    
    return y
  

  def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the CPU model.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor after passing through the model and redimensionalizing.
    """
    if not isinstance(x, torch.Tensor):
      raise TypeError("Input x must be a torch.Tensor")
    
    # Apply dimensionless transformation and pass through the CPU model
    #print(min(x[:, 1]), max(x[:, 1]))
    #print(min(self.dimensionless(x)[:, 1]), max(self.dimensionless(x)[:, 1]))
    #print(min(self.normalize(self.dimensionless(x))[:, 1]), max(self.normalize(self.dimensionless(x))[:, 1]))


    y = self.model_cpu(self.normalize(self.dimensionless(x)))
    
    # Redimensionalize the model output
    return self.redimension(y=self.denormalize(y=y))
  
  def get_forward_callable(self) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a callable for CPU inference with the current model state.
    This function creates a deep copy of the model on the CPU, sets it to eval mode,
    and returns a function that performs a forward pass using the CPU model.

    Returns:
      Callable[[torch.Tensor], torch.Tensor]: A function that takes a torch.Tensor as input and returns the model output.
    """
    # Load the model on the CPU
    if isinstance(self.model, DataParallel):
      self.model_cpu = copy.deepcopy(self.model.module)
    else:
      self.model_cpu = copy.deepcopy(self.model)

    self.model_cpu.to('cpu')
    self.model_cpu.eval()

    def forward_cpu(x: torch.Tensor) -> torch.Tensor:
      return self.forward_cpu(x)
    
    return forward_cpu


  def define_domain(self, lb: np.ndarray, ub: np.ndarray) -> None:
    """
    Defines the domain boundaries for the PINN model.

    Args:
      lb (np.ndarray): Lower bounds of the domain. (x, y, z, t)
      ub (np.ndarray): Upper bounds of the domain. (x, y, z, t)
    """
    self.lb = lb
    self.ub = ub

    # Extract time boundaries from the domain
    self.t_min = lb[3]
    self.t_max = ub[3]

#MARK: Dimensionless

  def set_dimensionless(self, l_scale: float, u_scale: float, p_scale: float, t_scale: float = None) -> None:
    """
    Sets the dimensionless scales for the PINN model.

    Args:
      l_scale (float): Length scale for non-dimensionalization. l* = l / l_scale
      u_scale (float): Velocity scale for non-dimensionalization. u* = u / u_scale
      p_scale (float): Pressure scale for non-dimensionalization. p* = p / p_scale
        - for high viscosity flows, p_scale = μ * u_scale / l_scale
        - for high velocity flows, p_scale = ρ * u_scale^2
      t_scale (float, optional): Time scale for non-dimensionalization. t* = t / t_scale (If not provided, it is calculated as l_scale / u_scale.)
    """
    self.l_scale = l_scale
    self.u_scale = u_scale
    self.p_scale = p_scale
    self.t_scale = t_scale if t_scale is not None else l_scale / u_scale

  def dimensionless(self, x: torch.Tensor = None, y: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """
    Applies dimensionless transformation to the input and/or output tensors.

    Args:
      x (torch.Tensor, optional): Input tensor to be transformed. Defaults to None.
      y (torch.Tensor, optional): Output tensor to be transformed. Defaults to None.

    Returns:
      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]: Transformed input and/or output tensors.
    """
    if x is not None:
      # Apply dimensionless transformation to the input tensor
      x = x * torch.tensor([1/self.l_scale, 1/self.l_scale, 1/self.l_scale, 1/self.t_scale], device=x.device)

    if y is not None:
      # Apply dimensionless transformation to the output tensor
      scale = [1/self.u_scale, 1/self.u_scale, 1/self.u_scale]
      if y.shape[1] == 4:
        scale.append(1/self.p_scale)
      y = y * torch.tensor(scale, device=y.device)

    if x is not None and y is not None:
      return x, y
    elif x is not None:
      return x
    elif y is not None:
      return y
    return None
  
  def redimension(self, x: torch.Tensor = None, y: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """
    Applies redimensionalization to the input and/or output tensors.

    Args:
      x (torch.Tensor, optional): Input tensor to be redimensionalized. Defaults to None.
      y (torch.Tensor, optional): Output tensor to be redimensionalized. Defaults to None.

    Returns:
      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]: Redimensionalized input and/or output tensors.
    """
    if x is not None:
      # Apply redimensionalization to the input tensor
      x = x * torch.tensor([self.l_scale, self.l_scale, self.l_scale, self.t_scale], device=x.device)

    if y is not None:
      # Apply redimensionalization to the output tensor
      scale = [self.u_scale, self.u_scale, self.u_scale]
      if y.shape[1] == 4:
        scale.append(self.p_scale)
      y = y * torch.tensor(scale, device=y.device)

    if x is not None and y is not None:
      return x, y
    elif x is not None:
      return x
    elif y is not None:
      return y
    return None    

  
#MARK: Normalization
  def normalize(self, x: torch.Tensor = None, y: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """
    Normalizes the input and/or output tensors.

    Args:
      x (torch.Tensor, optional): Input tensor to be normalized. Defaults to None.
      y (torch.Tensor, optional): Output tensor to be normalized. Defaults to None.

    Returns:
      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]: Normalized input and/or output tensors.
    """
    if x is not None:
      # Normalize the input tensor
      scales = self.norm_scales[:4].to(x.device)
      offsets = self.norm_offsets[:4].to(x.device)
      x = (x - offsets) / scales

    if y is not None:
      # Normalize the output tensor
      if y.shape[1] == 4:
        scales = self.norm_scales[4:].to(y.device)
        offsets = self.norm_offsets[4:].to(y.device)
      elif y.shape[1] == 3:
        scales = self.norm_scales[4:7].to(y.device)
        offsets = self.norm_offsets[4:7].to(y.device)
      y = (y - offsets) / scales

    if x is not None and y is not None:
      return x, y
    elif x is not None:
      return x
    elif y is not None:
      return y
    return None


  def denormalize(self, x: torch.Tensor = None, y: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
    """
    Denormalizes the input and/or output tensors.

    Args:
      x (torch.Tensor, optional): Input tensor to be denormalized. Defaults to None.
      y (torch.Tensor, optional): Output tensor to be denormalized. Defaults to None.

    Returns:
      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]: Denormalized input and/or output tensors.
    """
    if x is not None:
      # Denormalize the input tensor
      scales = self.norm_scales[:4].to(x.device)
      offsets = self.norm_offsets[:4].to(x.device)
      x = x * scales + offsets

    if y is not None:
      # Denormalize the output tensor
      if y.shape[1] == 4:
        scales = self.norm_scales[4:].to(y.device)
        offsets = self.norm_offsets[4:].to(y.device)
      elif y.shape[1] == 3:
        scales = self.norm_scales[4:7].to(y.device)
        offsets = self.norm_offsets[4:7].to(y.device)
      y = y * scales + offsets

    if x is not None and y is not None:
      return x, y
    elif x is not None:
      return x
    elif y is not None:
      return y
    return None



#MARK: - Data Handling
  def add_data_points(self, data: np.ndarray, batch_size = None, test_size = 0.05, train_test_file = None, n_acc: int = 1, weight: float = 1.0) -> None:
    """
    Adds data points for supervised learning to the PINN model, applies dimensionless transformation and normalization, and prepares train/test splits.

    Args:
      data (np.ndarray): Array containing the data points. The array should have at least 8 columns:
      - Column 0: Index or identifier (optional, otherwise empty column)
      - Columns 1-4: Input features (x, y, z, t)
      - Columns 5-7: Output features (u, v, w)
      batch_size (int, optional): Batch size for training. If None, uses all data.
      test_size (float, optional): Fraction of data to use for testing. Default is 0.05.
      train_test_file (str, optional): Path to a .npz file with precomputed train/test splits.
      n_acc (int, optional): Number of batches for gradient accumulation. Default is 1.
      weight (float, optional): Weight for the data points. Default is 1.0.
    """

    if data.shape[1] < 8:
      raise ValueError("Data array must have at least 8 columns: [index, x, y, z, t, u, v, w]")

    self.data = data
    self.test_size = test_size
    self.lambda_data = weight

    # Extract input (X) and output (Y) tensors from the data array
    X = torch.from_numpy(data[:, 1:5]).float()  # x, y, z, t
    Y = torch.from_numpy(data[:, 5:8]).float()  # u, v, w

    # Apply dimensionless transformation to the input and output tensors
    X, Y = self.dimensionless(X, Y)

    # Set Normalization
    self.norm_scales = torch.tensor([X[:, 0].max() - X[:, 0].min(),
                                     X[:, 1].max() - X[:, 1].min(),
                                     X[:, 2].max() - X[:, 2].min(),
                                     X[:, 3].max() - X[:, 3].min(),
                                     Y[:, 0].max() - Y[:, 0].min(),
                                     Y[:, 1].max() - Y[:, 1].min(),
                                     Y[:, 2].max() - Y[:, 2].min(),
                                     1], device=self.device) #/2    # --> for -1 to 1 normalization

    self.norm_offsets = torch.tensor([X[:, 0].min(),
                                      X[:, 1].min(),
                                      X[:, 2].min(),
                                      X[:, 3].min(),
                                      Y[:, 0].min(),
                                      Y[:, 1].min(),
                                      Y[:, 2].min(),
                                      0], device=self.device) #+ 1 * self.norm_scales   # --> for -1 to 1 normalization

    #Z-Score normalization:
    # self.norm_scales = torch.tensor([X[:, 0].std(),
    #                             X[:, 1].std(),
    #                             X[:, 2].std(),
    #                             X[:, 3].std(),
    #                             Y[:, 0].std(),
    #                             Y[:, 1].std(),
    #                             Y[:, 2].std(),
    #                             1], device=self.device)

    # self.norm_offsets = torch.tensor([X[:, 0].mean(),
    #                               X[:, 1].mean(),
    #                               X[:, 2].mean(),
    #                               X[:, 3].mean(),
    #                               Y[:, 0].mean(),
    #                               Y[:, 1].mean(),
    #                               Y[:, 2].mean(),
    #                               0], device=self.device)

    # Compute scaling facors for weighted MSE loss
    loss_scale_u = 1
    loss_scale_v = self.norm_scales[5]/self.norm_scales[4]
    loss_scale_w = self.norm_scales[6]/self.norm_scales[4]

    
    self.loss_scales = torch.tensor([loss_scale_u, loss_scale_v, loss_scale_w])

    #print(f"Normalization scales: {self.norm_scales}")
    #print(f"New RFF Scales: {self.model.scales}")

    # Rescale RFF-Parameters
    if hasattr(self.model, "rff"):      
      self.model.rff.scales = nn.Parameter(self.model.rff.scales * self.norm_scales[0:4], requires_grad=False)

    # Rescale HBC-Parameters
    if hasattr(self.model, "hbc"):
      self.model.hbc.offset = nn.Parameter(torch.tensor(((self.model.hbc.offset[0]-self.norm_offsets[4])/self.norm_scales[4],
                                             (self.model.hbc.offset[1]-self.norm_offsets[5])/self.norm_scales[5],
                                             (self.model.hbc.offset[2]-self.norm_offsets[6])/self.norm_scales[6],
                                             (self.model.hbc.offset[3]-self.norm_offsets[7])/self.norm_scales[7])), requires_grad=False)



    X, Y = self.normalize(X, Y)
    # Set the number of data points
    
    self.N_DATA = len(data[:,0])

    if test_size > 0:
      train_idx = np.random.choice(self.N_DATA, int(self.N_DATA*(1-test_size)), replace=False)
      test_idx = np.setdiff1d(np.arange(self.N_DATA), train_idx)
      self.X_train = X[train_idx, :].to(self.device)
      self.Y_train = Y[train_idx, :].to(self.device)
      self.X_test = X[test_idx, :].to(self.device)
      self.Y_test = Y[test_idx, :].to(self.device)

      if train_test_file is None:
        np.savez("data.npz",
        x_train = self.redimension(self.denormalize(self.X_train)).cpu().numpy(),
        y_train = self.redimension(y=self.denormalize(y=self.Y_train)).cpu().numpy(),
        x_test = self.redimension(self.denormalize(self.X_test)).cpu().numpy(),
        y_test = self.redimension(y=self.denormalize(y=self.Y_test)).cpu().numpy())
    
    else:
      self.X_train = X.to(self.device)
      self.Y_train = Y.to(self.device)
      self.X_test = None
      self.Y_test = None

    self.N_DATA = len(self.X_train[:, 0])

    if train_test_file is not None:

      data = np.load(train_test_file)

      x_train = data["x_train"]
      y_train = data["y_train"]
      x_test = data["x_test"]
      y_test = data["y_test"]

      self.X_train = torch.tensor(x_train, device = self.device)
      self.Y_train = torch.tensor(y_train, device = self.device)
      self.X_test = torch.tensor(x_test, device = self.device)
      self.Y_test = torch.tensor(y_test, device = self.device)

      self.X_train, self.Y_train = self.dimensionless(self.X_train, self.Y_train)
      self.X_test, self.Y_test = self.dimensionless(self.X_test, self.Y_test)

      self.X_train, self.Y_train = self.normalize(self.X_train, self.Y_train)
      self.X_test, self.Y_test = self.normalize(self.X_test, self.Y_test)



    self.N_DATA = len(self.X_train[:, 0])
    
    self.data_batch_size = batch_size if batch_size is not None else self.N_DATA
    self.n_acc_data = n_acc

  
    self.dataset = CustomDataset(self.X_train, self.Y_train, batch_size=self.data_batch_size, shuffle=True)
    self.dataloader = DataLoader(self.dataset, batch_size=1)

#MARK: Boundary Conditions
  def add_boundary_condition(self, surface_sampler: callable, N_BC_POINTS: int, weight: float, values = [0, 0, 0], batch_size = None, n_acc = 1) -> None:
    """
    Adds boundary condition points for supervised boundary loss to the PINN model.

    Args:
      surface_sampler (callable): Function that samples points on the boundary surface and returns their coordinates as a numpy array.
      N_BC_POINTS (int): Number of boundary condition points to sample.
      weight (float): Weight for the boundary condition loss term.
      values (list, optional): Target values for the boundary condition (default: [0, 0, 0]).
      batch_size (int, optional): Batch size for boundary condition points (default: all points).
      n_acc (int, optional): Number of accumulations for boundary condition batches (default: 1).
    """

    if self.N_DATA == 0:
      raise RuntimeError("Data points must be added before boundary condition points.")
    
    self.surface_sampler = surface_sampler
    boundary = self.surface_sampler(N_BC_POINTS)

    # Convert boundary points to tensor
    X_boundary = torch.from_numpy(boundary).float()
    Y_boundary = torch.ones_like(X_boundary[:,:3]) * torch.tensor(values).float()

    X_boundary, Y_boundary = self.dimensionless(X_boundary, Y_boundary)
    X_boundary, Y_boundary = self.normalize(X_boundary, Y_boundary)

    # Set the number of boundary condition points and the loss weight
    self.N_BC = N_BC_POINTS
    self.lambda_bc = weight
    self.bc_values = values
    self.bc_batch_size = batch_size if batch_size is not None else self.N_BC
    self.n_acc_bc = n_acc

    self.bc_dataset = CustomDataset(X_boundary, Y_boundary, batch_size=self.bc_batch_size, shuffle=True)
    self.bc_dataloader = DataLoader(self.bc_dataset, batch_size=1)

#MARK: Physics Points
  def add_physics_points(self, N_COLLOCATION: int, batch_size: int, geometry: callable = None, weight: float = 1.0, keep_percentage: float = 20, numerical = False, p_scale = 1, n_acc = 1) -> None:
    """
    Adds physics collocation points (interior points) for enforcing the PDE residual in the PINN model.

    Args:
      N_COLLOCATION (int): Number of collocation (physics) points to sample.
      batch_size (int): Batch size for the physics DataLoader.
      geometry (callable, optional): Function that returns a boolean mask for points inside the domain geometry. Defaults to None.
      weight (float, optional): Weight for the physics (Navier-Stokes) loss term. Defaults to 1.0.
      keep_percentage (float, optional): Percentage of existing collocation points to retain when updating points. Defaults to 20.
      numerical (bool, optional): If True, use numerical physics loss. Defaults to False.
      p_scale (float, optional): Scaling factor for pressure normalization. Defaults to 1.
      n_acc (int, optional): Number of accumulations for physics batches. Defaults to 1.
    """
    if self.N_DATA == 0:
      raise RuntimeError("Data points must be added before physics points.")

    current_nr_points = self.N_COLLOCATION
    self.norm_scales[7] = p_scale
    self.norm_offsets[7] = 0

    if N_COLLOCATION > 0:
      # Sample interior points within the domain
      if current_nr_points == 0 or keep_percentage == 0:
       new_points = torch.from_numpy(sample_interior_points(N_COLLOCATION, self.lb, self.ub, geometry)).float()
       self.X_physics = self.normalize(self.dimensionless(new_points))
      else:
        nr_keep = int(keep_percentage/100 * current_nr_points)
        nr_new = N_COLLOCATION - nr_keep
        new_points = torch.from_numpy(sample_interior_points(nr_new, self.lb, self.ub, geometry)).float()

        keep_idx = np.random.choice(current_nr_points, nr_keep, replace=False)
        keep_points = self.X_physics[keep_idx]
        self.X_physics = torch.cat((keep_points, new_points), dim=0)
      
      # Create a dataset and dataloader for the collocation points
      self.physics_dataset = CustomDataset(self.X_physics, batch_size=batch_size, shuffle = True)
      self.physics_dataloader = DataLoader(self.physics_dataset, batch_size=1)

    # Set the number of collocation points, batches, geometry, and loss weight
    self.N_COLLOCATION = N_COLLOCATION
    self.physics_batch_size = batch_size
    self.geometry = geometry
    self.lambda_ns = weight
    self.numerical = numerical
    self.n_acc_physics = n_acc

  def update_points(self, keep_percentage: float) -> None:
    """
    Updates the collocation and boundary points for the PINN model.

    Args:
      keep_percentage (float): Percentage of existing collocation points to retain when updating points.
    """
    # Update the number of collocation points if collocation growth is enabled
    if self.collocation_growth:
      new_point_nr = self.update_collocation_nr(self.epoch)
    else:
      new_point_nr = self.N_COLLOCATION
    
    # Add physics collocation points if there are any
    if new_point_nr > 0:
      self.add_physics_points(new_point_nr, self.physics_batch_size, self.geometry, self.lambda_ns, keep_percentage, self.numerical, self.norm_scales[7], self.norm_offsets[7])
    
    # Add boundary condition points if there are any
    if self.N_BC > 0:
      self.add_boundary_condition(self.surface_sampler, self.N_BC, self.lambda_bc, self.bc_values, self.bc_batch_size)

  def add_collocation_growth(self, n_init: int, n_max: int, epoch_start: int, epoch_end: int, increase_scheme: str = 'linear', **kwargs) -> None:
    """
    Enables dynamic collocation point growth during training.

    Args:
      n_init (int): Initial number of collocation points at the start of growth.
      n_max (int): Maximum number of collocation points at the end of growth.
      epoch_start (int): Epoch at which to begin increasing the number of collocation points.
      epoch_end (int): Epoch at which to reach the maximum number of collocation points.
      increase_scheme (str, optional): Strategy for increasing collocation points. Defaults to 'linear'.
        - 'linear': Linearly increases the number of points from n_init to n_max between epoch_start and epoch_end.
        - 'exponential': Exponentially increases the number of points; requires 'epsilon' in kwargs (controls the end ratio).
        - 'logarithmic': Logarithmically increases the number of points.
      **kwargs: Additional keyword arguments for specific increase schemes (e.g., 'epsilon' for exponential).
    """
    self.collocation_growth = True

    def update_collocation_nr(epoch: int) -> int:
      """
      Updates the number of collocation points based on the current epoch.

      Args:
        epoch (int): Current epoch.

      Returns:
        int: Updated number of collocation points.
      """
      if epoch < epoch_start:
        return self.N_COLLOCATION
      elif epoch > epoch_end:
        return n_max
      else:
        if increase_scheme == 'linear':
          # Linear increase
          return int(n_init + (n_max - n_init) * (epoch - epoch_start) / (epoch_end - epoch_start))
        elif increase_scheme == 'exponential':
          # Exponential increase
          epsilon = kwargs.get('epsilon', 0.1)
          k = np.log(epsilon) / (epoch_end - epoch_start)
          return int(n_init + (n_max - n_init) * (1 - np.exp(k * (epoch - epoch_start))))
        elif increase_scheme == 'logarithmic':
          # Logarithmic increase
          return int((n_max - n_init) / np.log(epoch_end - epoch_start) * np.log(epoch - epoch_start + 1) + n_init)
        else:
          raise ValueError("Invalid increase scheme")

    self.update_collocation_nr = update_collocation_nr
     
#MARK: Optimizer & Scheduler  
  def add_optimizer(self, optim: str = "adam", lr: float = 1e-3, lbfgs_type: str = "standard") -> None:
    """
    Adds an optimizer to the PINN model.

    Args:
      optim (str, optional): The optimizer to use. Defaults to "adam".
      - "adam": Adam optimizer.
      - "sgd": Stochastic Gradient Descent optimizer.
      - "lbfgs": Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer.
      lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
      lbfgs_type (str, optional): Type of L-BFGS optimizer (applied only when LBFGS is activated). Defaults to "standard".
      - "standard": Standard L-BFGS optimizer implemented in PyTorch
      - "full_overlap": Full overlap L-BFGS optimizer (see https://github.com/hjmshi/PyTorch-LBFGS)
      - "multibatch": Multibatch L-BFGS optimizer with partial overlap (see https://github.com/hjmshi/PyTorch-LBFGS)
    Raises:
      ValueError: If an invalid optimizer is specified.
    """
    self.optim_lr=lr
    self.optim = optim
    if optim == "adam":
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) #betas=(.9, .999)
    elif optim == "adam2":
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.95, .95))
    elif optim == "soap":
      self.optimizer = SOAP(self.model.parameters(), lr = lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    elif optim == "sgd":
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    elif optim == "lbfgs":
      if lbfgs_type == "standard":
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
      elif lbfgs_type == "full_overlap": 
        self.optimizer = LBFGS(self.model.parameters(), lr=lr, line_search='None')
        #May not work on multiple GPUs
      elif lbfgs_type == "multibatch":
        self.optimizer = LBFGS(self.model.parameters(), lr=lr, line_search='None')
        #May not work on multiple GPUs
      else:
        raise ValueError("Invalid lbfgs_type specified. Choose from 'standard', 'full_overlap', or 'multibatch'.")
      self.amp_enabled = False
      self.lbfgs_type = lbfgs_type
    else:
      raise ValueError("Invalid optimizer specified. Choose from 'adam', 'sgd', or 'lbfgs'.")
    
  def add_postTraining(self, lr: float = 1, epochs: int = 1000, lbfgs_type: str = "standard") -> None:
    """
    Adds an additional post-training stage using the L-BFGS optimizer.

    Args:
      lr (float, optional): Learning rate for the L-BFGS optimizer. Defaults to 1.
      epochs (int, optional): Number of post-training epochs. Defaults to 1000.
      lbfgs_type (str, optional): Type of L-BFGS optimizer. Options are "standard", "full_overlap", or "multibatch". Defaults to "standard".
    """
    if lbfgs_type == "standard":
      self.post_optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
    elif lbfgs_type == "full_overlap": 
      self.post_optimizer = LBFGS(self.model.parameters(), lr=lr, line_search='Armijo')
      #May not work on multiple GPUs
    elif lbfgs_type == "multibatch":
      self.post_optimizer = LBFGS(self.model.parameters(), lr=lr, line_search='None')
      #May not work on multiple GPUs
    else:
      raise ValueError("Invalid lbfgs_type specified. Choose from 'standard', 'full_overlap', or 'multibatch'.")
    self.amp_enabled = False
    
    self.post_lr = lr

    self.amp_enabled = False
    self.lbfgs_type = lbfgs_type
    
    self.post_epochs = epochs

  def add_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    """
    Adds a learning rate scheduler to the PINN model.
    Set up via: PINN_3D.add_scheduler(LRScheduler(optimizer=PINN_3D.optimizer, ...))

    Args:
      scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to use.
    """
    self.scheduler = scheduler


#MARK: Prepare Plots
  def add_2D_plot(self, plot_dims: Tuple[int, int], dim3_slice: float, t_slice: float, plot_data: bool = False, component1: int = 0, centered1: bool = False, vmin1: float = None, vmax1: float = None,
                  component2: int = 4, centered2: bool = False, vmin2: float = None, vmax2: float = None,
                  lb: np.ndarray = None, ub: np.ndarray = None, resolution: List[int] = None, dim3_tolerance: float = None) -> None:
    """
    Adds a 2D plot configuration to the PINN model.
    Call multiple times to add multiple plots.


    Args:
      plot_dims (Tuple[int, int]): Indices of the two spatial dimensions to plot (e.g., (0, 1) for x-y).
      dim3_slice (float): Value at which to slice the third spatial dimension.
      t_slice (float): Value at which to slice the time dimension.
      plot_data (bool, optional): If True, overlay data points on the plot. Defaults to False.
      component1 (int, optional): Index of the first field/component to plot. Defaults to 0.
      centered1 (bool, optional): If True, center the color scale for component1. Defaults to False.
      vmin1 (float, optional): Minimum value for the color scale of component1. Defaults to None.
      vmax1 (float, optional): Maximum value for the color scale of component1. Defaults to None.
      component2 (int, optional): Index of the second field/component to plot. Defaults to 4.
      centered2 (bool, optional): If True, center the color scale for component2. Defaults to False.
      vmin2 (float, optional): Minimum value for the color scale of component2. Defaults to None.
      vmax2 (float, optional): Maximum value for the color scale of component2. Defaults to None.
      lb (np.ndarray, optional): Lower bounds for the plot region. Defaults to the model's lower bounds.
      ub (np.ndarray, optional): Upper bounds for the plot region. Defaults to the model's upper bounds.
      resolution (List[int], optional): Resolution of the plot grid as [n_x, n_y]. Defaults to [100, 100].
      dim3_tolerance (float, optional): Tolerance for showing data points in the third dimension. Defaults to 5% of the model's third dimension range.

    Field/component indices:
      0: Magnitude of velocity (|u|)
      1: u component of velocity
      2: v component of velocity
      3: w component of velocity
      4: Pressure (p)
      5: du1/dx
      6: du1/dy
      7: du2/dx
      8: du2/dy
      9: Vorticity
      (NYI): Q-criterion

    """

    plot_setup = {
      'type': '2D',
      'plot_dims': plot_dims,
      'dim3_slice': dim3_slice,
      't_slice': t_slice,
      'plot_data': plot_data,
      'component1': component1,
      'component2': component2,
      'lb': lb if lb is not None else self.lb[0:2],
      'ub': ub if ub is not None else self.ub[0:2],
      'resolution': resolution if resolution is not None else [100, 100],
      'dim3_tolerance': dim3_tolerance if dim3_tolerance is not None else 0.05 * (self.ub[2] - self.lb[2]),
      'centered1': centered1,
      'vmin1': vmin1,
      'vmax1': vmax1,
      'centered2': centered2,
      'vmin2': vmin2,
      'vmax2': vmax2
    }
    self.plot_setups.append(plot_setup)


  #MARK: Callback
  def callback(self) -> None:
    """
    Callback function to print the current epoch and losses.
    This function is called periodically during training to provide updates on the training progress.
    """
    print(f'------ Epoch {self.epoch} ------')
    if self.N_DATA > 0:
      print(f'    Data Loss: {self.loss_data:.3e}')
    if self.N_BC > 0:
      print(f'    BC Loss: {self.loss_bc:.3e}')
    if self.N_COLLOCATION > 0:
      print(f'    NS Loss: {self.loss_ns:.3e}')
    print(f'    Total Loss: {self.loss_total:.3e}')
    if self.autoweight:
      print(f"    Weights: Data: {self.lambda_data:.3e}, BC: {self.lambda_bc:.3e}, NS:{self.lambda_ns:.3e}")
    if self.detailed_err:

      # abs_rmse, rel_rmse, abs_mae, rel_mae = detailed_data_err(self.model, self.X_train, self.Y_train, self.denormalize, self.redimension)
      # print("    Detailed Data Errors (Train):")
      # print(f"      MAE (u,v,w) in m/s: {abs_mae}")
      # print(f"      MAE (u,v,w) in %: {rel_mae}")
      # print(f"      RMSE (u,v,w) in m/s: {abs_rmse}")
      # print(f"      RMSE (u,v,w) in %: {rel_rmse}") --> too slow for large datasets

      if self.X_test is not None:
        abs_rmse, rel_rmse, abs_mae, rel_mae = detailed_data_err(self.model, self.X_test, self.Y_test, self.denormalize, self.redimension)
        print("    Detailed Data Errors (for Test-Data):")
        print(f"      MAE (u,v,w) in m/s: {abs_mae}")
        print(f"      MAE (u,v,w) in %: {rel_mae}")
        print(f"      RMSE (u,v,w) in m/s: {abs_rmse}")
        print(f"      RMSE (u,v,w) in %: {rel_rmse}")

        self.hist_RMSE_test_u.append(abs_rmse[0])
        self.hist_RMSE_test_v.append(abs_rmse[1])
        self.hist_RMSE_test_w.append(abs_rmse[2])


  #MARK: Loss Computation
  def get_mse_loss(
      self, 
      x: torch.Tensor, 
      y: torch.Tensor, 
      flat_grad: bool = False
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the scaled mean squared error (MSE) loss between model predictions and targets,
    performs backpropagation (with optional AMP), and returns the detached loss and gradient.

    Args:
      x (torch.Tensor): Input tensor for the model.
      y (torch.Tensor): Target tensor.
      flat_grad (bool, optional): If True, returns the gradient as a flat vector. Defaults to False.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: The detached loss tensor and the gradient tensor (possibly flattened).
    """
    with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.amp_enabled):
      self.optimizer.zero_grad()
      loss_data = scaled_mse_loss(self.model, x, y, self.loss_scales)
      if self.amp_enabled:
        scaled_loss_data = self.scaler.scale(loss_data)
        scaled_loss_data.backward()
      else:
        loss_data.backward()
      grad = get_gradient_vector(self.model, flat=flat_grad)
      return loss_data.detach(), grad
    
  def get_physics_loss(self, x: torch.Tensor, flat_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the physics (Navier-Stokes) loss for the given collocation points and returns the loss and its gradient.

    Args:
      x (torch.Tensor): Collocation points (physics points) as input tensor.
      flat_grad (bool, optional): If True, returns the gradient as a flat vector. Defaults to False.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: The detached physics loss tensor and the gradient tensor (possibly flattened).
    """
    with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.amp_enabled):
      self.optimizer.zero_grad()

      if self.numerical:
        # Compute numerical physics loss if enabled
        loss_ns = numerical_physics_loss(self.model, x, self.Re, 0.0025, 0.0025, 0.0025, 2.0E-3, self.denormalize)
      else:
        # Compute analytical physics loss and track pressure extrema
        loss_ns, pmax, pmin = physics_loss(self.model, x, self.Re, self.normalize, self.denormalize, True)
        loss_ns = loss_ns
        self.p_hist.append(pmax.cpu())

      # Backpropagate the loss with or without AMP
      if self.amp_enabled:
        scaled_loss_ns = self.scaler.scale(loss_ns)
        scaled_loss_ns.backward(retain_graph=True)
      else:
        loss_ns.backward()

      # Get the gradient vector (optionally flattened)
      grad = get_gradient_vector(self.model, flat=flat_grad)

      return loss_ns.detach(), grad
    


  def compute_losses(self) -> torch.Tensor:
    """
    Computes the losses for the PINN model, including data loss, boundary condition loss, and physics loss.
    The losses are computed and backpropagated with optional automatic mixed precision (AMP) scaling.

    Returns:
      torch.Tensor: The total loss.
    """
    # Zero the gradients of the optimizer to ensure no accumulation from previous steps
    self.optimizer.zero_grad()

    # Initialize lists to collect gradients and weights for each loss term (data, BC, physics)
    grads = []   # Stores gradients for each loss component
    weights = [] # Stores current weights (lambdas) for each loss component

    # Compute data loss and its gradient if data points are available
    if self.N_DATA > 0:
        self.loss_data = 0    # Accumulate data loss over batches
        grad_data = 0         # Accumulate data gradient over batches
        n_batches = int(self.n_acc_data)  # Number of batches for gradient accumulation

        # Loop over data batches for gradient accumulation
        for idx in range(n_batches):
            x, y = next(iter(self.dataloader))  # Get next batch of data
            loss, grad = self.get_mse_loss(x[0], y[0])  # Compute MSE loss and gradient
            self.loss_data += loss / n_batches          # Average loss over batches
            grad_data += grad / n_batches               # Average gradient over batches

        grads.append(grad_data)         # Store data gradient
        weights.append(self.lambda_data) # Store data loss weight

    # Compute boundary condition loss and gradient if BC points are available
    if self.N_BC > 0:
        self.loss_bc = 0    # Accumulate BC loss
        grad_bc = 0         # Accumulate BC gradient
        n_batches = int(self.n_acc_bc)  # Number of BC batches

        # Loop over BC batches for gradient accumulation
        for idx in range(n_batches):
            x_b, y_b = next(iter(self.bc_dataloader))  # Get next BC batch
            loss, grad = self.get_mse_loss(x_b[0], y_b[0])  # Compute BC loss and gradient
            self.loss_bc += loss / n_batches                # Average BC loss
            grad_bc += grad / n_batches                     # Average BC gradient

        grads.append(grad_bc)         # Store BC gradient
        weights.append(self.lambda_bc) # Store BC loss weight

    # Compute physics (PDE) loss and gradient if collocation points are available
    if self.N_COLLOCATION > 0:
        self.loss_ns = 0    # Accumulate physics loss
        grad_ns = 0         # Accumulate physics gradient
        n_batches = int(self.n_acc_physics)  # Number of physics batches

        # Loop over physics batches for gradient accumulation
        for idx in range(n_batches):
            batch = next(iter(self.physics_dataloader))[0]  # Get next physics batch
            loss, grad = self.get_physics_loss(batch)        # Compute physics loss and gradient
            self.loss_ns += loss / n_batches                 # Average physics loss
            grad_ns += grad / n_batches                      # Average physics gradient

        grads.append(grad_ns)         # Store physics gradient
        weights.append(self.lambda_ns) # Store physics loss weight

    # Optionally update loss weights using an auto-weighting scheme
    if self.autoweight and (self.epoch % self.weight_update_freq == 0) and (self.inner_iter == 0):
        weights = self.WeightUpdater.update(weights, grads)  # Update weights based on gradients

        # Assign updated weights to the respective loss components
        if not self.N_BC == 0:
            self.lambda_data, self.lambda_bc, self.lambda_ns = weights
        else:
            self.lambda_data, self.lambda_ns = weights

    # Compute the new combined gradient vector using the (possibly updated) weights
    new_grads = self.WeightUpdater.new_grads(weights, grads)
    apply_gradient_vector(self.model, new_grads)  # Apply the combined gradient to the model

    self.inner_iter += 1  # Increment inner iteration counter (for gradient accumulation or weight updates)

    # Calculate the total weighted loss for logging and optimization
    self.loss_total = (
        self.lambda_data * self.loss_data +
        self.lambda_bc * self.loss_bc +
        self.lambda_ns * self.loss_ns
    )

    return self.loss_total  # Return the total loss for this step


  #MARK: Output Updating
  def update_all(self):
    """
    Updates the training state of the model at each epoch.
    This method performs the following operations:
    - Checks for NaN or excessively large loss values. If detected, restores the last best model checkpoint and stops training.
    - Saves the model checkpoint if the current loss improves upon the best recorded loss at specified intervals.
    - Steps the learning rate scheduler if available.
    - Records the current losses, lambda weights, and learning rate history.
    - Periodically updates training points if point update frequency is set.
    - Prints callback information and plots training history at specified intervals.
    - Plots the loss history and model field at specified intervals.
    Side Effects:
      - May restore model state from disk.
      - May save model state to disk.
      - Updates internal history lists.
      - May print to stdout and generate plots.
    Returns:
      None
    """


    current_loss = self.loss_total.item()

    if (current_loss>1e12 or np.isnan(current_loss)) and os.path.exists(f"{self.NAME}_best_state.pt"):
      print("NaN detected! Restoring last best model...")
      self.load(f"{self.NAME}_best_state.pt")
      print("Stopping training.")
      self.epoch = self.max_epochs + (self.post_epochs if self.post_optimizer is not None else 0)
      sys.stdout.flush()
      self.plot_field()
      self.plot_history()
      return
    else:
      current_loss = self.loss_total.item()
      if (current_loss < self.best_loss) and (self.epoch % self.save_freq == 0):
        self.best_loss = current_loss
        self.save(f"{self.NAME}_best_state.pt")


    # Step the learning rate scheduler if available
    if self.scheduler is not None:
      if isinstance(self.scheduler, ReduceLROnPlateau_custom):
        self.scheduler.step(current_loss)
      else:
        self.scheduler.step()
    
    # Record loss history
    self.hist_total.append(self.loss_total)
    self.hist_data.append(self.loss_data)
    self.hist_bc.append(self.loss_bc)
    self.hist_ns.append(self.loss_ns)

    # Record lambda history if auto-weighting is enabled
    self.hist_l1.append(self.lambda_data)
    self.hist_l2.append(self.lambda_bc)
    self.hist_l3.append(self.lambda_ns)
    
    # Record learning rate history if scheduler is available
    if self.scheduler is not None:
      self.hist_lr.append(self.scheduler.get_last_lr())
    
    # Update points periodically if point update frequency is set
    if self.point_update_freq is not None:
      if self.epoch % self.point_update_freq == 0:
        self.update_points(self.point_keep_percentage)
    
    # Print callback information periodically
    if self.epoch % self.print_freq == 0:
      self.callback()
      sys.stdout.flush()
      self.plot_history()

    # Plot loss history and field at specified intervals
    if self.epoch % self.plot_freq == 0:
      self.plot_field()

  
  #MARK: LBFGS - Methods
  def step_standard_LBFGS(self, optimizer) -> None:

    update_batches_in_inner_iters = True  # Whether to update data batches in inner iterations of LBFGS --> causes curvature pairs calculated from different batches

    if not update_batches_in_inner_iters:
      if self.N_DATA>0: self.dataset.autostepping = False
      if self.N_BC>0: self.bc_dataset.autostepping = False
      if self.N_COLLOCATION>0: self.physics_dataset.autostepping = False
      self.inner_iter = 0

    optimizer.step(self.compute_losses)

    if not update_batches_in_inner_iters:
      if self.N_DATA>0: self.dataset.__manual_step()
      if self.N_BC>0: self.bc_dataset.__manual_step()
      if self.N_COLLOCATION>0: self.physics_dataset.__manual_step()


  def step_full_overlap_LBFGS(self, optimizer) -> None:
    """
    Performs a single step of the full-overlap L-BFGS optimizer.

    Args:
      optimizer: The L-BFGS optimizer instance.

    This method handles the full-overlap L-BFGS step, including line search and warm-up learning rate scheduling.
    It defines a closure for the optimizer, computes the current loss and gradient, performs the optimizer step,
    and updates the curvature pairs.
    """

    line_search = optimizer.param_groups[0]['line_search']

    # Warm-up learning rate scheduling
    warm_up_epochs = 50
    warm_up = self.epoch > self.max_epochs and ((self.epoch - self.max_epochs) <= warm_up_epochs)

    if warm_up:
      self.inner_iter = 1 #to prevent weight updating

    #Only needed for line search
    def closure():
      optimizer.zero_grad()

      with torch.set_grad_enabled(line_search=='Wolfe'):
        if self.N_DATA > 0:
          X, Y = self.dataset.__getitem__(0, step = False)
          loss_data = self.lambda_data * scaled_mse_loss(self.model, X, Y, self.loss_scales)
          loss = loss_data
            
        if self.N_BC > 0:
          X_b, Y_b = self.bc_dataset.__getitem__(0, step = False)
          loss_bc = self.lambda_bc * scaled_mse_loss(self.model, X_b, Y_b, self.loss_scales)
          loss += loss_bc

      if self.N_COLLOCATION > 0:
        batch = self.physics_dataset.__getitem__(0, step = False)
        if self.numerical:
          with torch.set_grad_enabled(line_search=='Wolfe'):
            loss_ns = self.lambda_ns * numerical_physics_loss(self.model, batch, self.Re, 0.01, 0.01, 0.01, 0.05, self.denormalize)
        else:
          loss_ns = self.lambda_ns * physics_loss(self.model, batch, self.Re, self.normalize, self.denormalize)

        loss += loss_ns
      
      return loss

    # Warm up learning rate
    if line_search == 'None':
      if warm_up:
        optimizer.param_groups[0]['lr'] = (self.post_lr - self.post_lr*self.optim_lr)/warm_up_epochs * (self.epoch - self.max_epochs) +self.post_lr*self.optim_lr

    # Get the current loss and gradient
    current_loss = self.compute_losses()
    grad = optimizer._gather_flat_grad()

    # Perform the two-loop recursion to compute the search direction
    p = optimizer.two_loop_recursion(-grad)


    # Perform the L-BFGS step
    if line_search in ['Armijo', 'Wolfe']:
      options = {'closure': closure, 'current_loss': current_loss}

    if line_search == 'Wolfe': 
      new_loss, grad, lr, _, _, _, _, _ = optimizer.step(p, grad, options=options)

    else: 
      if line_search == 'Armijo':
        new_loss, lr, _, _, _, _ = optimizer.step(p, grad, options=options)
      else:
        lr = optimizer.step(p, grad)
      new_loss = self.compute_losses()
      grad = optimizer._gather_flat_grad()
    
    # Update the curvature pairs with the new gradient
    optimizer.curvature_update(grad, eps=0.2, damping=True)


  def step_multibatch_LBFGS(self, optimizer) -> None:
    """
    Performs a single optimization step using a multi-batch L-BFGS approach with overlapping batches for data, 
    boundary conditions, and physics constraints.
    Args:
      optimizer: An L-BFGS optimizer instance with custom methods for two-loop recursion and curvature updates.
    Workflow:
      - Defines an overlap ratio for batches and splits each batch into main and overlapping parts.
      - Computes losses and gradients for each batch and its overlap for data, boundary, and physics terms.
      - Aggregates gradients and losses using a weighted combination that accounts for overlap.
      - Handles warm-up epochs with a custom learning rate schedule.
      - Optionally updates loss weights if automatic weighting is enabled.
      - Stores previous batch gradients and losses for use in the next step.
      - Updates the optimizer's curvature information for L-BFGS.
    Notes:
      - Assumes the existence of attributes such as N_DATA, N_BC, N_COLLOCATION, dataloader, bc_dataloader, 
        physics_dataloader, lambda_data, lambda_bc, lambda_ns, WeightUpdater, and others.
      - Requires that the optimizer supports custom methods: two_loop_recursion, step, and curvature_update.
    """

    # Define the overlap ratio for batch splitting
    overlap_ratio = 0.25                # should be in (0, 0.5)

    def split_data(data, overlap_size):
      batch_data = data[:-overlap_size, :]
      overlap_data = data[-overlap_size:, :]
      return batch_data, overlap_data

    # Combine the results (gradients / losses) from the main and overlapping batches
    def combine_result(prev, batch, next_):
      return overlap_ratio * (prev + next_) + (1 - 2*overlap_ratio) * batch
      
    weights = []

    grad_batch = []
    grad_next = []

    loss_batch = []
    loss_next = []

    first_call = not hasattr(self, 'total_grad_prev')
    warm_up_epochs = 50
    warm_up = self.epoch-1 > self.max_epochs and ((self.epoch-1 - self.max_epochs) <= warm_up_epochs)

    # split the batches into main and overlapping parts
    if not hasattr(self, 'overlap_size_data') and self.N_DATA > 0:
      self.overlap_size_data = int(self.data_batch_size * overlap_ratio)
      self.dataset.batch_size = self.data_batch_size - self.overlap_size_data
      self.dataset.l = self.dataset.__len__()      

    if not hasattr(self, 'overlap_size_bc') and self.N_BC > 0:
      self.overlap_size_bc = int(self.bc_batch_size * overlap_ratio)
      self.bc_dataset.batch_size = self.bc_batch_size - self.overlap_size_bc

    if not hasattr(self, 'overlap_size_physics') and self.N_COLLOCATION > 0:
      self.overlap_size_physics = int(self.physics_batch_size * overlap_ratio)
      self.physics_dataset.batch_size = self.physics_batch_size - self.overlap_size_physics

    
    # Get the current losses and gradients for each loss term
    # Data Loss
    if self.N_DATA > 0:
      x_data, y_data = next(iter(self.dataloader))
      x_data_batch, x_data_next = split_data(x_data[0], self.overlap_size_data)
      y_data_batch, y_data_next = split_data(y_data[0], self.overlap_size_data)      

      loss_data_batch, grad_data_batch = self.get_mse_loss(x_data_batch, y_data_batch, flat_grad = True)
      loss_data_next, grad_data_next = self.get_mse_loss(x_data_next, y_data_next, flat_grad = True)
      
      loss_batch.append(loss_data_batch)
      loss_next.append(loss_data_next)

      grad_batch.append(grad_data_batch)
      grad_next.append(grad_data_next)

      weights.append(self.lambda_data)

    # BC Loss
    if self.N_BC > 0:
      x_bc, y_bc = next(iter(self.bc_dataloader))
      x_bc_batch, x_bc_next = split_data(x_bc[0], self.overlap_size_bc)
      y_bc_batch, y_bc_next = split_data(y_bc[0], self.overlap_size_bc)

      loss_bc_batch, grad_bc_batch = self.get_mse_loss(x_bc_batch, y_bc_batch, flat_grad = True)
      loss_bc_next, grad_bc_next = self.get_mse_loss(x_bc_next, y_bc_next, flat_grad = True)

      loss_batch.append(loss_bc_batch)
      loss_next.append(loss_bc_next)

      grad_batch.append(grad_bc_batch)
      grad_next.append(grad_bc_next)

      weights.append(self.lambda_bc)


    # Physics Loss
    if self.N_COLLOCATION > 0:
      x_physics_next_list = []
      loss_ns_next, loss_ns_batch, grad_ns_next, grad_ns_batch = None, None, None, None

      # Accumulate gradients and losses for the physics batches (needed if batches are too small for valid curvature pairs)
      for i in range(self.n_acc_physics):
  
        x_physics = next(iter(self.physics_dataloader))[0].float()
        x_physics_batch, x_physics_next = split_data(x_physics, self.overlap_size_physics)
    
        x_physics_next_list.append(x_physics_next)

        loss_ns_batch_, grad_ns_batch_ = self.get_physics_loss(x_physics_batch, flat_grad = True)
        loss_ns_next_, grad_ns_next_ = self.get_physics_loss(x_physics_next, flat_grad = True)

        if loss_ns_batch == None:
          loss_ns_batch = loss_ns_batch_ / self.n_acc_physics
          loss_ns_next = loss_ns_next_ / self.n_acc_physics
          grad_ns_batch = grad_ns_batch_ / self.n_acc_physics
          grad_ns_next = grad_ns_next_ / self.n_acc_physics
        else:
          loss_ns_batch += loss_ns_batch_ / self.n_acc_physics
          loss_ns_next += loss_ns_next_ / self.n_acc_physics
          grad_ns_batch += grad_ns_batch_ / self.n_acc_physics
          grad_ns_next += grad_ns_next_ / self.n_acc_physics

      
      loss_batch.append(loss_ns_batch)
      loss_next.append(loss_ns_next)

      grad_batch.append(grad_ns_batch)
      grad_next.append(grad_ns_next)

      weights.append(self.lambda_ns)

    # Apply the current loss weights to the gradients
    total_grad_batch = self.WeightUpdater.new_grads(weights, grad_batch)
    total_grad_next = self.WeightUpdater.new_grads(weights, grad_next)


    #Combine the losses from the main and overlapping batches
    if hasattr(self, 'loss_data_prev'):
      self.loss_data = combine_result(self.loss_data_prev, loss_data_batch, loss_data_next)
    if hasattr(self, 'loss_bc_prev'):
      self.loss_bc = combine_result(self.loss_bc_prev, loss_bc_batch, loss_bc_next)
    if hasattr(self, 'loss_ns_prev'):
      self.loss_ns = combine_result(self.loss_ns_prev, loss_ns_batch, loss_ns_next)

    # Compute the total loss (for logging)
    self.loss_total = self.lambda_data * self.loss_data + self.lambda_bc * self.loss_bc +self.lambda_ns * self.loss_ns

    if not first_call:
      #Warm up learning rate
      if warm_up and optimizer.param_groups[0]['line_search'] == 'None':
        optimizer.param_groups[0]['lr'] = (self.post_lr - self.post_lr*self.optim_lr)/warm_up_epochs * (self.epoch-1 - self.max_epochs) +self.post_lr*self.optim_lr

      # Combine the gradients from the main and overlapping batches
      total_grad = combine_result(self.total_grad_prev, total_grad_batch, total_grad_next)

      # Perform the two-loop recursion to compute the search direction
      p = optimizer.two_loop_recursion(-total_grad)

      # Perform the L-BFGS step and storing the overlapping gradient for future curvature pair calculation
      lr = optimizer.step(p, g_Ok = total_grad_next, g_Sk = total_grad)
    else:
      optimizer.param_groups[0]['lr']=self.post_lr*self.optim_lr
      

    # Update the loss weights if auto-weighting is enabled
    if not warm_up:
      if self.autoweight and self.epoch % self.weight_update_freq == 0:
        weights = self.WeightUpdater.update(weights, grad_batch)
        self.lambda_data, self.lambda_bc, self.lambda_ns = weights

    # With the new applied parameters after the LBFGS step, compute the gradients for the "next" batch again 
    # and use as "prev" overlapping batch in the next iteration

    grad_prev = []

    #Compute gradients
    if self.N_DATA > 0:
      self.loss_data_prev, grad_data_prev = self.get_mse_loss(x_data_next, y_data_next, flat_grad = True)
      grad_prev.append(grad_data_prev)

    if self.N_BC > 0:
      self.loss_bc_prev, grad_bc_prev = self.get_mse_loss(x_bc_next, y_bc_next, flat_grad = True)
      grad_prev.append(grad_bc_prev)

    if self.N_COLLOCATION > 0:
      self.loss_ns_prev = None
      grad_ns_prev = None
      for i in range(len(x_physics_next_list)):
        loss_ns_prev_, grad_ns_prev_ = self.get_physics_loss(x_physics_next_list[i], flat_grad = True)

        if self.loss_ns_prev == None:
          self.loss_ns_prev = loss_ns_prev_ / self.n_acc_physics
          grad_ns_prev = grad_ns_prev_ / self.n_acc_physics
        else:
          self.loss_ns_prev += loss_ns_prev_ / self.n_acc_physics
          grad_ns_prev += grad_ns_prev_ / self.n_acc_physics

      grad_prev.append(grad_ns_prev)


    # Apply the current loss weights to the gradients
    self.total_grad_prev = self.WeightUpdater.new_grads(weights, grad_prev)

    # Compute the new curvature pairs for the next iteration using the new gradients and the previous gradients of the same iter
    if not first_call:
      optimizer.curvature_update(self.total_grad_prev, eps = 0.2, damping = True)


  def lbfgs_step(self, optimizer) -> None:
    """
    Performs a single optimization step using the specified L-BFGS type.
    Args:
      optimizer: The L-BFGS optimizer instance.

    This method checks the L-BFGS type and calls the appropriate step function for standard, full-overlap, or multi-batch L-BFGS.
    """

    if self.lbfgs_type == "standard":
      self.step_standard_LBFGS(optimizer)
    elif self.lbfgs_type == "full_overlap":
      self.step_full_overlap_LBFGS(optimizer)
    elif self.lbfgs_type == "multibatch":
      self.step_multibatch_LBFGS(optimizer)
      
  #MARK: Stepping

  def step(self) -> None:
    """
    Performs a single optimization step for the PINN model.
    This function computes the losses, updates the optimizer, and records the loss history.
    """
    self.inner_iter = 0
    start = time.time()
    if self.epoch <= self.max_epochs:
      if self.optim == "lbfgs":
        self.lbfgs_step(self.optimizer)
      else:
        # Compute losses and perform optimization step
        self.compute_losses()
        
        if self.amp_enabled:
          # Update optimizer with AMP scaling
          self.scaler.step(self.optimizer)
          self.scaler.update()
        else:
          # Update optimizer without AMP scaling
          self.optimizer.step()

    elif self.post_optimizer is not None:
      self.lbfgs_step(self.post_optimizer)

    step_time = time.time()-start
    
    if self.epoch % self.print_freq == 0:
      print(f'Step Time: {step_time}')

    self.update_all()
    self.epoch += 1


 
  #MARK: Train
  def train(self, epochs: int, print_freq: int = 100, plot_freq: int = 500, point_update_freq: int = None, point_keep_percentage: float = 20, autoweight_scheme = None, autoweight_freq = 10, save_freq: int = 50) -> None:
    """
    Trains the PINN model for a specified number of epochs.

    Args:
      epochs (int): Number of epochs to train the model.
      print_freq (int, optional): Frequency (in epochs) to print training progress. Defaults to 100.
      plot_freq (int, optional): Frequency (in epochs) to plot loss history and field. Defaults to 500.
      point_update_freq (int, optional): Frequency (in epochs) to update collocation and boundary points. Defaults to None (no update).
      point_keep_percentage (float, optional): Percentage of collocation points to keep when updating points. Defaults to 20.
      autoweight_scheme (str or None, optional): Scheme for automatic loss weighting. Defaults to None (no auto-weighting).
        - 1: ConflictFree
        - 2: Wang 2023 (gradient norms)
        - 3: Wang 2021 (Data loss weight is set to 1)
        - 4: Wang 2021 (PDE weight is set to 1)
      autoweight_freq (int, optional): Frequency (in epochs) to update loss weights if auto-weighting is enabled. Defaults to 10.
      save_freq (int, optional): Frequency (in epochs) to save model checkpoints. Defaults to 50.
    """
    # Set random seed for reproducibility
    self.best_loss = float('inf')
    self.max_epochs = epochs
    
    self.WeightUpdater = WeightUpdater(autoweight_scheme, alpha = 0.9, device=self.device)

    if autoweight_scheme is not None:
      self.autoweight = True
      self.autoweight_type = autoweight_scheme
      self.lambda_data = 1
      self.lambda_bc = 1
      self.lambda_ns = 1
      self.weight_update_freq = autoweight_freq


    # Set training parameters
    self.save_freq = save_freq
    self.print_freq = print_freq
    self.plot_freq = plot_freq
    self.point_update_freq = point_update_freq
    self.point_keep_percentage = point_keep_percentage

    self.norm_hist = []

    # Training loop
    for i in range(epochs+1):
      self.step()  # Perform a single optimization step

    if self.post_optimizer is not None:
      self.scheduler = None
      while self.epoch <= (epochs+self.post_epochs):
        self.step()

      
    self.plot_history()  # Plot the final loss history
    self.plot_field()  # Plot the final field
    self.save_hist(f"{self.NAME}_history.csv")

    self.save(f"{self.NAME}_final_state.pt")
      
    print("Training finished!")

    
  #MARK: Plot History
  def plot_history(self) -> None:
    """
    Plots the training history for the PINN model.

    This includes:
      - Total loss, data loss, boundary condition loss, and Navier-Stokes loss over epochs.
      - Learning rate history if a scheduler is used.
      - Loss weights (lambdas) if auto-weighting is enabled.
      - Test RMSE curves if available.
      - Maximum nondimensional pressure history if available.

    """
    # Stack the loss histories and move them to CPU
    total = torch.stack(self.hist_total, 0).cpu()
    dat = torch.stack(self.hist_data, 0).cpu()
    bc = torch.stack(self.hist_bc, 0).cpu()
    ns = torch.stack(self.hist_ns, 0).cpu()

    # Create a new figure and axis for the plot
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot()
    ax.set_title("Loss History")
    ax.set_xlabel("Iterations")
    ax.set_yscale("log")

    # Plot the loss histories
    plt.plot(total, label="Total")
    plt.plot(dat, label="Data")
    plt.plot(ns, label="NS")
    plt.plot(bc, label="BC")

    # Plot the learning rate history if a scheduler is used
    if self.scheduler is not None:
      plt.plot(self.hist_lr, label="lr")

    #Plot the lambda history
    if self.autoweight:
      if self.autoweight_type > 1:
        l1 = torch.stack(self.hist_l1, 0).cpu()
        if self.N_BC > 0: l2 = torch.stack(self.hist_l2, 0).cpu()
        l3 = torch.stack(self.hist_l3, 0).cpu()
        plt.plot(l1, label="λ_data", linestyle="--")
        if self.N_BC > 0: plt.plot(l2, label="λ_bc", linestyle="--") 
        plt.plot(l3, label="λ_ns", linestyle="--")

    # Add legend and save the plot
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.savefig(f"{self.NAME}_History.png")
    plt.close()

    if len(self.hist_RMSE_test_u) > 0:
      x = np.arange(0, len(self.hist_RMSE_test_u), 1) * self.print_freq

      fig2 = plt.figure(figsize = [10,8])
      plt.plot(x,self.hist_RMSE_test_u, label = "u")
      plt.plot(x,self.hist_RMSE_test_v, label = "v")
      plt.plot(x,self.hist_RMSE_test_w, label = "w")
      plt.title("Test Data Error over Iterations")
      plt.xlabel("Iterations")
      plt.ylabel("RMSE_Test (m/s)")
      plt.yscale("log")
      plt.legend()
      plt.savefig(f"{self.NAME}_RMSE.png")
      plt.close()

    if len(self.p_hist) > 0:
      fig3 = plt.figure(figsize = [10,8])
      plt.plot(self.p_hist)
      plt.title("Maximum Nondimensional Pressure over Iterations")
      plt.xlabel("Iterations")
      plt.ylabel("p*_max (-)")
      #plt.yscale("log")
      #plt.legend()
      plt.savefig(f"{self.NAME}_p_max.png")
      plt.close()

     

  #MARK: Plot Field
  def plot_field(self) -> None:
    """
    Plots the field for the PINN model using the specified plot setups.
    This function saves the model, loads it on the CPU, and generates 2D plots based on the provided configurations.
    """

    # Generate 2D plots based on the provided plot setups
    for setup in self.plot_setups:
      if setup['type'] == '2D':
        plotPINN_2D(
          self.get_forward_callable(),
          plot_dims=setup['plot_dims'],
          dim3_slice=setup['dim3_slice'],
          t_slice=setup['t_slice'],
          data1=self.data if setup['plot_data'] else None,
          component1=setup['component1'],
          component2=setup['component2'],
          lb=setup['lb'],
          ub=setup['ub'],
          resolution=setup['resolution'],
          dim3_tolerance=setup['dim3_tolerance'],
          centered1=setup['centered1'],
          vmin1=setup['vmin1'],
          vmax1=setup['vmax1'],
          centered2=setup['centered2'],
          vmin2=setup['vmin2'],
          vmax2=setup['vmax2']
        )


  #MARK: Save History
  def save_hist(self, path: str) -> None:
    """
    Saves the loss history to a specified path.

    Args:
      path (str): Path to save the loss history.
    """
    hist = torch.tensor(self.hist_total)
    epochs = torch.arange(len(self.hist_total))
    hist = torch.vstack((epochs, hist))
    header = 'Epoch,Total'

    if len(self.hist_ns) == len(self.hist_total):
        hist = torch.vstack((hist, torch.tensor(self.hist_ns)))
        header = f"{header},NS"

    if len(self.hist_bc) == len(self.hist_total):
        hist = torch.vstack((hist, torch.tensor(self.hist_bc)))
        header = f"{header},BC"

    if len(self.hist_data) == len(self.hist_total):
        hist = torch.vstack((hist, torch.tensor(self.hist_data)))
        header = f"{header},Data"

    if len(self.hist_lr) == len(self.hist_total):
        hist = torch.vstack((hist, torch.squeeze(torch.tensor(self.hist_lr))))
        header = f"{header},LR"

    hist = hist.detach().numpy().T
    np.savetxt(path, hist, header=header, delimiter=",", comments="")

    if self.X_train is not None:
      epochs = torch.arange(len(self.hist_RMSE_test_u))*self.print_freq
      hist = torch.vstack((epochs, torch.tensor(self.hist_RMSE_test_u), torch.tensor(self.hist_RMSE_test_v), torch.tensor(self.hist_RMSE_test_w)))
      header = 'Epoch,RMSE_test_u,RMSE_test_v,RMSE_test_w'
      hist = hist.detach().numpy().T
      np.savetxt(path.replace('.csv', '_rmse.csv'), hist, header=header, delimiter=",", comments="")

  #Mark Save & load state dict
  def save(self, path: str) -> None:
    """
    Saves the trained model to the specified path.

    Args:
      path (str): Path to save the trained model.
    """
    save_predictable(self, path)

  def load(self, path: str) -> None:
    """
    Loads a trained model from the specified path.

    Args:
      path (str): Path to load the trained model.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    self.l_scale = checkpoint['l_scale']
    self.u_scale = checkpoint['u_scale']
    self.t_scale = checkpoint['t_scale']
    self.p_scale = checkpoint['p_scale']
    self.norm_scales = checkpoint['norm_scales']
    self.norm_offsets = checkpoint['norm_offsets']
    self.plot_setups = checkpoint['plot_setups']

    if isinstance(self.model, DataParallel):
      self.model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
      self.model.load_state_dict(checkpoint['model_state_dict'])

    self.model_cpu = copy.deepcopy(self.model)
    self.model_cpu.to('cpu')

#MARK: ---- END of PINN ---






#MARK: Load & Save trained models
def save_predictable(PINN: PINN_3D, path: str) -> None:
  """
  Saves the PINN model to the specified path. Doesn't work with SDF yet.

  Args:
    model (PINN): The PINN model to save.
    path (str): Path to save the PINN model.
  """

  if isinstance(PINN.model, DataParallel):
    state_dict = PINN.model.module.state_dict()
  else:
    state_dict = PINN.model.state_dict()

  torch.save({
              'model_state_dict': state_dict,
              'norm_scales': PINN.norm_scales,
              'norm_offsets': PINN.norm_offsets,
              'NAME': PINN.NAME,
              'Re': PINN.Re,
              'N_NEURONS': PINN.N_NEURONS,
              'N_LAYERS': PINN.N_LAYERS,
              'RFF': PINN.fourier_feature,
              'mapping_size': PINN.mapping_size if hasattr(PINN, 'mapping_size') else None,
              'l_scale': PINN.l_scale,
              'u_scale': PINN.u_scale,
              't_scale': PINN.t_scale,
              'p_scale': PINN.p_scale,
              'plot_setups': PINN.plot_setups
              }, path)
  

def load_predictable(path: str) -> PINN_3D:
  """
  Loads a saved PINN_3D model from the specified path.

  Args:
    path (str): Path to the saved model file.

  Returns:
    PINN_3D: The loaded PINN_3D model instance with weights and configuration restored.
  """
  checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
  
  PINN = PINN_3D(
    model=FCN,
    NAME = checkpoint['NAME'],
    Re = checkpoint['Re'],
    N_NEURONS = checkpoint['N_NEURONS'],
    N_LAYERS = checkpoint['N_LAYERS'],
    fourier_feature = checkpoint['RFF'],
    mapping_size = checkpoint['mapping_size'] if 'mapping_size' in checkpoint else None,
  )

  PINN.l_scale = checkpoint['l_scale']
  PINN.u_scale = checkpoint['u_scale']
  PINN.t_scale = checkpoint['t_scale']
  PINN.p_scale = checkpoint['p_scale']
  PINN.norm_scales = checkpoint['norm_scales']
  PINN.norm_offsets = checkpoint['norm_offsets']
  PINN.plot_setups = checkpoint['plot_setups']
  
  if isinstance(PINN.model, DataParallel):
    PINN.model.module.load_state_dict(checkpoint['model_state_dict'])
  else:
    PINN.model.load_state_dict(checkpoint['model_state_dict'])

  PINN.model_cpu = copy.deepcopy(PINN.model)
  PINN.model_cpu.to('cpu')

  return PINN



#MARK: Custom Dataset
class CustomDataset(Dataset):
  def __init__(self, Xdata: torch.Tensor, Ydata: torch.Tensor = None, device = None, batch_size: int = None, shuffle: bool = False, min_size: int = None):
    """
    Custom dataset for batching and shuffling data for PINN training.

    Args:
      Xdata (torch.Tensor): Input data tensor.
      Ydata (torch.Tensor, optional): Output/target data tensor. Defaults to None.
      device (str or torch.device, optional): Device to move data to. Defaults to CUDA if available.
      batch_size (int, optional): Batch size for data loading. Defaults to all data.
      shuffle (bool, optional): Whether to shuffle data at the start of each epoch. Defaults to False.
      min_size (int, optional): Minimum size for the last batch. Defaults to 512 or batch_size if shuffle is True.
    """
    self.Xdata = Xdata
    self.Ydata = Ydata

    # Set batch size
    self.batch_size = batch_size if batch_size is not None else len(Xdata[:, 0])
    self.shuffle = shuffle

    # Current batch index
    self.current_idx = 0
    
    # Set minimum batch size for the last batch
    if self.shuffle == True and min_size == None:
      self.min_size = batch_size
    elif min_size is not None:
      self.min_size = min_size
    else:
      self.min_size = 512

    # Whether to automatically step to next batch after each __getitem__ call
    self.autostepping = True
      
    # Number of batches in the dataset
    self.l = self.__len__()

    # Set device
    if device is not None:
      self.device = device
    else:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def __len__(self) -> int:
    """
    Returns the number of batches in the dataset.

    Returns:
      int: Number of batches.
    """
    l = len(self.Xdata[:, 0]) // self.batch_size

    # Add one more batch if the remainder is at least min_size
    if len(self.Xdata[:, 0]) % self.batch_size >= self.min_size:
      l += 1

    return l

  def __manual_step(self):
    """
    Manually step to the next batch index.
    """
    if self.current_idx == self.l-1:
      self.current_idx = 0
    else:
      self.current_idx += 1

  def __getitem__(self, dummy_idx: int, step = True) -> torch.Tensor:
    """
    Returns the next batch of data (optionally shuffling at the start of each epoch).

    Args:
      dummy_idx (int): Dummy index (ignored, for compatibility with DataLoader).
      step (bool, optional): Whether to step to the next batch after this call. Defaults to True.

    Returns:
      tuple(torch.Tensor, torch.Tensor) or torch.Tensor: Batch of input (and output) data.
    """
    idx = self.current_idx

    # Shuffle data at the start of each epoch
    if idx == 0 and self.shuffle:
      indices = torch.randperm(len(self.Xdata[:, 0]))
      self.Xdata = self.Xdata[indices, :]
      if self.Ydata is not None:
        self.Ydata = self.Ydata[indices, :]

    # Compute batch slice indices
    idx_0 = idx * self.batch_size
    idx_1 = (idx + 1) * self.batch_size

    # Adjust for last batch if it is smaller
    if idx_1 > len(self.Xdata[:, 0]):
      idx_1 = len(self.Xdata[:, 0])

    # Step to next batch if enabled
    if step and self.autostepping:
      if idx == self.l-1:
        self.current_idx = 0
      else:
        self.current_idx += 1  

    # Return batch (with or without targets)
    if self.Ydata is not None:
      return self.Xdata[idx_0:idx_1, :].to(self.device).float(), self.Ydata[idx_0:idx_1, :].to(self.device).float()
    else:
      return self.Xdata[idx_0:idx_1, :].to(self.device).float()


#MARK: Physics Point Sampling
def sample_interior_points(nb: int, lb: np.ndarray, ub: np.ndarray, geometry: callable = None) -> np.ndarray:
  """
  Samples interior points within the specified domain using Latin Hypercube Sampling (LHS).

  Args:
    nb (int): Number of points to sample.
    lb (np.ndarray): Lower bounds of the domain.
    ub (np.ndarray): Upper bounds of the domain.
    geometry (callable, optional): Function to define the geometry of the domain. Should return True if a point is inside the domain and False if a point is outside the domain. Defaults to None.

  Returns:
    np.ndarray: Array of sampled points within the domain.
  """

  # Initialize Latin Hypercube Sampler
  sampler = qmc.LatinHypercube(d=4)
  
  # Generate initial sample
  sample = sampler.random(n=nb)
  sample = qmc.scale(sample, l_bounds=lb, u_bounds=ub)

  # Check if points are inside the geometry and resample if necessary
  if geometry is not None:
    insiders = geometry(sample[:, 0], sample[:, 1], sample[:, 2])
    while any(insiders):
      new_sample = sampler.random(n=sum(insiders))
      sample[insiders, :] = qmc.scale(new_sample, l_bounds=lb, u_bounds=ub)
      insiders = geometry(sample[:, 0], sample[:, 1], sample[:, 2])

  return sample




