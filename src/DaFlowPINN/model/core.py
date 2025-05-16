import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from typing import List, Tuple, Union

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
  def __init__(self,
               model: nn.Module,
               N_LAYERS: int = 5,
               N_NEURONS: int = 512,
               hardBC_sdf: callable = None,
               fourier_feature: bool = False,
               **kwargs):

    super().__init__()
    
    if fourier_feature:
      self.fourier_feature = True
      mapping_size = kwargs.get('mapping_size', 512)
      self.mapping_size = mapping_size if mapping_size is not None else 512
      scale_x = kwargs.get('scale_x', 1)
      scale_y = kwargs.get('scale_y', 1)
      scale_z = kwargs.get('scale_z', 1)
      scale_t = kwargs.get('scale_t', 1)
      self.network = model(2*self.mapping_size, 4, N_NEURONS, N_LAYERS)
      self.rff = RFF(4, n_freqs=self.mapping_size, scales=[scale_x, scale_y, scale_z, scale_t])
    else:
      self.fourier_feature = False
      self.network = model(4, 4, N_NEURONS, N_LAYERS)

    if hardBC_sdf is not None:
      self.hardBC = True
      self.hbc = HardBC(hardBC_sdf)
    else:
      self.hardBC = False

  def forward(self,x):
    x = x.float()
    if self.hardBC:
      x0 = x.detach()
    if self.fourier_feature:
      x = self.rff(x)
    x = self.network(x)
    if self.hardBC:
      x = self.hbc(x,x0)

    return x

#MARK: --- Begin of PINN ---
class PINN_3D(nn.Module):
  """
  Physics-Informed Neural Network (PINN) for 3D problems.
  This class implements a PINN for solving 3D partial differential equations (PDEs) using neural networks. 
  It supports boundary conditions, collocation points for physics-based losses, and data points for supervised learning.
  Attributes:
    model (nn.Module): The neural network model.
    NAME (str): The name of the model.
    Re (float): Reynolds number for non-dimensionalization.
    N_LAYERS (int): Number of layers in the neural network.
    N_NEURONS (int): Number of neurons per layer in the neural network.
    amp_enabled (bool): Flag to enable automatic mixed precision (AMP) training.
    device (torch.device): The device to run the model on (CPU or GPU).
    N_COLLOCATION (int): Number of collocation points.
    N_BC (int): Number of boundary condition points.
    N_DATA (int): Number of data points.
    point_update_freq (int): Frequency of updating points.
    collocation_growth (bool): Flag to enable collocation growth.
    epoch (int): Current training epoch.
    model_cpu (nn.Module): Model for CPU inference.
    hist_total (list): History of total losses.
    hist_data (list): History of data losses.
    hist_ns (list): History of Navier-Stokes losses.
    hist_bc (list): History of boundary condition losses.
    hist_lr (list): History of learning rates.
    plot_setups (list): List of plot setups.
    l_scale (float): Length scale for non-dimensionalization.
    u_scale (float): Velocity scale for non-dimensionalization.
    t_scale (float): Time scale for non-dimensionalization.
    p_scale (float): Pressure scale for non-dimensionalization.
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
      model (nn.Module): The neural network model.
      NAME (str): The name of the model.
      Re (float): Reynolds number for non-dimensionalization.
      N_LAYERS (int, optional): Number of layers in the neural network. Defaults to 4.
      N_NEURONS (int, optional): Number of neurons per layer in the neural network. Defaults to 256.
      amp_enabled (bool, optional): Flag to enable automatic mixed precision (AMP) training. Defaults to False.
    """
    torch.manual_seed(123)
    # Set default data type to float
    torch.set_default_dtype(torch.float)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(123)
    super().__init__()
    

    self.model = Network(model, N_LAYERS, N_NEURONS, hardBC_sdf, fourier_feature, **kwargs)


    self.fourier_feature = fourier_feature
    if self.fourier_feature:
      self.mapping_size = kwargs.get('mapping_size', 256)
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

    # Initialize history lists for losses and learning rates
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
    #self.model.Re = torch.nn.Parameter(torch.tensor(Re, device=self.device, dtype=torch.float))
    self.l_scale = 1
    self.u_scale = 1
    self.t_scale = 1
    self.p_scale = 1

    self.pmin = np.inf
    self.pmax = -np.inf

    # Initialize normalization scales and offsets
    self.norm_scales = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], device=self.device)
    self.norm_offsets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], device=self.device)

    self.p_ref = None

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
    x_dimless = self.dimensionless(x)
    
    # Pass the dimensionless input through the model
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
    y = self.model_cpu(self.normalize(self.dimensionless(x)))
    
    # Redimensionalize the model output
    return self.redimension(y=self.denormalize(y=y))
  
  def get_forward_callable(self) -> torch.Tensor:
    
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

  def add_p_ref_point(self, x: float, y: float, z: float, t: float, p: float = 0) -> None:
    """
    Adds a reference point for pressure to the PINN model.

    Args:
      x (float): x-coordinate of the reference point.
      y (float): y-coordinate of the reference point.
      z (float): z-coordinate of the reference point.
      t (float): Time of the reference point.
      p (float, optional): Pressure value at the reference point. Defaults to 0.
    """
    self.p_ref = torch.empty(1, 5, device=self.device)
    self.p_ref[0] = torch.tensor([x, y, z, t, p], device=self.device).float()


#MARK: Dimensionless

  def set_dimensionless(self, l_scale: float, u_scale: float, p_scale: float, t_scale: float = None) -> None:
    """
    Sets the dimensionless scales for the PINN model. Call before set_normalization!!!

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
  def add_data_points(self, data: np.ndarray, batch_size = None, test_size = 0.05) -> None:
    """
    Adds data points for supervised learning to the PINN model.

    Args:
      data (np.ndarray): Array containing the data points. The array should have the following columns:
      - Columns 1-4: Input features (x, y, z, t)
      - Columns 5-7: Output features (u, v, w)
    """
    if data.shape[1] < 8:
      raise ValueError("Data array must have at least 8 columns: [index, x, y, z, t, u, v, w]")

    self.data = data
    self.test_size = test_size

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
                                     1], device=self.device) #/2

    self.norm_offsets = torch.tensor([X[:, 0].min(),
                                      X[:, 1].min(),
                                      X[:, 2].min(),
                                      X[:, 3].min(),
                                      Y[:, 0].min(),
                                      Y[:, 1].min(),
                                      Y[:, 2].min(),
                                      0], device=self.device) #+ 1 * self.norm_scales

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

    loss_scale_u = 1
    loss_scale_v = self.norm_scales[5]/self.norm_scales[4]
    loss_scale_w = self.norm_scales[6]/self.norm_scales[4]

    
    self.loss_scales = torch.tensor([loss_scale_u, loss_scale_v, loss_scale_w])

    if hasattr(self.model, "rff"):      
      self.model.rff.scales = nn.Parameter(self.model.rff.scales * self.norm_scales[0:4], requires_grad=False)

    print(f"Normalization scales: {self.norm_scales}") #print to rescalculate fourier feature scales
    #print(f"New RFF Scales: {self.model.scales}")

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

      np.savez("DA_CASE01_t_23_25_p200.npz",
      x_train = self.redimension(self.denormalize(self.X_train)).cpu().numpy(),
      y_train = self.redimension(y=self.denormalize(y=self.Y_train)).cpu().numpy(),
      x_test = self.redimension(self.denormalize(self.X_test)).cpu().numpy(),
      y_test = self.redimension(y=self.denormalize(y=self.Y_test)).cpu().numpy())
      print("done...")
    
    else:
      self.X_train = X.to(self.device)
      self.Y_train = Y.to(self.device)
      self.X_test = None
      self.Y_test = None

    self.N_DATA = len(self.X_train[:, 0])


    #for tests:

    # if self.N_DATA > 300000:
    #   name = "200"
    # elif self.N_DATA > 75000:
    #   name = "050"
    # elif self.N_DATA > 15000:
    #   name = "010"
    # else:
    #   name = "001"  

    # data = np.load(f"/scratch/jpelz/ma-pinns/DA_CASE01_t_23_25_p{name}.npz")

    # x_train = data["x_train"]
    # y_train = data["y_train"]
    # x_test = data["x_test"]
    # y_test = data["y_test"]

    # self.X_train = torch.tensor(x_train, device = self.device)
    # self.Y_train = torch.tensor(y_train, device = self.device)
    # self.X_test = torch.tensor(x_test, device = self.device)
    # self.Y_test = torch.tensor(y_test, device = self.device)

    # self.X_train, self.Y_train = self.dimensionless(self.X_train, self.Y_train)
    # self.X_test, self.Y_test = self.dimensionless(self.X_test, self.Y_test)

    # self.X_train, self.Y_train = self.normalize(self.X_train, self.Y_train)
    # self.X_test, self.Y_test = self.normalize(self.X_test, self.Y_test)



    self.N_DATA = len(self.X_train[:, 0])
    
    self.data_batch_size = batch_size if batch_size is not None else self.N_DATA

  
    self.dataset = CustomDataset(self.X_train, self.Y_train, batch_size=self.data_batch_size, shuffle=True)
    self.dataloader = DataLoader(self.dataset, batch_size=1)

#MARK: Boundary Conditions
  def add_boundary_condition(self, surface_sampler: callable, N_BC_POINTS: int, weight: float, values = [0, 0, 0], batch_size = None) -> None:
    """
    Adds boundary condition points to the PINN model.

    Args:
      surface_sampler (callable): Function to sample points on the boundary surface.
      N_BC_POINTS (int): Number of boundary condition points.
      weight (float): Weight for the boundary condition loss.
    """

    if self.N_DATA == 0:
      raise RuntimeError("Data points must be added before boundary condition points.")
    
    self.surface_sampler = surface_sampler
    boundary = self.surface_sampler(N_BC_POINTS)

    # Convert boundary points to tensor and enable gradient computation
    X_boundary = torch.from_numpy(boundary).float()
    Y_boundary = torch.ones_like(X_boundary[:,:3]) * torch.tensor(values).float()

    X_boundary, Y_boundary = self.dimensionless(X_boundary, Y_boundary)
    X_boundary, Y_boundary = self.normalize(X_boundary, Y_boundary)

    # Set the number of boundary condition points and the loss weight
    self.N_BC = N_BC_POINTS
    self.lambda_bc = weight
    self.bc_values = values
    self.bc_batch_size = batch_size if batch_size is not None else self.N_BC

    self.bc_dataset = CustomDataset(X_boundary, Y_boundary, batch_size=self.bc_batch_size, shuffle=True)
    self.bc_dataloader = DataLoader(self.bc_dataset, batch_size=1)

#MARK: Physics Points
  def add_physics_points(self, N_COLLOCATION: int, batch_size: int, geometry: callable = None, weight: float = 1.0, keep_percentage: float = 20, numerical = False, p_scale = 1, p_offset = 0, n_acc = 1) -> None:
    """
    Adds physics collocation points to the PINN model. Overwrites existing points.

    Args:
      N_COLLOCATION (int): Number of collocation points.
      N_BATCHES (int): Number of batches for the DataLoader.
      geometry (callable, optional): Function to define the geometry of the domain. Should return True if a point is inside the domain and false a a point is outside of the domain. Defaults to None.
      weight (float, optional): Weight for the physics loss. Defaults to 1.0.
      keep_percentage (float, optional): Percentage of the collocation points to keep if points already exist. Defaults to 20.
    """
    if self.N_DATA == 0:
      raise RuntimeError("Data points must be added before physics points.")

    current_nr_points = self.N_COLLOCATION
    self.norm_scales[7] = p_scale#/2
    self.norm_offsets[7] = 0 #p_offset + 1*p_scale

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
        X_physics = torch.cat((keep_points, new_points), dim=0)
      
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

  def update_points(self, keep_percentage) -> None:
    """
    Updates the collocation and boundary points for the PINN model.
    This function is called periodically during training to refresh the collocation and boundary points.
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
    Enables collocation growth during training.

    Args:
      n_init (int): Initial number of collocation points.
      n_max (int): Maximum number of collocation points.
      epoch_start (int): Epoch to start increasing collocation points.
      epoch_end (int): Epoch to stop increasing collocation points.
      increase_scheme (str, optional): Scheme to increase collocation points. Defaults to 'linear'.
        - 'linear': Linear increase.
        - 'exponential': Exponential increase. Requires 'epsilon' in kwargs (difference ratio between given and actual end-number of points.).
        - 'logarithmic': Logarithmic increase.
      **kwargs: Additional arguments for specific increase schemes.
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
  def add_optimizer(self, optim: str = "adam", lr: float = 1e-3, lbfgs_type = "standard") -> None:
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
      - "full_overlap": Full overlap L-BFGS optimizer
      - "multibatch": Multibatch L-BFGS optimizer with partial overlap
    Raises:
      ValueError: If an invalid optimizer is specified.
    """
    self.optim_lr=lr
    self.optim = optim
    if optim == "adam":
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) #betas=(.95, .95)
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
    
  def add_postTraining(self, lr: float = 1, epochs: int = 1000, lbfgs_type = "standard") -> None:
    """Adds an additional training stage using L-BFGS optimizer."""
    if lbfgs_type == "standard":
      self.post_optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
    elif lbfgs_type == "full_overlap": 
      self.post_optimizer = LBFGS(self.model.parameters(), lr=lr, line_search='None')
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

  def add_scheduler(self, scheduler) -> None:
    """
    Adds a learning rate scheduler to the PINN model.

    Args:
      scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use.
    """
    self.scheduler = scheduler


#MARK: Prepare Plots
  def add_2D_plot(self, plot_dims: Tuple[int, int], dim3_slice: float, t_slice: float, plot_data: bool = False, component1: int = 0, centered1: bool = False, vmin1: float = None, vmax1: float = None,
                  component2: int = 4, centered2: bool = False, vmin2: float = None, vmax2: float = None,
                  lb: np.ndarray = None, ub: np.ndarray = None, resolution: List[int] = None, dim3_tolerance: float = None) -> None:
    """
    Adds a 2D plot setup to the PINN model.

    Args:
      plot_dims (Tuple[int, int]): Dimensions to plot (e.g., (0, 1) for x-y plane).
      dim3_slice (float): Slice value for the third dimension.
      t_slice (float): Slice value for the time dimension.
      plot_data (bool, optional): Flag to plot data points. Defaults to False.
      component (int, optional): Component to plot (e.g., 0 for mag, 1 for u, ... -> see list). Defaults to 0.
      lb (np.ndarray, optional): Lower bounds for the plot. Defaults to model's lower bounds.
      ub (np.ndarray, optional): Upper bounds for the plot. Defaults to model's upper bounds.
      resolution (List[int], optional): Resolution of the plot. Defaults to [100, 100].
      dim3_tolerance (float, optional): Tolerance for the third dimension slice. Defaults to 5% of the model's third dimension range.

    Components:
      0: Magnitude of velocity
      1: u component of velocity
      2: v component of velocity
      3: w component of velocity
      4: Pressure (p)
      5: du1/dx
      6: du1/dy
      7: du2/dx
      8: du2/dy
      9: Vorticity
      NYI: Q-criterion

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
      # print(f"      RMSE (u,v,w) in %: {rel_rmse}")

      if self.X_test is not None:
        abs_rmse, rel_rmse, abs_mae, rel_mae = detailed_data_err(self.model, self.X_test, self.Y_test, self.denormalize, self.redimension)
        print("    Detailed Data Errors (Test):")
        print(f"      MAE (u,v,w) in m/s: {abs_mae}")
        print(f"      MAE (u,v,w) in %: {rel_mae}")
        print(f"      RMSE (u,v,w) in m/s: {abs_rmse}")
        print(f"      RMSE (u,v,w) in %: {rel_rmse}")

        self.hist_RMSE_test_u.append(abs_rmse[0])
        self.hist_RMSE_test_v.append(abs_rmse[1])
        self.hist_RMSE_test_w.append(abs_rmse[2])


      
    # print(f'Norm = {self.norm_hist[-1]}')
    # print(f'Mean norm = {torch.mean(torch.stack(self.norm_hist))}')


  #MARK: Loss Computation


  def get_mse_loss(self, x, y, flat_grad = False):
    with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.amp_enabled):
      self.optimizer.zero_grad()

      loss_data = scaled_mse_loss(self.model, x, y, self.loss_scales)
      if self.amp_enabled:
        scaled_loss_data = self.scaler.scale(loss_data)
        scaled_loss_data.backward()
      else:
        loss_data.backward()

      grad = get_gradient_vector(self.model, flat = flat_grad)

      return loss_data.detach(), grad
    
  def get_physics_loss(self, x, flat_grad = False):
    with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.amp_enabled):
      self.optimizer.zero_grad()

      if self.numerical:
        loss_ns = numerical_physics_loss(self.model, x, self.Re, 0.0025, 0.0025, 0.0025, 2.0E-3, self.denormalize)
      else:
        loss_ns, pmax, pmin = physics_loss(self.model, x, self.Re, self.normalize, self.denormalize, True)
        loss_ns = loss_ns

        self.p_hist.append(pmax.cpu())

      
      
      if self.amp_enabled:
        scaled_loss_ns = self.scaler.scale(loss_ns)
        scaled_loss_ns.backward(retain_graph=True)
      else:
        loss_ns.backward()

      grad = get_gradient_vector(self.model, flat = flat_grad)
      
      #self.norm_scales[7]=abs(pmax-pmin)
      #self.norm_offsets[7]=pmin

      #lr = 5E-2
      #if pmax > 1: self.norm_scales[7]*=(1+lr)
      #else: self.norm_scales[7]*=(1-lr)

      #if pmin > 0: self.norm_scales[7]*=(1+lr)
      #else: self.norm_scales[7]*=(1-lr)


      return loss_ns.detach(), grad
    


  def compute_losses(self) -> torch.Tensor:
    """
    Computes the losses for the PINN model, including data loss, boundary condition loss, and physics loss.
    The losses are computed and backpropagated with optional automatic mixed precision (AMP) scaling.

    Returns:
      torch.Tensor: The total loss.
    """
    # Zero the gradients of the optimizer
    self.optimizer.zero_grad()
    #torch.autograd.set_detect_anomaly(True)

    grads =[]
    weights = []

    
    # Compute data loss if data points are available
    if self.N_DATA > 0:
      self.loss_data = 0
      grad_data = 0
      n_batches = len(self.dataloader) if self.grad_acc else 1
      for idx in range(n_batches):
        x, y = next(iter(self.dataloader))

        loss, grad = self.get_mse_loss(x[0], y[0])
        
        self.loss_data += loss / n_batches
        grad_data += grad / n_batches

      grads.append(grad_data)
      weights.append(self.lambda_data)
      #self.optimizer.step()

      # Compute boundary condition loss if boundary points are available
      if self.N_BC > 0:
        self.loss_bc = 0
        grad_bc = 0
        n_batches = len(self.bc_dataloader) if self.grad_acc else 1
        for idx in range(n_batches):
          x_b, y_b = next(iter(self.bc_dataloader))

          loss, grad = self.get_mse_loss(x_b[0], y_b[0])

          self.loss_bc += loss / n_batches
          grad_bc += grad / n_batches

        
        grads.append(grad_bc)
        weights.append(self.lambda_bc)
        #self.optimizer2.step()

      # Compute physics loss if collocation points are available
      if self.N_COLLOCATION > 0:
        self.loss_ns = 0
        grad_ns = 0
        n_batches = len(self.physics_dataloader) if self.grad_acc else 1
        for idx in range(n_batches):
          batch = next(iter(self.physics_dataloader))[0]

          loss, grad = self.get_physics_loss(batch)
          
          self.loss_ns += loss / n_batches
          grad_ns += grad / n_batches
          

        grads.append(grad_ns)
        weights.append(self.lambda_ns)
        #self.optimizer3.step()


      if self.autoweight and (self.epoch % self.weight_update_freq == 0) and (self.inner_iter == 0):
        weights = self.WeightUpdater.update(weights, grads)

        if not self.N_BC == 0:
          self.lambda_data, self.lambda_bc, self.lambda_ns = weights
        else:
          self.lambda_data, self.lambda_ns = weights

      
      new_grads = self.WeightUpdater.new_grads(weights, grads)
      apply_gradient_vector(self.model, new_grads)

      self.inner_iter += 1
    

      # Calculate the total loss
      self.loss_total = self.lambda_data*self.loss_data + self.lambda_bc*self.loss_bc + self.lambda_ns*self.loss_ns

      return self.loss_total


  #MARK: Output Updating
  def update_all(self):

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

    
    self.hist_l1.append(self.lambda_data)
    self.hist_l2.append(self.lambda_bc)
    self.hist_l3.append(self.lambda_ns)
    
    # Record learning rate history if scheduler is available
    if self.scheduler is not None:
      self.hist_lr.append(self.scheduler.get_last_lr())
    # else:
    #   self.hist_lr.append(self.optim_lr)
    
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
      #self.plot_history()
      self.plot_field()



  
  #MARK: LBFGS - Methods
  def step_standard_LBFGS(self, optimizer) -> None:

    update_batches_in_inner_iters = True

    if not update_batches_in_inner_iters:
      if self.N_DATA>0: self.dataset.autostepping = False
      if self.N_BC>0: self.bc_dataset.autostepping = False
      if self.N_COLLOCATION>0: self.physics_dataset.autostepping = False
      self.inner_iter = 0

    

    optimizer.step(self.compute_losses)


    if not update_batches_in_inner_iters:
      if self.N_DATA>0: self.dataset._manual_step()
      if self.N_BC>0: self.bc_dataset._manual_step()
      if self.N_COLLOCATION>0: self.physics_dataset._manual_step()
    
  
  
  def step_full_overlap_LBFGS(self, optimizer) -> None:
    
    line_search = optimizer.param_groups[0]['line_search']
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

    #Warm up learning rate

    if line_search == 'None':
      if warm_up:
        optimizer.param_groups[0]['lr'] = (self.post_lr - self.post_lr*self.optim_lr)/warm_up_epochs * (self.epoch - self.max_epochs) +self.post_lr*self.optim_lr


    current_loss = self.compute_losses()
    grad = optimizer._gather_flat_grad()
    p = optimizer.two_loop_recursion(-grad)

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
    
    optimizer.curvature_update(grad, eps=0.2, damping=True)


  def step_multibatch_LBFGS(self, optimizer) -> None:
    overlap_ratio = 0.25                # should be in (0, 0.5)

    def split_data(data, overlap_size):
      batch_data = data[:-overlap_size, :]
      overlap_data = data[-overlap_size:, :]
      return batch_data, overlap_data

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

    if self.N_COLLOCATION > 0:
      x_physics_next_list = []
      loss_ns_next, loss_ns_batch, grad_ns_next, grad_ns_batch = None, None, None, None

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

    total_grad_batch = self.WeightUpdater.new_grads(weights, grad_batch)
    total_grad_next = self.WeightUpdater.new_grads(weights, grad_next)

    if hasattr(self, 'loss_data_prev'):
      self.loss_data = combine_result(self.loss_data_prev, loss_data_batch, loss_data_next)
    if hasattr(self, 'loss_bc_prev'):
      self.loss_bc = combine_result(self.loss_bc_prev, loss_bc_batch, loss_bc_next)
    if hasattr(self, 'loss_ns_prev'):
      self.loss_ns = combine_result(self.loss_ns_prev, loss_ns_batch, loss_ns_next)

    self.loss_total = self.lambda_data * self.loss_data + self.lambda_bc * self.loss_bc +self.lambda_ns * self.loss_ns

    if not first_call:
      #Warm up learning rate
      if warm_up and optimizer.param_groups[0]['line_search'] == 'None':
        optimizer.param_groups[0]['lr'] = (self.post_lr - self.post_lr*self.optim_lr)/warm_up_epochs * (self.epoch-1 - self.max_epochs) +self.post_lr*self.optim_lr
      update = True
      total_grad = combine_result(self.total_grad_prev, total_grad_batch, total_grad_next)
      p = optimizer.two_loop_recursion(-total_grad)
      lr = optimizer.step(p, g_Ok = total_grad_next, g_Sk = total_grad)
    else:
      optimizer.param_groups[0]['lr']=self.post_lr*self.optim_lr
      

    if not warm_up:
      if self.autoweight and self.epoch % self.weight_update_freq == 0:
        weights = self.WeightUpdater.update(weights, grad_batch)
        self.lambda_data, self.lambda_bc, self.lambda_ns = weights

    grad_prev = []
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

    self.total_grad_prev = self.WeightUpdater.new_grads(weights, grad_prev)

    if not first_call:
      optimizer.curvature_update(self.total_grad_prev, eps = 0.2, damping = True)



  def lbfgs_step(self, optimizer) -> None:
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
  def train(self, epochs: int, print_freq: int = 100, plot_freq: int = 500, point_update_freq: int = None, point_keep_percentage: float = 20, gradient_accumulation: bool = False, autoweight_scheme = None, autoweight_freq = 10, save_freq: int = 50) -> None:
    """
    Trains the PINN model for a specified number of epochs.

    Args:
      epochs (int): Number of epochs to train the model.
      print_freq (int, optional): Frequency of printing training progress. Defaults to 100.
      plot_freq (int, optional): Frequency of plotting loss history and field. Defaults to 500.
      point_update_freq (int, optional): Frequency of updating collocation and boundary points. Defaults to None.
      point_keep_percentage (float, optional): Percentage of collocation points to keep if points already exist. Defaults to 20.
      conflictfree (bool, optional): Flag to enable ConFIG. Defaults to False.
    
    """
    # Set random seed for reproducibility
    self.best_loss = float('inf')
    self.max_epochs = epochs
    self.grad_acc = gradient_accumulation
    
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
      # def warmupLR(idx_epoch):
      #   warmup_epochs = 100
      #   return idx_epoch / warmup_epochs if idx_epoch < warmup_epochs else 1

      #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.post_optimizer, lr_lambda=warmupLR)

      # if self.scheduler is not None:
      #   self.scheduler.optimizer = self.post_optimizer
      while self.epoch <= (epochs+self.post_epochs):
        self.step()
        if self.scheduler is not None:
          if self.scheduler.get_last_lr()[0] < 1e-6:
            break

      
    self.plot_history()  # Plot the final loss history
    self.plot_field()  # Plot the final field
    self.save_hist(f"{self.NAME}_history.csv")
    print("Training finished!")

    self.save(f"{self.NAME}_final_state.pt")


  #MARK: Plot History
  def plot_history(self) -> plt.Figure:
    """
    Plots the loss history of the PINN model, including total loss, data loss, boundary condition loss, and Navier-Stokes loss.
    Optionally plots the learning rate history if a scheduler is used.

    Returns:
      plt.Figure: The figure object containing the loss history plot.
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

    return fig

     

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

  #Mark Save & load state dict
  def save(self, path: str) -> None:
    """
    Saves the trained model to the specified path.

    Args:
      path (str): Path to save the trained model.
    """
    if isinstance(self.model, DataParallel):
      state_dict = self.model.module.state_dict()
    else:
      state_dict = self.model.state_dict()

    torch.save({
                'model_state_dict': state_dict,
                'norm_scales': self.norm_scales,
                'norm_offsets': self.norm_offsets
                }, path)

  def load(self, path: str) -> None:
    """
    Loads a trained model from the specified path.

    Args:
      path (str): Path to load the trained model.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    self.norm_scales = checkpoint['norm_scales']
    self.norm_offsets = checkpoint['norm_offsets']

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
    Initializes the PhysicsDataset with the given data.

    Args:
      data (torch.Tensor): The data to be used in the dataset.
    """
    self.Xdata = Xdata
    self.Ydata = Ydata

    self.batch_size = batch_size if batch_size is not None else len(Xdata[:, 0])
    self.shuffle = shuffle

    self.current_idx = 0
    
    if self.shuffle == True and min_size == None:
      self.min_size = batch_size
    elif min_size is not None:
      self.min_size = min_size
    else:
      self.min_size = 512

    self.autostepping = True
      

    self.l = self.__len__()

    if device is not None:
      self.device = device
    else:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


  def __len__(self) -> int:
    """
    Returns the length of the dataset.

    Returns:
      int: The number of data points in the dataset.
    """

    l = len(self.Xdata[:, 0]) // self.batch_size

    if len(self.Xdata[:, 0]) % self.batch_size >= self.min_size:
      l += 1

    return l

  def _manual_step(self):
    if self.current_idx == self.l-1:
      self.current_idx = 0
    else:
      self.current_idx += 1

  def __getitem__(self, dummy_idx: int, step = True) -> torch.Tensor:
    """
    Returns the data point at the specified index.

    Args:
      idx (int): The index of the data point to retrieve.

    Returns:
      torch.Tensor: The data point at the specified index.
    """

    idx = self.current_idx

    if idx == 0 and self.shuffle:
      indices = torch.randperm(len(self.Xdata[:, 0]))
      self.Xdata = self.Xdata[indices, :]
      if self.Ydata is not None:
        self.Ydata = self.Ydata[indices, :]

    idx_0 = idx * self.batch_size
    idx_1 = (idx + 1) * self.batch_size

    if idx_1 > len(self.Xdata[:, 0]):
      idx_1 = len(self.Xdata[:, 0])

    if step and self.autostepping:
      if idx == self.l-1:
        self.current_idx = 0
      else:
        self.current_idx += 1  

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




