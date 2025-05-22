import torch
from typing import Callable, Tuple, List


def wall(
  dim: int = 0,
  position: float = 0,
  direction: int = 1,
  order: int = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
  """
  Creates a wall boundary function for use in PINN (Physics-Informed Neural Network) boundary conditions.

  Args:
    dim (int): The dimension of the wall normal (default: 0).
    position (float): The position of the wall along the specified dimension (default: 0).
    direction (int): The direction of the wall normal; should be 1 or -1 (default: 1).
    order (int): The order of the wall function. 
      - If 1, returns a linear wall function with f(0)=0 and f'(0)=1.
      - If 2, returns a smoothed wall function using an exponential term, with f''(0)=1 (default: 1).

  Returns:
    function: A function f(p) that takes a tensor of points `p` and computes the signed distance value for each point.

  Notes:
    - For order=1, the function returns a signed distance from the wall.
    - For order=2, the function returns a smoothed version of the signed distance, useful for differentiability.
  """

  if order == 1:
    def f(p: torch.Tensor) -> torch.Tensor:
      d = p[:,dim]-position
      return d * direction
    
  elif order == 2:
    k = 1000 #smoothness parameter
    def f(p: torch.Tensor) -> torch.Tensor:
      d = p[:,dim]-position
      return (d+ 1/(k*2)*(1-torch.exp(-k*(d**2)))) * direction
  return f

def circle(
  r: float,
  center: Tuple[float, float] = (0, 0),
  dims: Tuple[int, int] = (0, 1),
  inside: bool = False,
  order: int = 1,
  scale: Tuple[float, float] = (1, 1),
  shift: Tuple[float, float] = (0, 0),
  ratio: float = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
  """
  Creates a signed distance function (SDF) for a circle (or ellipse) in arbitrary dimensions.
  
  Args:
    r (float): Radius of the circle.
    center (tuple, optional): Center coordinates of the circle in the selected dimensions. Defaults to (0, 0).
    dims (tuple, optional): Indices of the dimensions to use for the circle. Defaults to (0, 1).
    inside (bool, optional): If True, the SDF is positive inside the circle; otherwise, negative. Defaults to False.
    order (int, optional): Order of the SDF. If 1, returns standard SDF; if 2, returns a smoothed SDF for f''(0)=1. Defaults to 1.
    scale (tuple, optional): Scaling factors for each selected dimension, used for adapting to normalized values. Defaults to (1, 1).
    shift (tuple, optional): Shifts for each selected dimension, used for adapting to normalized values. Defaults to (0, 0).
    ratio (float, optional): Aspect ratio (r2/r1) for the second dimension, allowing for elliptical shapes or normalization. Defaults to 1.

  Returns:
    function: A function f(p) that computes the signed distance from points p to the circle (or ellipse),
      with sign determined by the 'inside' parameter and normalization applied via scale, shift, and ratio.
  """
  i = dims[0]
  j = dims[1]
  inside_sign = -1 if inside else 1

  if order == 1:
    def f(p: torch.Tensor) -> torch.Tensor:
      p1 = p[:, i] * scale[0] + shift[0]
      p2 = p[:, j] * scale[1] + shift[1]
      d = torch.sqrt((p1 - center[0]) ** 2 + ((p2 - center[1]) * ratio) ** 2) - r
      return d * inside_sign
  elif order == 2:
    def f(p: torch.Tensor) -> torch.Tensor:
      p1 = p[:, i] * scale[0] + shift[0]
      p2 = p[:, j] * scale[1] + shift[1]
      d = torch.sqrt((p1 - center[0]) ** 2 + ((p2 - center[1]) * ratio) ** 2) - r
      return (d + 1 / 2 * (1 - torch.exp(-d ** 2))) * inside_sign

  return f


def line_segment(
  x0: Tuple[float, float] = (0, 0),
  x1: Tuple[float, float] = (1, 1)
) -> Callable[[torch.Tensor], torch.Tensor]:
  """
  Returns an approximate distance function (ADF) for a finite line segment in 2D.

  Args:
    x0 (tuple): Start point of the segment (x, y).
    x1 (tuple): End point of the segment (x, y).

  Returns:
    Callable: A function f(p) that takes a tensor of points p (shape [N, 2])
          and returns the signed distance from each point to the segment.
  """
  x0_ = torch.tensor(x0).float()
  x1_ = torch.tensor(x1).float()
  L_ = torch.linalg.norm(x1_ - x0_)

  def f(p: torch.Tensor) -> torch.Tensor:
    x0 = x0_.to(p.device)
    x1 = x1_.to(p.device)
    L = L_.to(p.device)
    c = (x0 + x1) / 2
    # Perpendicular distance from point to the infinite line
    d = ((p[:, 0] - x0[0]) * (x1[1] - x0[1]) - (p[:, 1] - x0[1]) * (x1[0] - x0[0])) / L
    # Parameter t determines if the projection falls within the segment
    t = 1 / L * ((L / 2) ** 2 - torch.linalg.norm(p[:, 0:2] - c, dim=1) ** 2)
    # Combine perpendicular and parallel distances for the segment SDF
    return torch.sqrt(d ** 2 + ((torch.sqrt(t ** 2 + d ** 4) - t) / 2) ** 2)
  return f


def merge_segments(
  line_segments: List[Callable[[torch.Tensor], torch.Tensor]],
  smoothness: int = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
  """
  Merges multiple line segment distance functions into a single smooth approximate distance function.

  Args:
    line_segments (List[Callable[[torch.Tensor], torch.Tensor]]): 
      List of distance functions for individual line segments.
    smoothness (int, optional): 
      Smoothness parameter controlling the sharpness of the merge. Defaults to 1.

  Returns:
    Callable[[torch.Tensor], torch.Tensor]: 
      A function that computes the merged distance for input points.
  """
  s = smoothness
  def f(p: torch.Tensor) -> torch.Tensor:
    merged = 0.0
    for segment in line_segments:
      merged += 1 / (segment(p) ** s)
    return 1 / (merged ** (1 / s))
  return f


def unite(
  f1: Callable[[torch.Tensor], torch.Tensor],
  f2: Callable[[torch.Tensor], torch.Tensor],
  order: int = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
  """
  Returns a function representing the union (logical OR) of two signed distance functions (SDFs).
  The result is a smooth approximation of the minimum of the two SDFs.

  Args:
    f1: First SDF function.
    f2: Second SDF function.
    order: Controls the smoothness of the union. Higher values yield sharper transitions.

  Returns:
    A function that computes the smooth union of f1 and f2 for input tensor p.
  """
  s = order + 1
  def f(p: torch.Tensor) -> torch.Tensor:
    return (f1(p) + f2(p) - (abs(f1(p))**s + abs(f2(p))**s)**(1/s))
  return f

def subtract(
  f1: Callable[[torch.Tensor], torch.Tensor],
  f2: Callable[[torch.Tensor], torch.Tensor],
  order: int = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
  """
  Returns a function representing the subtraction (difference) of two signed distance functions (SDFs).
  The result is a smooth approximation of the set difference (A \ B).

  Args:
    f1: First SDF function (A).
    f2: Second SDF function (B).
    order: Controls the smoothness of the subtraction. Higher values yield sharper transitions.

  Returns:
    A function that computes the smooth subtraction of f2 from f1 for input tensor p.
  """
  s = order + 1
  def f(p: torch.Tensor) -> torch.Tensor:
    return (f1(p) + f2(p) + (abs(f1(p))**s + abs(f2(p))**s)**(1/s))
  return f
