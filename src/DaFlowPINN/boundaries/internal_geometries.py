import numpy as np
from typing import Callable, Tuple

def halfcylinder_3d(
    r: float = 0.125,
    dims: Tuple[int, int] = (0, 1),
    center: Tuple[float, float] = (0, 0)
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Creates a function to check if points are inside a half-cylinder in 3D space.
    The halfcylinder is oriented in the negative direction of the first dimension.
    
    Args:
        r (float): Radius of the half-cylinder.
        dims (tuple): Indices of the dimensions to use for the half-cylinder.
                      The first index is the negative-direction of the cylinder's curved side.
                      Dimensions can be 0, 1, or 2.
        center (tuple): Center coordinates of the half-cylinder.
    """

    def check_inside(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Determines if the given coordinates are inside a half-cylinder with diameter 0.25.

        Args:
            x (np.ndarray): x-coordinates.
            y (np.ndarray): y-coordinates.
            z (np.ndarray): z-coordinates.

        Returns:
            np.ndarray: Boolean array indicating if each coordinate is inside the half-cylinder.
        """

        if dims[0] == 0:
            x_ = x - center[0]
        elif dims[0] == 1:
            x_ = y - center[0]
        elif dims[0] == 2:
            x_ = z - center[0]
        
        if dims[1] == 0:
            y_ = x - center[1]
        elif dims[1] == 1:
            y_ = y - center[1]
        elif dims[1] == 2:
            y_ = z - center[1]
    
        # Check if the points are inside the half-cylinder
        inside = ((x_**2 + y_**2) < r**2) & (x_ < 0)
        return inside.astype(bool)

    return check_inside

