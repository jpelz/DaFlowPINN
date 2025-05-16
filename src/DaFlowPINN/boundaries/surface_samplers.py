import numpy as np
from scipy.stats import qmc
from typing import List, Callable, Tuple

def empty() -> callable:
    """
    Basic example for sampler functions
    """
    def sampler(nb: int) -> np.ndarray:
        """
        Generates points on the boundary surface.

        Args:
            nb (int): The number of points to generate.

        Returns:
            np.ndarray: An array of shape (nb, 4) containing the sampled points.
        """
        return np.zeros((nb, 4))

    return sampler

def halfcylinder(center: Tuple[float, float, float], r: float, h: float, tmin: float, tmax: float) -> callable:
    """
    Generates a sampler function for a half-cylinder surface.

    Args:
        center (tuple): The center of the half-cylinder (x, y, z).
        r (float): The radius of the half-cylinder.
        h (float): The height of the half-cylinder.
        tmin (float): The minimum value for the time dimension.
        tmax (float): The maximum value for the time dimension.

    Returns:
        function: A sampler function that generates a specified numer of points on the half-cylinder surface.
    """
    def sampler(nb: int) -> np.ndarray:
        x_c, y_c, z_c = center

        sampler = qmc.LatinHypercube(d=3)
        nb_front = int(nb * np.pi / (np.pi + 2))
        nb_back = nb - nb_front

        lb_front = [np.pi / 2, -h / 2, tmin]
        ub_front = [np.pi * 3 / 2, h / 2, tmax]
        lb_back = [-r, -h / 2, tmin]
        ub_back = [r, h / 2, tmax]

        # Cylinder front
        sample = sampler.random(n=nb_front)
        sample = qmc.scale(sample, l_bounds=lb_front, u_bounds=ub_front)
        theta = sample[:, 0]
        z_front = sample[:, 1]
        t_front = sample[:, 2]
        x_front = r * np.cos(theta)
        y_front = r * np.sin(theta)

        # Cylinder back
        sample2 = sampler.random(n=nb_back)
        sample2 = qmc.scale(sample2, l_bounds=lb_back, u_bounds=ub_back)
        y_back = sample2[:, 0]
        z_back = sample2[:, 1]
        t_back = sample2[:, 2]
        x_back = np.zeros_like(y_back)

        # Combine front and back samples
        x_b = np.append(x_front, x_back) + x_c
        y_b = np.append(y_front, y_back) + y_c
        z_b = np.append(z_front, z_back) + z_c
        t_b = np.append(t_front, t_back)

        return np.transpose(np.vstack((x_b, y_b, z_b, t_b)))

    return sampler

def duct(lb: Tuple[float, float, float, float], ub: Tuple[float, float, float, float]) -> callable:
    """
    Generates a sampler function for a duct surface (excluding the back wall).

    Args:
        lb (tuple): The lower bounds (xmin, ymin, zmin, tmin).
        ub (tuple): The upper bounds (xmax, ymax, zmax, tmax).

    Returns:
        function: A sampler function that generates a specified number of points on the duct surface.
    """
    xmin, ymin, zmin, tmin = lb
    xmax, ymax, zmax, tmax = ub

    def sampler(nb: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=3)  # 3D sampling

        # Distribute points evenly across the four surfaces (no back wall)
        nb_side = nb // 4  # Each side wall gets nb_side points
        nb_top_bottom = nb // 4  # Each top/bottom wall gets nb_top_bottom points

        # Side wall (y = ymin)
        sample_ymin = sampler.random(n=nb_side)
        sample_ymin = qmc.scale(sample_ymin, l_bounds=[xmin, zmin, tmin], u_bounds=[xmax, zmax, tmax])
        x_ymin, z_ymin, t_ymin = sample_ymin.T
        y_ymin = np.full_like(x_ymin, ymin)

        # Side wall (y = ymax)
        sample_ymax = sampler.random(n=nb_side)
        sample_ymax = qmc.scale(sample_ymax, l_bounds=[xmin, zmin, tmin], u_bounds=[xmax, zmax, tmax])
        x_ymax, z_ymax, t_ymax = sample_ymax.T
        y_ymax = np.full_like(x_ymax, ymax)

        # Bottom wall (z = zmin)
        sample_zmin = sampler.random(n=nb_top_bottom)
        sample_zmin = qmc.scale(sample_zmin, l_bounds=[xmin, ymin, tmin], u_bounds=[xmax, ymax, tmax])
        x_zmin, y_zmin, t_zmin = sample_zmin.T
        z_zmin = np.full_like(x_zmin, zmin)

        # Top wall (z = zmax)
        sample_zmax = sampler.random(n=nb_top_bottom)
        sample_zmax = qmc.scale(sample_zmax, l_bounds=[xmin, ymin, tmin], u_bounds=[xmax, ymax, tmax])
        x_zmax, y_zmax, t_zmax = sample_zmax.T
        z_zmax = np.full_like(x_zmax, zmax)

        # Combine all points (no back wall)
        x_b = np.concatenate([x_ymin, x_ymax, x_zmin, x_zmax])
        y_b = np.concatenate([y_ymin, y_ymax, y_zmin, y_zmax])
        z_b = np.concatenate([z_ymin, z_ymax, z_zmin, z_zmax])
        t_b = np.concatenate([t_ymin, t_ymax, t_zmin, t_zmax])

        return np.transpose(np.vstack((x_b, y_b, z_b, t_b)))

    return sampler

def wall(dim1: int, dim2: int, dim1_min: float, dim1_max: float, dim2_min: float, dim2_max: float, dim3_value: float, t_min: float, t_max: float) -> callable:
    """
    Generates a sampler function for a wall surface in a 3D space with a fixed value for one dimension.

    Args:
        dim1 (int): The first dimension index (0, 1, or 2).
        dim2 (int): The second dimension index (0, 1, or 2).
        dim1_min (float): The minimum value for the first dimension.
        dim1_max (float): The maximum value for the first dimension.
        dim2_min (float): The minimum value for the second dimension.
        dim2_max (float): The maximum value for the second dimension.
        dim3_value (float): The fixed value for the third dimension.
        t_min (float): The minimum value for the time dimension.
        t_max (float): The maximum value for the time dimension.

    Returns:
        callable: A sampler function that generates a specified number of points on the wall surface.
    """
    lb = [dim1_min, dim2_min, t_min]
    ub = [dim1_max, dim2_max, t_max]

    # Determine the third dimension which is not dim1 or dim2
    dim3 = next(i for i in range(3) if i not in [dim1, dim2])

    def sampler(nb: int) -> np.ndarray:
        """
        Generates points on the wall surface.

        Args:
            nb (int): The number of points to generate.

        Returns:
            np.ndarray: An array of shape (nb, 4) containing the sampled points.
        """
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=nb)
        sample = qmc.scale(sample, l_bounds=lb, u_bounds=ub)

        X = np.zeros((nb, 4))
        X[:, dim1] = sample[:, 0]
        X[:, dim2] = sample[:, 1]
        X[:, dim3] = dim3_value * np.ones_like(sample[:, 0])
        X[:, 3] = sample[:, 2]

        return X

    return sampler

def combine(samplers: List[Callable[[int], np.ndarray]], weights=None) -> Callable[[int], np.ndarray]:
    """
    Combines multiple sampler functions into a single sampler function.

    Args:
        samplers (List[Callable[[int], np.ndarray]]): A list of sampler functions.
        weights (List[float], optional): A list of weights for each sampler. If None, equal weights are used.

    Returns:
        Callable[[int], np.ndarray]: A combined sampler function that generates points from all input samplers.
    """

    # Normalize weights if provided, otherwise use equal weights
    if weights is None:
        weights = np.ones(len(samplers)) / len(samplers)
    else:
        weights = np.array(weights)
        weights /= np.sum(weights)

    def combined_sampler(nb: int) -> np.ndarray:
        """
        Generates points using all combined samplers.

        Args:
            nb (int): The total number of points to generate.

        Returns:
            np.ndarray: An array containing the combined sampled points from all samplers.
        """
        data = []
        for sampler, weight in zip(samplers, weights):
            num_points = int(nb * weight)  # Calculate the number of points for each sampler based on weight
            data.append(sampler(num_points))  # Generate points using the sampler and append to data list
        return np.concatenate(data)  # Concatenate all sampled points into a single array

    return combined_sampler


