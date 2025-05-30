from pathlib import Path
import numpy as np
from scipy.stats import binned_statistic_dd
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET
import vtk
from vtkmodules.util import numpy_support
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, RBFInterpolator
import sys

def bin_averaging(sample, values, bins):
    """
    Bin and interpolate the data using binned_statistic_dd and RegularGridInterpolator.
    
    Parameters:
    - sample: The data points to be binned.
    - values: The values to be averaged in each bin.
    - bins: The bin edges for each dimension.
    
    Returns:
    - filled_stat: The interpolated values on the grid.
    """
    # Compute mean velocity magnitude per bin
    stat, edges, binnumber = binned_statistic_dd(
        sample=sample,
        values=values,
        statistic='mean',
        bins=bins
    )

    # Create grid coordinates
    x_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
    y_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
    z_centers = 0.5 * (edges[2][:-1] + edges[2][1:])

    # Mask NaNs (simple interpolation can't handle NaNs)
    valid_mask = ~np.isnan(stat)
    filled_stat = stat.copy()

    # Interpolate only where data is valid
    # Use only valid (non-NaN) points for interpolation, not assuming a regular grid
    valid_points = np.array(np.nonzero(valid_mask)).T

    # Fill NaNs by averaging valid neighboring bins
    def average_neighbors(arr, mask):
        filled = arr.copy()
        idxs = np.argwhere(~mask)
        for idx in idxs:
            x, y, z = idx
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < arr.shape[0] and
                            0 <= ny < arr.shape[1] and
                            0 <= nz < arr.shape[2] and
                            mask[nx, ny, nz]):
                            neighbors.append(arr[nx, ny, nz])
            if neighbors:
                filled[x, y, z] = np.mean(neighbors)
        return filled

    filled_stat = average_neighbors(filled_stat, valid_mask)

    return filled_stat, (x_centers, y_centers, z_centers)


def gaussian_weighted_binning(sample, values, bins, sigma=0.1, cutoff=0.3):

    U, V, W = values


    x_centers = 0.5 * (bins[0][:-1] + bins[0][1:])
    y_centers = 0.5 * (bins[1][:-1] + bins[1][1:])
    z_centers = 0.5 * (bins[2][:-1] + bins[2][1:])

    nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)

    

    particles = np.column_stack(sample)

    kdtree = cKDTree(particles)

    grid_points = np.array(np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')).reshape(3, -1).T

    u_grid = np.zeros(grid_points.shape[0])
    v_grid = np.zeros(grid_points.shape[0])
    w_grid = np.zeros(grid_points.shape[0])

    dx = x_centers[1] - x_centers[0]

    for i, gp in enumerate(grid_points):
        # Find the indices of the points within the cutoff distance
        indices = kdtree.query_ball_point(gp, cutoff)

        if not indices:
            distances, indices = kdtree.query(gp, k=1)
        
        if np.isscalar(indices):
            indices = [indices]
        else:
            indices = list(indices)

        

        # Compute the weights using a Gaussian function
        distances = np.linalg.norm(particles[indices] - gp, axis=1)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
    
        u_grid[i] = np.average(U[indices], weights=weights)
        v_grid[i] = np.average(V[indices], weights=weights)
        w_grid[i] = np.average(W[indices], weights=weights)

    # Reshape to grid
    shape = (nx, ny, nz)
    U_field = u_grid.reshape(shape)
    V_field = v_grid.reshape(shape)
    W_field = w_grid.reshape(shape)

    return U_field, V_field, W_field, (x_centers, y_centers, z_centers)


def gaussian_interpolation(sample, values, edges, sigma=0.1):

    centers = [0.5 * (edges[i][:-1] + edges[i][1:]) for i in range(len(edges))]

    grid_x, grid_y, grid_z = np.meshgrid(centers[0], centers[1], centers[2], indexing='ij')
    grid_points = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

    particles = np.column_stack(sample)

    values = np.column_stack(values)

    rbf = RBFInterpolator(particles, values, kernel='gaussian', epsilon=sigma, neighbors=25)

    interp = rbf(grid_points)  # shape: (n_points, 3)
    u_grid = interp[:, 0].reshape(grid_x.shape)
    v_grid = interp[:, 1].reshape(grid_y.shape)
    w_grid = interp[:, 2].reshape(grid_z.shape)

    return u_grid, v_grid, w_grid, centers


def binning_3D(data, lb, ub, t_step=48, nx=50, ny=50, nz=50, sigma=0.1):

    t_vec = data[:, 4] #time vector
    t_vec = np.unique(t_vec) #unique time steps
    t_bin = t_vec[t_step] #time step to bin
    data = data[data[:, 4] == t_bin, :] #select time step

    print(f"Number of particles at time step {t_bin}: {data.shape[0]}"),
    print(f"Time step: {t_bin} s")

    X = data[:, 1] #x-coordinates
    Y = data[:, 2] #y-coordinates
    Z = data[:, 3] #z-coordinates
    U = data[:, 5] #x-velocities
    V = data[:, 6] #y-velocities
    W = data[:, 7] #z-velocities



    print(f"Sigma: {sigma} m")


    print(f"Number of bins in x: {nx}, y: {ny}, z: {nz}")


    # Choose your bin edges (e.g., 3D bins in X, Y, Z)
    x_edges = np.linspace(lb[0], ub[0], num=nx+1)
    y_edges = np.linspace(lb[1], ub[1], num=ny+1)
    z_edges = np.linspace(lb[2], ub[2], num=nz+1)


    # u_binned, centers = bin_averaging(
    #     sample=(X, Y, Z),
    #     values=U,
    #     bins=[x_edges, y_edges, z_edges]
    # )

    # v_binned, _ = bin_averaging(
    #     sample=(X, Y, Z),
    #     values=V,
    #     bins=[x_edges, y_edges, z_edges]
    # )

    # w_binned, _ = bin_averaging(
    #     sample=(X, Y, Z),
    #     values=W,
    #     bins=[x_edges, y_edges, z_edges]
    # )

    u_binned, v_binned, w_binned, centers = gaussian_weighted_binning(
        sample=(X, Y, Z),
        values=(U, V, W),
        bins=[x_edges, y_edges, z_edges],
        sigma=sigma,
        cutoff=3*sigma
    )

    # u_binned, v_binned, w_binned, centers = gaussian_interpolation(
    #     sample=(X, Y, Z),
    #     values=(U, V, W),
    #     edges=[x_edges, y_edges, z_edges],
    #     sigma=sigma
    # )

    return u_binned, v_binned, w_binned, centers

if __name__ == "__main__":

    n = sys.argv[1]  # Set the number of rows to load
    sigma = float(sys.argv[2])  # Set the sigma value

    #t = 24

    data = np.loadtxt(f"/scratch/jpelz/da-challenge/DA_CASE01/velocity_files/DA_CASE01_TR_ppp_0_{n}_velocities.dat", delimiter=" ", skiprows=1)
    lb = [0, 1E-3, 1E-3]
    ub = [0.1, 0.0991, 0.0991]


    #data = np.loadtxt("G:\Meine Ablage\MA\DA_CASE01/velocity_files\DA_CASE01_TR_ppp_0_010_velocities.dat", skiprows=1, delimiter=" ", max_rows=n)



    time_steps = np.unique(data[:, 4])[:]
    output_dir = "/scratch/jpelz/da-challenge/DA_CASE01/binned/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")

    for i, t_step in enumerate(time_steps):
        # U, V, W, centers = binning_3D(data, t_step=i, sigma=1,
        #                               L_x=0.1, L_y=0.1, L_z=0.1,
        #                               nx=50, ny=50, nz=50)

        U, V, W, centers = binning_3D(data, t_step=i, sigma=sigma,
                                      lb=lb, ub=ub,
                                      nx=101, ny=100, nz=100)

        structured_grid = vtk.vtkImageData()

        structured_grid.SetDimensions(U.shape[0]+1, U.shape[1]+1, U.shape[2]+1)
        spacing_x = centers[0][1] - centers[0][0]
        spacing_y = centers[1][1] - centers[1][0]
        spacing_z = centers[2][1] - centers[2][0]
        structured_grid.SetOrigin(
            centers[0][0] - spacing_x / 2,
            centers[1][0] - spacing_y / 2,
            centers[2][0] - spacing_z / 2
        )
        structured_grid.SetSpacing(centers[0][1] - centers[0][0],
                                   centers[1][1] - centers[1][0],
                                   centers[2][1] - centers[2][0])


        vector_data = np.stack((U, V, W), axis=-1)
        vtk_vector = numpy_support.numpy_to_vtk(
            vector_data.reshape(-1, 3, order='F'),
            deep=True
        )
        vtk_vector.SetName("Velocity")
        structured_grid.GetCellData().AddArray(vtk_vector)


        # Write the structured grid to a VTK file
        writer = vtk.vtkXMLImageDataWriter()
        filename = f"binned_n{n}_{i:03d}.vti"
        writer.SetFileName(f"{output_dir}/{filename}")
        writer.SetInputData(structured_grid)
        writer.Write()

        # Add the file to the collection
        ET.SubElement(collection, "DataSet", timestep=str(t_step), group="", part="0", file=filename)

    tree = ET.ElementTree(root)
    tree.write(f"{output_dir}/binned_n{n}.pvd")