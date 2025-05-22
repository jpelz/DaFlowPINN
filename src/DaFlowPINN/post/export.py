import torch
import numpy as np
import h5py
import vtk
from vtkmodules.util import numpy_support
import xml.etree.ElementTree as ET

from typing import List

def export_h5(
    predict: callable,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    t_slice: float,
    dx: float,
    dy: float,
    dz: float,
    filename: str,
    p_ref_idx: List[int] = [0, 0, 0]
) -> None:
    """
    Export predicted velocity, pressure, and their gradients to an HDF5 (.h5) file.

    Args:
        predict: Callable that takes a tensor of shape (N, 4) and returns predictions.
        x_grid, y_grid, z_grid: 3D numpy arrays defining the spatial grid.
        t_slice: Time value for the prediction.
        dx, dy, dz: Grid spacings in x, y, z directions.
        filename: Output HDF5 file path.
        p_ref_idx: Index [ix, iy, iz] for pressure reference point (default [0,0,0]).
    """
    # Prepare input for prediction
    X = np.stack(
        (
            x_grid.flatten(order='F'),
            y_grid.flatten(order='F'),
            z_grid.flatten(order='F'),
            np.ones_like(x_grid.flatten(order='F')) * t_slice
        ),
        axis=-1
    )
    X = torch.tensor(X).float()
    result = predict(X).detach().numpy()

    # Reshape predictions to grid
    VX = result[:, 0].reshape(x_grid.shape, order='F')
    VY = result[:, 1].reshape(x_grid.shape, order='F')
    VZ = result[:, 2].reshape(x_grid.shape, order='F')
    p = result[:, 3].reshape(x_grid.shape, order='F')

    # Shift pressure to reference point
    p = p - p[p_ref_idx[0], p_ref_idx[1], p_ref_idx[2]]

    # Compute velocity gradients
    dVXdX = np.gradient(VX, dx, axis=0)
    dVXdY = np.gradient(VX, dy, axis=1)
    dVXdZ = np.gradient(VX, dz, axis=2)

    dVYdX = np.gradient(VY, dx, axis=0)
    dVYdY = np.gradient(VY, dy, axis=1)
    dVYdZ = np.gradient(VY, dz, axis=2)

    dVZdX = np.gradient(VZ, dx, axis=0)
    dVZdY = np.gradient(VZ, dy, axis=1)
    dVZdZ = np.gradient(VZ, dz, axis=2)

    # Export data to HDF5 file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=x_grid.flatten(order='F') * 1E3)
        f.create_dataset('Y', data=y_grid.flatten(order='F') * 1E3)
        f.create_dataset('Z', data=z_grid.flatten(order='F') * 1E3)
        f.create_dataset('VX', data=VX.flatten(order='F'))
        f.create_dataset('VY', data=VY.flatten(order='F'))
        f.create_dataset('VZ', data=VZ.flatten(order='F'))
        f.create_dataset('p', data=p.flatten(order='F'))
        f.create_dataset('dVXdX', data=dVXdX.flatten(order='F'))
        f.create_dataset('dVXdY', data=dVXdY.flatten(order='F'))
        f.create_dataset('dVXdZ', data=dVXdZ.flatten(order='F'))
        f.create_dataset('dVYdX', data=dVYdX.flatten(order='F'))
        f.create_dataset('dVYdY', data=dVYdY.flatten(order='F'))
        f.create_dataset('dVYdZ', data=dVYdZ.flatten(order='F'))
        f.create_dataset('dVZdX', data=dVZdX.flatten(order='F'))
        f.create_dataset('dVZdY', data=dVZdY.flatten(order='F'))
        f.create_dataset('dVZdZ', data=dVZdZ.flatten(order='F'))


def export_vts(
    predict: callable,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    t_slices: list,
    filename: str,
    p_ref_idx: list = [0, 0, 0]
) -> None:
    """
    Export predicted velocity and pressure fields to VTK Structured Grid (.vts) files for ParaView.

    Args:
        predict: Callable that takes a tensor of shape (N, 4) and returns predictions.
        x_grid, y_grid, z_grid: 3D numpy arrays defining the spatial grid.
        t_slices: List of time values for which to export data.
        filename: Output file prefix for the VTS files.
        p_ref_idx: Index [ix, iy, iz] for pressure reference point (default [0,0,0]).
    """
    for t in t_slices:
        # Prepare input for prediction
        X = np.stack(
            (
                x_grid.flatten(order='F'),
                y_grid.flatten(order='F'),
                z_grid.flatten(order='F'),
                np.ones_like(x_grid.flatten(order='F')) * t
            ),
            axis=-1
        )
        X = torch.tensor(X).float()
        result = predict(X).detach().numpy()

        # Reshape predictions to grid
        VX = result[:, 0].reshape(x_grid.shape, order='F')
        VY = result[:, 1].reshape(x_grid.shape, order='F')
        VZ = result[:, 2].reshape(x_grid.shape, order='F')
        p = result[:, 3].reshape(x_grid.shape, order='F')

        # Shift pressure to reference point
        p = p - p[p_ref_idx[0], p_ref_idx[1], p_ref_idx[2]]

        # Create a VTK structured grid
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(x_grid.shape)

        # Create points array for the grid
        points = vtk.vtkPoints()
        points_array = np.column_stack(
            (
                x_grid.flatten(order='F'),
                y_grid.flatten(order='F'),
                z_grid.flatten(order='F')
            )
        )
        points.SetData(numpy_support.numpy_to_vtk(points_array, deep=True))
        structured_grid.SetPoints(points)

        # Helper function to add data arrays to the grid
        def add_array_to_grid(data: np.ndarray, name: str) -> None:
            vtk_data = numpy_support.numpy_to_vtk(data.flatten(order='F'), deep=True)
            vtk_data.SetName(name)
            structured_grid.GetPointData().AddArray(vtk_data)

        # Add velocity and pressure data arrays
        add_array_to_grid(VX, 'VX')
        add_array_to_grid(VY, 'VY')
        add_array_to_grid(VZ, 'VZ')
        add_array_to_grid(p, 'p')

        # Write the structured grid to a VTK file
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(structured_grid)
        writer.SetFileName(f'{filename}_{t:.3f}.vts')
        writer.Write()


def export_vti(
    predict: callable,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    t_slices: list,
    filename: str,
    p_ref_idx: list = [0, 0, 0]
) -> None:
    """
    Export predicted velocity and pressure fields as VTK ImageData (.vti) files and a .pvd collection for ParaView.

    Args:
        predict: Callable that takes a tensor of shape (N, 4) and returns predictions.
        x_grid, y_grid, z_grid: 3D numpy arrays defining the spatial grid.
        t_slices: List of time values for which to export data.
        filename: Output file prefix for the VTI and PVD files.
        p_ref_idx: Index [ix, iy, iz] for pressure reference point (default [0,0,0]).
    """
    # Compute grid spacing
    dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
    dy = y_grid[0, 1, 0] - y_grid[0, 0, 0]
    dz = z_grid[0, 0, 1] - z_grid[0, 0, 0]

    # Compute origin (lower-left corner of first voxel)
    origin = (
        x_grid[0, 0, 0] - dx / 2,
        y_grid[0, 0, 0] - dy / 2,
        z_grid[0, 0, 0] - dz / 2
    )

    # Cell dimensions (ImageData uses point dimensions = cell_dim + 1)
    nx, ny, nz = x_grid.shape
    dims = (nx + 1, ny + 1, nz + 1)

    # Set up PVD file structure
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")

    for i, t in enumerate(t_slices):
        # Prepare input for prediction
        X = np.stack((
            x_grid.flatten(order='F'),
            y_grid.flatten(order='F'),
            z_grid.flatten(order='F'),
            np.ones_like(x_grid.flatten(order='F')) * t
        ), axis=-1)
        X = torch.tensor(X).float()
        result = predict(X).detach().numpy()

        # Reshape predictions to grid
        VX = result[:, 0].reshape(x_grid.shape, order='F')
        VY = result[:, 1].reshape(x_grid.shape, order='F')
        VZ = result[:, 2].reshape(x_grid.shape, order='F')
        p = result[:, 3].reshape(x_grid.shape, order='F')

        # Shift pressure to reference point
        p -= p[p_ref_idx[0], p_ref_idx[1], p_ref_idx[2]]

        # Combine velocity components into a vector array
        velocity = np.stack((VX, VY, VZ), axis=-1)

        # Create vtkImageData (VTI)
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(dims)
        image_data.SetOrigin(origin)
        image_data.SetSpacing((dx, dy, dz))

        # Convert and assign CellData arrays
        vel_vtk = numpy_support.numpy_to_vtk(velocity.reshape(-1, 3, order='F'), deep=True)
        vel_vtk.SetName("Velocity")
        image_data.GetCellData().AddArray(vel_vtk)

        p_vtk = numpy_support.numpy_to_vtk(p.flatten(order='F'), deep=True)
        p_vtk.SetName("Pressure")
        image_data.GetCellData().AddArray(p_vtk)

        # Write .vti file
        vti_filename = f"{filename}_ParaView_{i:03d}.vti"
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(vti_filename)
        writer.SetInputData(image_data)
        writer.Write()

        # Add to .pvd collection
        ET.SubElement(collection, "DataSet", timestep=str(t), group="", part="0", file=vti_filename)

    # Write .pvd file
    pvd_path = f"{filename}_ParaView.pvd"
    tree = ET.ElementTree(root)
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)
