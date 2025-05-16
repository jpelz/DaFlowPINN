import torch
import numpy as np
import h5py
import vtk
from vtkmodules.util import numpy_support

def export_h5(predict: callable, x_grid, y_grid, z_grid, t_slice, dx, dy, dz, filename, p_ref_idx=[0,0,0]):
    X = np.stack((x_grid.flatten(order='F'), 
                  y_grid.flatten(order='F'), 
                  z_grid.flatten(order='F'), 
                  np.ones_like(x_grid.flatten(order='F'))*t_slice), axis=-1)
    X = torch.tensor(X).float()
    result = predict(X).detach().numpy()

    VX = result[:, 0].reshape(x_grid.shape, order='F')
    VY = result[:, 1].reshape(x_grid.shape, order='F')
    VZ = result[:, 2].reshape(x_grid.shape, order='F')
    p = result[:, 3].reshape(x_grid.shape, order='F')

    # Shift pressure to reference point
    p = p - p[p_ref_idx[0], p_ref_idx[1], p_ref_idx[2]]

    # Calculate gradients

    dVXdX = np.gradient(VX, dx, axis=0)
    dVXdY = np.gradient(VX, dy, axis=1)
    dVXdZ = np.gradient(VX, dz, axis=2)

    dVYdX = np.gradient(VY, dx, axis=0)
    dVYdY = np.gradient(VY, dy, axis=1)
    dVYdZ = np.gradient(VY, dz, axis=2)

    dVZdX = np.gradient(VZ, dx, axis=0)
    dVZdY = np.gradient(VZ, dy, axis=1)
    dVZdZ = np.gradient(VZ, dz, axis=2)

    #export to h5 file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=x_grid.flatten(order='F')*1E3)
        f.create_dataset('Y', data=y_grid.flatten(order='F')*1E3)
        f.create_dataset('Z', data=z_grid.flatten(order='F')*1E3)
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


def export_vts(predict: callable, x_grid, y_grid, z_grid, t_slices, filename, p_ref_idx=[0,0,0]):
    # Export to VTK file for ParaView

    for t in t_slices:
        X = np.stack((x_grid.flatten(order='F'), 
                    y_grid.flatten(order='F'), 
                    z_grid.flatten(order='F'), 
                    np.ones_like(x_grid.flatten(order='F'))*t), axis=-1)
        X = torch.tensor(X).float()
        result = predict(X).detach().numpy()

        VX = result[:, 0].reshape(x_grid.shape, order='F')
        VY = result[:, 1].reshape(x_grid.shape, order='F')
        VZ = result[:, 2].reshape(x_grid.shape, order='F')
        p = result[:, 3].reshape(x_grid.shape, order='F')

        # Shift pressure to reference point
        p = p - p[p_ref_idx[0], p_ref_idx[1], p_ref_idx[2]]


        # Create a VTK structured grid
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(x_grid.shape)

        # Create points
        points = vtk.vtkPoints()
        points_array = np.column_stack((x_grid.flatten(order='F'), 
                                        y_grid.flatten(order='F'), 
                                        z_grid.flatten(order='F')))
        points.SetData(numpy_support.numpy_to_vtk(points_array, deep=True))
        structured_grid.SetPoints(points)

        # Function to add data to the grid
        def add_array_to_grid(data, name):
            vtk_data = numpy_support.numpy_to_vtk(data.flatten(order='F'), deep=True)
            vtk_data.SetName(name)
            structured_grid.GetPointData().AddArray(vtk_data)

        # Add velocity and pressure data
        add_array_to_grid(VX, 'VX')
        add_array_to_grid(VY, 'VY')
        add_array_to_grid(VZ, 'VZ')
        add_array_to_grid(p, 'p')

        # Write the structured grid to a VTK file
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(structured_grid)
        writer.SetFileName(f'{filename}_{t:.3f}.vts')
        writer.Write()

