import vtk
from vtkmodules.util import numpy_support
import os
import numpy as np
import sys

def write_unstructured_vtu(positions, velocities, ids, timestep_index, output_dir="particles_vtu", file_ending="p001"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_points = positions.shape[0]

    # Create VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(positions.astype(np.float32)))

    # Create an unstructured grid with vertices
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    # Add each point as a vertex cell
    for i in range(n_points):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        grid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())

    # Add velocity vector
    vtk_velocity = numpy_support.numpy_to_vtk(velocities.astype(np.float32))
    vtk_velocity.SetName("Velocity")
    grid.GetPointData().AddArray(vtk_velocity)

    # Add ID as a scalar array
    vtk_ids = numpy_support.numpy_to_vtk(ids.astype(np.int32))
    vtk_ids.SetName("ID")
    grid.GetPointData().AddArray(vtk_ids)

    # Write the file
    filename = os.path.join(output_dir, f"particles_{file_ending}_{timestep_index:04d}.vtu")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
    return filename



def export_unstructured_series(data, output_dir="particles_vtu", file_ending="p001"):
    t_vec = np.unique(data[:, 4])
    pvd_entries = []

    for i, t in enumerate(t_vec[:-1]):
        step_data = data[data[:, 4] == t]
        positions = step_data[:, 1:4]          # X, Y, Z
        velocities = step_data[:, 5:8]         # U, V, W
        ids = step_data[:, 0].astype(np.int32) # ID column

        vtu_filename = write_unstructured_vtu(positions, velocities, ids, i, output_dir, file_ending)
        pvd_entries.append((t, os.path.basename(vtu_filename)))

    return pvd_entries



def write_pvd(pvd_entries, output_dir="particles_vtu", pvd_filename="particles_series.pvd"):
    with open(os.path.join(output_dir, pvd_filename), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')

        for time_index, filename in pvd_entries:
            f.write(f'    <DataSet timestep="{time_index}" group="" part="0" file="{filename}"/>\n')

        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')


if __name__ == "__main__":

    file_ending=sys.argv[1]
    name_ending=sys.argv[2]
    data = np.loadtxt(f"/scratch/jpelz/ma-pinns/TrackGen/History_n{name_ending}_t150.txt", delimiter=" ")
    pvd_entries = export_unstructured_series(data, output_dir="/scratch/jpelz/ma-pinns/TrackGen/binned_data", file_ending=file_ending)
    write_pvd(pvd_entries, output_dir="/scratch/jpelz/ma-pinns/TrackGen/binned_data", pvd_filename=f"particles_{file_ending}.pvd")
    
    # data = np.loadtxt(f"/scratch/jpelz/da-challenge/DA_CASE01/velocity_files/DA_CASE01_TR_ppp_0_{name_ending}_velocities.dat", skiprows=1)
    # pvd_entries = export_unstructured_series(data, output_dir="/scratch/jpelz/da-challenge/DA_CASE01/particles_vtu", file_ending=file_ending)
    # write_pvd(pvd_entries, output_dir="/scratch/jpelz/da-challenge/DA_CASE01/particles_vtu", pvd_filename=f"particles_{file_ending}.pvd")
