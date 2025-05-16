import numpy as np
import os

from DaFlowPINN.model.architectures import FCN
from DaFlowPINN.boundaries import surface_samplers
from DaFlowPINN.boundaries.internal_geometries import halfcylinder_3d
from DaFlowPINN import PINN_3D

def run_PINN():

    name = "BasicExample_HalfCylinder_Re640_p010"

    headdir = os.getcwd()


    if not os.path.exists(f'{headdir}/{name}'):
      os.makedirs(f'{headdir}/{name}')
    os.chdir(f'{headdir}/{name}')
    
    #Define a PINN using a fully connected network
    PINN=PINN_3D(model=FCN, NAME=name, Re=640, 
                 N_LAYERS=5, N_NEURONS=512, 
                 hardBC_sdf=None, fourier_feature=True,
                 scale_x=0.3, scale_y=1.2, scale_z=1.5, scale_t=0.3, mapping_size=512)
    
    #Define Domain
    lb=[-0.5, -1.5, -0.5, 14.5] #Lower bound of the domain (x, y, z, t)
    ub=[7.5, 1.5, 0.5, 14.9] #Upper bound of the domain (x, y, z, t)
    PINN.define_domain(lb, ub)

    #Add Data Points
    data=np.loadtxt(f"{os.path.dirname(os.path.abspath(__file__))}/datasets/halfylinder_Re640/HalfcylinderTracks_p010_t14.5-15.dat", delimiter=" ")
    PINN.add_data_points(data, batch_size=8192)

    #Add Boundary Points:
    sampler=surface_samplers.halfcylinder(center=[0,0,0], r=0.125, h=1, tmin=lb[3], tmax=ub[3])
    PINN.add_boundary_condition(sampler, N_BC_POINTS=4096)

    #Add Physics Points:
    PINN.add_physics_points(N_COLLOCATION=8192, batch_size=1024, geometry=halfcylinder_3d(r=0.125))

    #Select Optimizer
    PINN.add_optimizer("adam", lr=1e-3)

    #Add plots
    t_plot=14.75 #select timestep to plot

    #XY plot of mag and p
    PINN.add_2D_plot(plot_dims=[0,1], component1=0, component2=4, dim3_slice=0, t_slice=t_plot, plot_data=False, resolution=[640, 240])

    #XZ plot of mag and p
    PINN.add_2D_plot(plot_dims=[0,2], component1=0, component2=4, dim3_slice=0, t_slice=t_plot, plot_data=False, resolution=[640, 80], lb=[-0.5, -0.5], ub=[7.5, 0.5])

    #ZY plot of mag and p
    PINN.add_2D_plot(plot_dims=[2,1], component1=0, component2=4, dim3_slice=3.5, t_slice=t_plot, plot_data=False, resolution=[80, 240], lb=[-0.5, -1.5], ub=[0.5, 1.5])
    
    #Train PINN
    PINN.train(epochs=2000, print_freq=100, plot_freq=1000, autoweight_scheme=3, autoweight_freq=1)

    os.chdir(headdir)


if __name__ == "__main__":
    run_PINN()
