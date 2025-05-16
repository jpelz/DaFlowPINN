import numpy as np

from DaFlowPINN.model.architectures import FCN
from DaFlowPINN.boundaries import surface_samplers
from DaFlowPINN.boundaries.internal_geometries import halfcylinder_3d
from DaFlowPINN.model.core import PINN_3D

def run_PINN():
    
    #Define a PINN using a fully connected network
    PINN=PINN_3D(model=FCN, NAME="Test_Simple", Re=640, 
                 N_LAYERS=4, N_NEURONS=256, 
                 hardBC_sdf=None, fourier_feature=False)
    
    #Define Domain
    lb=[-0.5, -1.5, -0.5, 14.5] #Lower bound of the domain (x, y, z, t)
    ub=[7.5, 1.5, 0.5, 15.0] #Upper bound of the domain (x, y, z, t)
    PINN.define_domain(lb, ub)

    #Add Data Points
    data=np.loadtxt("examples/datasets/halfylinder_Re640/HalfcylinderTracks_p010_t14.5-15.dat", delimiter=" ")
    PINN.add_data_points(data)

    #Add Boundary Points:
    sampler=surface_samplers.halfcylinder(center=[0,0,0], r=0.125, h=1, tmin=lb[3], tmax=ub[3])
    PINN.add_boundary_condition(sampler, N_BC_POINTS=4096, weight=0.1)

    #Add Physics Points:
    PINN.add_physics_points(N_COLLOCATION=8192, batch_size=1024, geometry=halfcylinder_3d(r=0.125), weight=1e-6)

    #Select Optimizer
    PINN.add_optimizer("adam", lr=1e-3)

    #Add plots (e.g. 3 XY-plots at t=14.5, 14.75, 15.0 and z=0) 
    t_plot=[14.5, 14.75, 15.0] #select timesteps to plot

    for t in t_plot:
        PINN.add_2D_plot(plot_dims=[0,1], dim3_slice=0, t_slice=t, plot_data=True, resolution=[640, 240])

    #Train PINN
    PINN.train(epochs=1000, print_freq=100, plot_freq=200, autoweight_scheme=3)


if __name__ == "__main__":
    run_PINN()
