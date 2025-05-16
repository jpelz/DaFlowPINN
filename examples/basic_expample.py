import numpy as np

from DaFlowPINN.model.architectures import FCN
from DaFlowPINN.boundaries import surface_samplers
from DaFlowPINN.model.core import PINN_3D, halfcylinder_3d

def run_PINN():
    
    #Define a PINN using a fully connected network
    PINN=PINN_3D(model=FCN, NAME="Test_Simple", Re=640, N_LAYERS=4, N_NEURONS=256, hardBC_sdf=None, fourier_feature=False)
    
    #Define Domain
    lb=[-0.5, -1.5, -0.5, 0] #Lower bound of the domain (x, y, z, t)
    ub=[7.5, 1.5, 0.5, 1] #Upper bound of the domain (x, y, z, t)
    PINN.define_domain(lb, ub)

    #Add Data Points
    data=np.loadtxt("/datasets/halfylinder_Re640/HalfcylinderTracks_p010_t14.5-15.dat", delimiter=" ")
    PINN.add_data_points(data)

    #Add Boundary Points:
    sampler=surface_samplers.halfcylinder(center=[0,0,0], r=0.125, h=1, tmin=lb[3], tmax=ub[3])
    PINN.add_boundary_condition(sampler, N_BC_POINTS=4096, weight=0.1)

    #Add Physics Points:
    PINN.add_physics_points(N_COLLOCATION=8192, N_BATCHES=4, geometry=halfcylinder_3d, weight=1e-6)

    #Optimizer
    PINN.add_optimizer("adam", lr=1e-3)


    #Add plots (e.g. 3 XY-plots at t=5, 10, 15 and z=0) 
    t_plot=[5, 10, 15] #select timesteps to plot

    for t in t_plot:
        PINN.add_2D_plot(plot_dims=[0,1], dim3_slice=0, t_slice=t, plot_data=True, resolution=[640, 240])

    #Train PINN
    PINN.train(epochs=1000, print_freq=100, plot_freq=200)


if __name__ == "__main__":
    run_PINN()
