import os
import time
import numpy as np
import torch
import sys

sys.path.append('/scratch/jpelz/ma-pinns')

from _project.src.DaFlowPINN.boundaries import surface_samplers, adf
from _project.src.DaFlowPINN.model.core import PINN_3D, save_predictable
from _project.src.DaFlowPINN.model.architectures import FCN, WangResNet
from _project.src.DaFlowPINN.config.config import Config

from _project.src.DaFlowPINN.post.export import export_h5, export_vts, export_vti

from _project.src.DaFlowPINN.training.optim.scheduler import ReduceLROnPlateau_custom, WarmupScheduler


def DA_CASE01_PINN(conf: Config):
    start=time.time()

    h = 0.1
    Re = 40000
    l_scale=2*h
    u_scale=0.2
    rho=1000
    p_scale=rho*u_scale**2

    
    dt = 5.9160E-03
    t_mid = 24*dt
    conf.domain.t_max = t_mid + (conf.domain.nt-1)/2*dt
    conf.domain.t_min = t_mid - (conf.domain.nt-1)/2*dt


    kwargs = {}

    if hasattr(conf.architecture, 'fourier_feature'):
      kwargs['mapping_size'] = conf.architecture.fourier_feature.mapping_size
      kwargs['scale_x'] = conf.architecture.fourier_feature.scale_x
      kwargs['scale_y'] = conf.architecture.fourier_feature.scale_y
      kwargs['scale_z'] = conf.architecture.fourier_feature.scale_z
      kwargs['scale_t'] = conf.architecture.fourier_feature.scale_t

      rff = True
    else:
      rff = False

    if conf.architecture.hard_bc:
      wall1 = adf.wall(dim=1, order=1)
      wall2 = adf.wall(dim=2, order=1)
      bc_sdf = adf.unite(wall1, wall2, order=1)
    else: 
      bc_sdf = None

    if conf.architecture.model == 'fully_connected':
      network = FCN
    elif conf.architecture.model == 'WangResNet':
      network = WangResNet
    else:
      raise ValueError("Model not implemented")

    PINN=PINN_3D(network, conf.name, Re,  conf.architecture.nr_layers, conf.architecture.nr_hidden,
                 amp_enabled=conf.training.amp_enabled, hardBC_sdf=bc_sdf,
                 fourier_feature=rff, **kwargs)

    #Define Domain

    lb=[0, 0, 0, conf.domain.t_min]
    ub=[h, h, h, conf.domain.t_max]
    PINN.define_domain(lb, ub)

    PINN.set_dimensionless(l_scale, u_scale, p_scale)

    #Add Data Points
    if hasattr(conf, 'data'):
      train_test_file =  conf.data.train_test_file if hasattr(conf.data, 'train_test_file') else None
      data=np.loadtxt(conf.data.file, delimiter=" ", skiprows=1)
      t_filter=np.bitwise_and(data[:,4]>=(conf.domain.t_min-0.001), data[:,4]<=(conf.domain.t_max+0.001))
      data=data[t_filter,:]

      batch_size = conf.data.batch_size if hasattr(conf.data, 'batch_size') else None
      weight = conf.data.weight if hasattr(conf.data, 'weight') else 1

      PINN.add_data_points(data, batch_size, train_test_file=train_test_file, weight=weight)

    #Boundary Points:
    if hasattr(conf, 'boundary'):
      sampler1=surface_samplers.wall(dim1 = 0, dim2 = 1, dim1_min = 0, dim1_max = h, dim2_min = 0, dim2_max = h, dim3_value = 0, t_min = conf.domain.t_min, t_max = conf.domain.t_max)
      sampler2=surface_samplers.wall(dim1 = 0, dim2 = 2, dim1_min = 0, dim1_max = h, dim2_min = 0, dim2_max = h, dim3_value = 0, t_min = conf.domain.t_min, t_max = conf.domain.t_max)
      sampler=surface_samplers.combine([sampler1, sampler2])
      batch_size = conf.boundary.batch_size if hasattr(conf.boundary, 'batch_size') else None
      PINN.add_boundary_condition(sampler, conf.boundary.nr_points, weight=conf.boundary.weight, batch_size=batch_size)

    #Physics Points:
    if hasattr(conf, 'physics'):
      numerical = conf.physics.numerical if hasattr(conf.physics, 'numerical') else False
      batch_size = conf.physics.batch_size if hasattr(conf.physics, 'batch_size') else None
      PINN.add_physics_points(conf.physics.nr_points,
                              batch_size,
                              weight=conf.physics.weight, numerical=numerical, p_scale=4/p_scale, n_acc=conf.physics.nr_batches)

      if hasattr(conf.physics, 'collocation_growth'):
        kwargs = {}
        if conf.physics.collocation_growth.type == 'exponential':
          kwargs['epsilon'] = conf.physics.collocation_growth.epsilon
        PINN.add_collocation_growth(conf.physics.nr_points,
                                    conf.physics.collocation_growth.max_points,
                                    conf.physics.collocation_growth.epoch_start,
                                    conf.physics.collocation_growth.epoch_end,
                                    increase_scheme=conf.physics.collocation_growth.type,
                                    **kwargs)

    #Optimizer
    PINN.add_optimizer(conf.training.optim.optimizer, lr=conf.training.optim.lr)

    if hasattr(conf.training, 'posttraining'):
      PINN.add_postTraining(conf.training.posttraining.lr, conf.training.posttraining.epochs, conf.training.posttraining.lbfgs_type)

    if hasattr(conf.training, 'scheduler'):
      if conf.training.scheduler.type=="cosine_annealing_warm_restarts":
        PINN.add_scheduler(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(PINN.optimizer,
                                                                                T_0=conf.training.scheduler.T_0,
                                                                                T_mult=conf.training.scheduler.T_mult))
        
      if conf.training.scheduler.type=="ReduceLROnPlateau_custom":
        PINN.add_scheduler(ReduceLROnPlateau_custom(PINN.optimizer,
                                                    epoch_window=conf.training.scheduler.epoch_window,
                                                    factor=conf.training.scheduler.factor,
                                                    threshold=conf.training.scheduler.threshold,
                                                    cooldown=conf.training.scheduler.cooldown))
      
      if conf.training.scheduler.type=="ExponentialLR":
        scheduler1 = WarmupScheduler(PINN.optimizer, epochs = 50)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(PINN.optimizer, conf.training.scheduler.decay)
        PINN.add_scheduler(torch.optim.lr_scheduler.SequentialLR(PINN.optimizer, schedulers=[scheduler1, scheduler2], milestones=[50]))


    #Plots
    t_plot=t_mid

    #XY velocity mag & pressure plot
    PINN.add_2D_plot([0,1], h/2, t_plot, plot_data=True, resolution=[100,100], component1=0, component2=4)
    #YZ velocity mag & pressure plot
    PINN.add_2D_plot([1,2], h/2, t_plot, plot_data=True, resolution=[100,100], component1=0, component2=4)

    #XY velocity mag & pressure plot
    PINN.add_2D_plot([0,1], h/2, t_plot, plot_data=False, resolution=[100,100], component1=0, component2=4)
    #YZ velocity mag & pressure plot
    PINN.add_2D_plot([1,2], h/2, t_plot, plot_data=False, resolution=[100,100], component1=0, component2=4)

    #XY u plot
    PINN.add_2D_plot([0,1], h/2, t_plot, plot_data=True, resolution=[100,100], component1=1, component2=1)
    #YZ u plot
    PINN.add_2D_plot([1,2], h/2, t_plot, plot_data=True, resolution=[100,100], component1=1, component2=1)

    #XY v plot
    PINN.add_2D_plot([0,1], h/2, t_plot, plot_data=True, resolution=[100,100], component1=2, component2=2, centered1=True)
    #YZ v plot
    PINN.add_2D_plot([1,2], h/2, t_plot, plot_data=True, resolution=[100,100], component1=2, component2=2, centered1=True)

    #XY w plot
    PINN.add_2D_plot([0,1], h/2, t_plot, plot_data=True, resolution=[100,100], component1=3, component2=3, centered1=True, centered2=True)
    #YZ w plot
    PINN.add_2D_plot([1,2], h/2, t_plot, plot_data=True, resolution=[100,100], component1=3, component2=3, centered1=True, centered2=True)



    #Train PINN
    autoweight = conf.training.autoweight.type if hasattr(conf.training, 'autoweight') else None
    freq = conf.training.autoweight.freq if hasattr(conf.training, 'autoweight') else None

    if hasattr(conf.training, 'init_state'):
        PINN.load(conf.training.init_state)

    PINN.train(conf.training.epochs, conf.training.print_freq, conf.training.plot_freq, 
              conf.training.point_update_freq, conf.training.keep_percentage, autoweight_scheme=autoweight, autoweight_freq=freq)
    
    end=time.time()

    duration=(end-start)/60

    print(f'Finished run of {conf.name} in {duration} minutes.')

    PINN.load(f"{conf.name}_final_state.pt")
    save_predictable(PINN, f'{conf.name}_predictable.pt')


    #export
    #Export result
    x = np.linspace(0, 0.1, 101)
    y = np.linspace(0.1E-3, 0.0991, 100)
    z = np.linspace(0.1E-3, 0.0991, 100)
    t = np.linspace(conf.domain.t_min, conf.domain.t_max, int(conf.domain.nt))

    dx = (x.max() - x.min()) / (len(x) - 1)
    dy = (y.max() - y.min()) / (len(y) - 1)
    dz = (z.max() - z.min()) / (len(z) - 1)

    y_grid, x_grid, z_grid = np.meshgrid(y, x, z)

    print('Exporting result...')
    #export_h5(PINN.get_forward_callable(), x_grid, y_grid, z_grid, t_mid, dx, dy, dz, f'{conf.name}_result_t{t_mid:.2f}.h5')

    export_vti(PINN.get_forward_callable(), x_grid, y_grid, z_grid, t, f'{conf.name}')

    

if __name__ == "__main__":
 conf_file = sys.argv[1]
 conf = Config(conf_file)
 conf.run(DA_CASE01_PINN, print_to_log=True)
    
