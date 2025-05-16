import sys
sys.path.append('/scratch/jpelz/ma-pinns/_project/src')



import numpy as np
import torch

import os
import time


from DaFlowPINN.boundaries import surface_samplers, adf

from DaFlowPINN.boundaries import surface_samplers, adf
from DaFlowPINN.boundaries.internal_geometries import halfcylinder_3d
from DaFlowPINN.model.core import PINN_3D, save_predictable
from DaFlowPINN.model.architectures import FCN, WangResNet
from DaFlowPINN.config import Config

from DaFlowPINN.post.evaluation import evaluatePINN, plotTruthXY
from DaFlowPINN.post.export import export_h5, export_vts

from DaFlowPINN.training.optim.scheduler import ReduceLROnPlateau_custom, WarmupScheduler

def PINN_Setup(conf: Config):
    start=time.time()

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
      wall = adf.wall(0,0.5/8)
      circle = adf.circle(0.125/8,(0.5/8,0.5),(0,1), order = 1, ratio=3/8)
      bc_sdf = adf.subtract(circle, wall, order=1)
    else:
      bc_sdf = None

    if conf.architecture.model == 'fully_connected':
      network = FCN
    elif conf.architecture.model == 'WangResNet':
      network = WangResNet
    else:
      raise ValueError("Model not implemented")

    PINN=PINN_3D(network, conf.name, 640, conf.architecture.nr_layers, conf.architecture.nr_hidden,
                amp_enabled=conf.training.amp_enabled, hardBC_sdf=bc_sdf,
                  fourier_feature=rff, **kwargs)


    print(f'PINN initialized (time: {time.time() - start} s)')
    clock = time.time()

    #Define Domain

    lb=[conf.domain.x_min,
        conf.domain.y_min,
        conf.domain.z_min,
        conf.domain.t_min]
    ub=[conf.domain.x_max,
        conf.domain.y_max,
        conf.domain.z_max,
        conf.domain.t_max]
    PINN.define_domain(lb, ub)


    #Add Data Points
    if hasattr(conf, 'data'):
      data=np.loadtxt(conf.data.file, delimiter=" ", skiprows=1)
      t_filter=np.bitwise_and(data[:,4]>=(conf.domain.t_min-0.01), data[:,4]<=(conf.domain.t_max+0.01))
      data=data[t_filter,:]

      batch_size = conf.data.batch_size if hasattr(conf.data, 'batch_size') else None

      PINN.add_data_points(data, batch_size)




    #Boundary Points:
    if hasattr(conf, 'boundary'):
      sampler=surface_samplers.halfcylinder([0,0,0], 0.125, 1, conf.domain.t_min, conf.domain.t_max)
      batch_size = conf.boundary.batch_size if hasattr(conf.boundary, 'batch_size') else None
      PINN.add_boundary_condition(sampler, conf.boundary.nr_points, weight=conf.boundary.weight, batch_size=batch_size)

    #Physics Points:
    if hasattr(conf, 'physics'):
      batch_size = conf.physics.batch_size if hasattr(conf.physics, 'batch_size') else None
      PINN.add_physics_points(conf.physics.nr_points, batch_size, geometry=halfcylinder_3d(r=0.125),
                              weight=conf.physics.weight, n_acc=conf.physics.nr_batches)
      
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


    #add plots
    t_plot=14.70
    z_plot=-0.01

    #velocity magnitude & pressure in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=False, resolution=[640, 240], centered2 = False)
    #velocity magnitude & pressure in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=True, resolution=[640, 240], centered2 = False)

    #u-plot in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=False, resolution=[640, 240], component1=1, component2=None, centered1=True)

    #v- & w-plot in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=False, resolution=[640, 240], component1=2, component2=3, centered1=True, centered2=True)



    #Train PINN

    autoweight = conf.training.autoweight.type if hasattr(conf.training, 'autoweight') else None
    freq = conf.training.autoweight.freq if hasattr(conf.training, 'autoweight') else None

    PINN.train(conf.training.epochs, conf.training.print_freq, conf.training.plot_freq,
              conf.training.point_update_freq, conf.training.keep_percentage, gradient_accumulation=False, autoweight_scheme=autoweight, autoweight_freq=freq)
    end=time.time()

    save_predictable(PINN, f'{conf.name}_predictable.pt')

    duration=f"{(end-start)/60} minutes"
    print(f'Training complete (time: {duration} minutes)')

    #Export result
    x = np.linspace(conf.domain.x_min, conf.domain.x_max, 320)
    y = np.linspace(conf.domain.y_min, conf.domain.y_max, 120)
    z = np.linspace(conf.domain.z_min, conf.domain.z_max, 50)
    t = np.linspace(conf.domain.t_min, conf.domain.t_max, 6)

    dx = (x.max() - x.min()) / (len(x) - 1)
    dy = (y.max() - y.min()) / (len(y) - 1)
    dz = (z.max() - z.min()) / (len(z) - 1)

    y_grid, x_grid, z_grid = np.meshgrid(y, x, z)

    print('Exporting result...')
    export_h5(PINN.get_forward_callable(), x_grid, y_grid, z_grid, t, dx, dy, dz, f'{conf.name}_result_t{t:.2f}.h5')

    export_vts(PINN.get_forward_callable(), x_grid, y_grid, z_grid, t, f'{conf.name}')


if __name__ == "__main__":
  conf_file = sys.argv[1]
  conf = Config(conf_file)
  conf.run(PINN_Setup)