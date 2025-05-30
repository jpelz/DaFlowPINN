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
from DaFlowPINN.post.export import export_h5, export_vts, export_vti

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
      train_test_file =  conf.data.train_test_file if hasattr(conf.data, 'train_test_file') else None
      data=np.loadtxt(conf.data.file, delimiter=" ", skiprows=1)
      t_filter=np.bitwise_and(data[:,4]>=(conf.domain.t_min-0.01), data[:,4]<=(conf.domain.t_max+0.01))
      data=data[t_filter,:]

      batch_size = conf.data.batch_size if hasattr(conf.data, 'batch_size') else None

      PINN.add_data_points(data, batch_size, train_test_file=train_test_file)




    #Boundary Points:
    if hasattr(conf, 'boundary'):
      sampler=surface_samplers.halfcylinder([0,0,0], 0.125, 1, conf.domain.t_min, conf.domain.t_max)
      batch_size = conf.boundary.batch_size if hasattr(conf.boundary, 'batch_size') else None
      PINN.add_boundary_condition(sampler, conf.boundary.nr_points, weight=conf.boundary.weight, batch_size=batch_size)


    #Physics Points:
    if hasattr(conf, 'physics'):
      numerical = conf.physics.numerical if hasattr(conf.physics, 'numerical') else False
      batch_size = conf.physics.batch_size if hasattr(conf.physics, 'batch_size') else None
      PINN.add_physics_points(conf.physics.nr_points, batch_size, geometry=halfcylinder_3d(r=0.125),
                              weight=conf.physics.weight, numerical=numerical, n_acc=conf.physics.nr_batches,
)

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


    #add plots
    t_plot=14.7
    z_plot=-0.01

    #velocity magnitude & pressure in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=False, resolution=[640, 240], centered2 = False)
    #velocity magnitude & pressure in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=True, resolution=[640, 240], centered2 = False)

    #u-plot in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=True, resolution=[640, 240], component1=1, component2=1, centered1=False, centered2=False)

    #v in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=True, resolution=[640, 240], component1=2, component2=2, centered1=True, centered2=True)

    #w in XY plane
    PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=True, resolution=[640, 240], component1=3, component2=3, centered1=True, centered2=True)

    #Q-Criterion in XY plane
    #PINN.add_2D_plot([0,1], z_plot, t_plot, plot_data=False, resolution=[640, 240], component1=15, component2=None)

    #vorticity magnitude in XY plane
    #PINN.add_2D_plot([0,1], 0, t_plot, plot_data=False, resolution=[640, 240], component1=14, component2=None)

    # #Q-Criterion in 3D
    # PINN.add_Q_crit_plot(t_plot, resolution=[100, 50, 25], iso_value = 0.1)



    # if conf.validation.validate:
    #   plotTruthXY(conf.validation.file, 0, t_plot)

    #PINN.load(f'{conf.name}_best_state.pt')

    #Train PINN

    autoweight = conf.training.autoweight.type if hasattr(conf.training, 'autoweight') else None
    freq = conf.training.autoweight.freq if hasattr(conf.training, 'autoweight') else None

    print(f'Setup complete (time: {time.time() - start} s)')

    PINN.train(conf.training.epochs, conf.training.print_freq, conf.training.plot_freq,
              conf.training.point_update_freq, conf.training.keep_percentage, autoweight_scheme=autoweight, autoweight_freq=freq)
    end=time.time()

    save_predictable(PINN, f'{conf.name}_predictable.pt')

    duration=f"{(end-start)/60} minutes"

    #Export result
    nx = 320
    ny = 120
    nz = 40
    #nt = int(len(np.unique(data[:,4])))

    dx = (conf.domain.x_max - conf.domain.x_min) / nx
    dy = (conf.domain.y_max - conf.domain.y_min) / ny
    dz = (conf.domain.z_max - conf.domain.z_min) / nz

    x = np.linspace(conf.domain.x_min+dx/2, conf.domain.x_max-dx/2, nx)
    y = np.linspace(conf.domain.y_min+dy/2, conf.domain.y_max-dy/2, ny)
    z = np.linspace(conf.domain.z_min+dz/2, conf.domain.z_max-dz/2, nz)
    t = np.linspace(conf.domain.t_max-0.2, conf.domain.t_max, 3)


    y_grid, x_grid, z_grid = np.meshgrid(y, x, z)

    print('Exporting result...')
    #export_h5(PINN.get_forward_callable(), x_grid, y_grid, z_grid, t, dx, dy, dz, f'{conf.name}_result_t{t:.2f}.h5')

    export_vti(PINN.get_forward_callable(), x_grid, y_grid, z_grid, t, f'{conf.name}')


    if conf.validation.validate:
      print('Evaluating PINN...')
      PINN_dir=os.getcwd()
      if not os.path.exists(f'{PINN_dir}/Validation'):
        os.makedirs(f'{PINN_dir}/Validation')
      os.chdir(f'{PINN_dir}/Validation')
      evaluatePINN(PINN.get_forward_callable(),
                  datafile=conf.validation.file,
                  outputfile=f"{conf.name}_stats.txt",
                  tmin=conf.domain.t_min,
                  tmax=conf.domain.t_max,
                  nt_max=conf.validation.nt_max,
                  zmin=conf.domain.z_min,
                  zmax=conf.domain.z_max,
                  nz_max=conf.validation.nz_max,
                  nx_max=conf.validation.nx_max,
                  ny_max=conf.validation.ny_max,
                  z_plot=conf.validation.z_plot,
                  t_plot=conf.validation.t_plot,
                  name=PINN.NAME,
                  duration=duration,
                  loss_NS = PINN.hist_ns[-1],
                  loss_BC = PINN.hist_bc[-1],
                  loss_data = PINN.hist_data[-1],
                  loss_total = PINN.hist_total[-1])
      os.chdir(PINN_dir)


if __name__ == "__main__":
  conf_file = sys.argv[1]
  conf = Config(conf_file)
  conf.run(PINN_Setup)