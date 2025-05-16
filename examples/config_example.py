import os
import time
import numpy as np
import torch

from code.architectures import FCN, FCN_HBC, FCN_RFF
from code import surface_samplers
from code.PINN import PINN_3D, halfcylinder_3d
from code.config import Config
from code.evaluation import evaluatePINN

def run_PINN(conf: Config):
    start=time.time()


    #Create PINN:
    if conf.architecture.model=="fully_connected":
      network=FCN
    if conf.architecture.model=="HBC":
      network=FCN_HBC
    if conf.architecture.model=="fourier_feature":
      network=FCN_RFF

    PINN=PINN_3D(network, conf.name, 640, conf.architecture.nr_layers, conf.architecture.nr_hidden,
                 amp_enabled=conf.training.amp_enabled)

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
      data=np.loadtxt(conf.data.file, delimiter=" ")
      t_filter=np.bitwise_and(data[:,4]>=conf.domain.t_min, data[:,4]<=conf.domain.t_max)
      data=data[t_filter,:]

      PINN.add_data_points(data)

    #Boundary Points:
    if hasattr(conf, 'boundary'):
      sampler=surface_samplers.halfcylinder([0,0,0], 0.125, 1, conf.domain.t_min, conf.domain.t_max)
      PINN.add_boundary_condition(sampler, conf.boundary.nr_points, weight=conf.boundary.weight)

    #Physics Points:
    if hasattr(conf, 'physics'):
      PINN.add_physics_points(conf.physics.nr_points,
                              conf.physics.nr_batches,
                              geometry=halfcylinder_3d,
                              weight=conf.physics.weight)

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

    if hasattr(conf.training, 'scheduler'):
      if conf.training.scheduler.type=="cosine_annealing_warm_restarts":
        PINN.add_scheduler(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(PINN.optimizer,
                                                                                T_0=conf.training.scheduler.T_0,
                                                                                T_mult=conf.training.scheduler.T_mult))

    #add plots
    t_plot=(lb[3]+ub[3])/2
    PINN.add_2D_plot([0,1], 0, t_plot, plot_data=True, resolution=[640, 240])

    #Train PINN
    PINN.train(conf.training.epochs, conf.training.print_freq, conf.training.plot_freq, 
               conf.training.point_update_freq)
    end=time.time()

    duration=end-start

    if conf.validation.validate:
      PINN_dir=os.getcwd()
      if not os.path.exists(f'{PINN_dir}/Validation'):
        os.makedirs(f'{PINN_dir}/Validation')
      os.chdir(f'{PINN_dir}/Validation')
      eval_model=torch.load(f'{PINN_dir}/{PINN.NAME}.pt',weights_only=False, map_location=torch.device('cpu'))
      evaluatePINN(eval_model,
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
                   duration=duration)
      os.chdir(PINN_dir)

if __name__ == "__main__":
    config=Config('/scratch/jpelz/ma-pinns/config.yaml')
    config.run(PINN_Setup = run_PINN) 
