import sys
sys.path.append('/scratch/jpelz/ma-pinns/_project/src')



import numpy as np

import os




from DaFlowPINN.model.core import load_predictable
from DaFlowPINN.config import Config

from DaFlowPINN.post.evaluation import evaluatePINN
from DaFlowPINN.post.export import export_vti


def PINN_Setup(conf: Config):



    PINN = load_predictable(f"{conf.name}_predictable.pt")


    #Export result
    nx = 320
    ny = 120
    nz = 40

    dx = (conf.domain.x_max - conf.domain.x_min) / nx
    dy = (conf.domain.y_max - conf.domain.y_min) / ny
    dz = (conf.domain.z_max - conf.domain.z_min) / nz

    x = np.linspace(conf.domain.x_min+dx/2, conf.domain.x_max-dx/2, nx)
    y = np.linspace(conf.domain.y_min+dy/2, conf.domain.y_max-dy/2, ny)
    z = np.linspace(conf.domain.z_min+dz/2, conf.domain.z_max-dz/2, nz)
    t = np.linspace(conf.domain.t_min, conf.domain.t_max, 3)


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
                  name=PINN.NAME)
      os.chdir(PINN_dir)


if __name__ == "__main__":
  conf_files = sys.argv[1:]
  for conf_file in conf_files:
      conf = Config(conf_file)
      conf.run(PINN_Setup)