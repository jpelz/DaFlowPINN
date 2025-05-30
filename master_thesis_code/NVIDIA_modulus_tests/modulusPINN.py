import os
import warnings

import numpy as np

from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box
from CustomGeo import InfiniteCylinder

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    SupervisedGridConstraint,
    PointwiseConstraint,
    PointwiseInteriorConstraint
)
from modulus.sym.dataset import DictGridDataset
from modulus.sym.loss import PointwiseLossNorm

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from modulus.sym.utils.io.vtk import var_to_polyvtk

from matplotlib import pyplot as plt
import scipy.interpolate

class Custom2DPlotter(InferencerPlotter):
    def __call__(self, invar, outvar):
        "Plotter for 2D data specified by keys"

        ndim=2

        keys = list(invar.keys())

        invar_plot={
            keys[0]: invar[keys[0]],
            keys[1]: invar[keys[1]]
        }
        

        # interpolate 2D data onto grid
        if ndim == 2:
            extent, outvar = self._interpolate_2D(800, 300, invar_plot, outvar)

        # make plots
        dims = list(invar.keys())
        fs = []

        if "u" in outvar and "v" in outvar and "w" in outvar:
            outvar["mag"] = np.sqrt(outvar["u"]**2 + outvar["v"]**2 + outvar["w"]**2)

        for k in outvar:
            f = plt.figure(figsize=(8, 3), dpi=200)
            if ndim == 1:
                plt.plot(invar[dims[0]][:, 0], outvar[:, 0])
                plt.xlabel(dims[0])
            elif ndim == 2:
                plt.imshow(outvar[k].T, origin="lower", extent=extent)
                plt.xlabel(dims[0])
                plt.ylabel(dims[1])
                plt.colorbar()
            plt.title(k)
            plt.tight_layout()
            fs.append((f, k))

        return fs
    
    def _interpolate_2D(self, size1, size2, invar, *outvars):
        "Interpolate 2D outvar solutions onto a regular mesh, copied from modulus.sym.utils.io but modified for multiple spacing"

        assert len(invar) == 2

        # define regular mesh to interpolate onto
        xs = [invar[k][:, 0] for k in invar]
        extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size1),
            np.linspace(extent[2], extent[3], size2),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (xs[0], xs[1]), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp

def testgeo():
    # make geometry
    radius=0.125
    #x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    cyl=InfiniteCylinder(center=(0, 0, 0), radius=radius, height=1)
    cut = Box((0,-radius,-0.5), (radius,radius,0.5))
    boundary=cyl-cut

    print("boundary created")

    outside=Box((-0.5,-1.5,-0.5),(7.5,1.5,0.5))
    flow_region=outside-cyl

    #sample boundary points
    np=1000
    bc_points=boundary.sample_boundary(nr_points=np)

    var_to_polyvtk(bc_points, "boundary")



@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.000390625, rho=1.0, dim=3, time=True)

    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # make geometry
    radius=0.125
    TMIN=14.7
    TMAX=15.0

    x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
    t_range = {t: (TMIN, TMAX)}

    
    cyl=InfiniteCylinder(center=(0, 0, 0), radius=radius, height=1)
    cut = Box((0,-radius,-0.5), (radius,radius,0.5))
    boundary=cyl-cut

    outside=Box((-0.5,-1.5,-0.5),(7.5,1.5,0.5))
    flow_region=outside-cyl

    flow_domain=Domain()

    bc=PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.BC,
        parameterization=t_range
    )
    flow_domain.add_constraint(bc,"boundary")

    #Measurement Data
    DATA_FILE="/scratch/jpelz/ma-pinns/TrackGen/History_n1000_t150.txt"

    data=np.loadtxt(DATA_FILE, delimiter=" ")
    t_filter=np.bitwise_and(data[:,4]>=TMIN, data[:,4]<=TMAX)
    data=data[t_filter,:]

    X={
        "x": np.expand_dims(data[:,1], -1),
        "y": np.expand_dims(data[:,2], -1),
        "z": np.expand_dims(data[:,3], -1),
        "t": np.expand_dims(data[:,4], -1)
    }

    Y={
        "u": np.expand_dims(data[:,5], -1),
        "v": np.expand_dims(data[:,6], -1),
        "w": np.expand_dims(data[:,7], -1)
    }
    
    data_driven = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=X,
        outvar=Y,
        batch_size=cfg.batch_size.Data
    )
    flow_domain.add_constraint(data_driven, "data_driven")

    # make physics constraint
    physics = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=flow_region,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.Physics,
        parameterization=t_range
    )
    flow_domain.add_constraint(physics, "physics")


    # Predicted plots
    NT=3
    z_plot=[0]
    t_plot=np.linspace(TMIN, TMAX, NT)

    for t in t_plot:
        for z in z_plot:
            x_plot=np.linspace(-0.5, 7.5, 800)
            y_plot=np.linspace(-1.5, 1.5, 300)
            

            xp,yp,zp,tp=np.meshgrid(x_plot, y_plot, z, t)

            invar_plot={
                "x": np.expand_dims(xp.flatten(), -1),
                "y": np.expand_dims(yp.flatten(), -1),
                "z": np.expand_dims(zp.flatten(), -1),
                "t": np.expand_dims(tp.flatten(), -1)
            }

            grid_inference = PointwiseInferencer(
                    nodes=nodes,
                    invar=invar_plot,
                    output_names=["u", "v", "w", "p"],
                    batch_size=1024,
                    plotter=Custom2DPlotter()
                )
            flow_domain.add_inferencer(grid_inference, f"inf_data_t{t:.2f}_z{z:.2f}")

            #Validation node
            datafile="/scratch/jpelz/ma-pinns/TrackGen/halfcylinder.nc"

            x_val, y_val, z_val, t_val, u_val, v_val, w_val=load_data(

            validator = PointwiseValidator(
                    nodes=nodes,
                    invar=invar_plot,
                    true_outvar=
                    batch_size=1024,
                    plotter=Custom2DPlotter()
                    )

    # make solver
    slv = Solver(cfg, flow_domain)

    # start solver
    slv.solve()
    
    


if __name__=="__main__":
    run()
    

    
