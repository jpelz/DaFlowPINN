# DaFlowPINN

<!-- Start Description -->

**DaFlowPINN** is a modular and extensible Python framework for building and training Physics-Informed Neural Networks (PINNs) for data assimilation of incompressible, time-dependend 3D fluid flows. It is primarily developed for postprocessing of Particle Tracking Velocimetry (PTV) data during a master's thesis. It includes components for model definition, training customization, boundary condition handling, and postprocessing utilities.

---
<!-- End Description -->

## 📦 Project Structure

```

DaFlowPINN/
├── boundaries/          # Define boundary conditions and samplers
├── post/                # Postprocessing: evaluation, plotting, export
├── model/               # Core model architecture and PINN logic
├── training/            # Training loop, loss functions, optimizers
│   └── optim/           # Custom optimizers and learning rate schedulers
├── config/              # Configuration management

````

---
<!-- Start Features -->

## 🚀 Features

- **Modular Architecture**: Swap in/out different architectures.
- **Parameterizable**: Configure the PINN using .yaml-files and run parameter sweeps to compare different settings.
- **Boundary Handling**: Tools for domain-specific boundary conditions. Option between soft and hard boundary conditions (using approximate distance functions).
- **Training Framework**: Includes autoweighting of losses and multiple optimizers (e.g., L-BFGS, SOAP).
- **Postprocessing Utilities**: Evaluate and visualize model results during training.
- **Example Scripts**: Jump-start your experiments using `examples/basic_example.py`.

---

## 🛠 Installation

For example in [Google Colab](http://colab.research.google.com/):

```bash
!git clone https://github.com/jpelz/DaFlowPINN.git
!pip install DaFlowPINN
````

---

## 📂 Examples

For showcasing the usage of the Framework, the PINN is used for Data Assimilation of a virtual Particle Tracking Velocimetry (PTV) Measurement. The particles where seeded into a DNS-Simulation of the flow around a halfcylinder [1] at Re = 640.

Run basic example:

```bash
python DaFlowPINN/examples/basic_example.py
```


Run example with configuration files:

```bash
python DaFlowPINN/examples/config_example.py DaFlowPINN/examples/basic_config.yaml
```

<!-- 
Available configurations (all with 10k particles per timestep):
- basic_conig.yaml (Vanilla PINN Setup)
- rff_config.yaml (PINN with Fourier Feature Embeddings)
- hbc_config.yaml (PINN with forced exact boundary conditions using approximate distance functions)
-->

To change the seeding density, epochs, optimizer,... edit the *.yaml file or create your own.

---

## 📚 Usage

Here’s a minimal example of how to import and use core components:

```python
import numpy as np
from DaFlowPINN import PINN_3D
from DaFlowPINN.model.architectures import FCN
from DaFlowPINN.boundaries import surface_samplers
from DaFlowPINN.boundaries.internal_geometries import halfcylinder_3d

#Define a PINN using a fully connected network (Re not relevant when no physics points)

PINN=PINN_3D(model=FCN, NAME=name, Re=640, 
                N_LAYERS=4, N_NEURONS=256)

#Define Domain
lb=[-0.5, -1.5, -0.5, 14.5] #Lower bound of the domain (x, y, z, t)
ub=[7.5, 1.5, 0.5, 15.0] #Upper bound of the domain (x, y, z, t)
PINN.define_domain(lb, ub)

#Add training data (has to be np.ndarray with the columns #ID,X,Y,Z,T,U,V,W)
data = np.loadtxt("DaFlowPINN/examples/datasets/halfylinder_Re640/HalfcylinderTracks_p010_t14.5-15.dat", delimiter=" ")
PINN.add_data_points(data)

#Add Boundary Points:
sampler=surface_samplers.halfcylinder(center=[0,0,0], r=0.125, h=1, tmin=lb[3], tmax=ub[3])
PINN.add_boundary_condition(sampler, N_BC_POINTS=4096)

#Add Physics Points:
PINN.add_physics_points(N_COLLOCATION=8192, batch_size=1024, geometry=halfcylinder_3d(r=0.125))


#Select optimizer
PINN.add_optimizer("adam", lr=1e-4)

#Add XY plot of velocity magnitude and pressure
PINN.add_2D_plot(component1=0, component2=4,
                 plot_dims=[0,1], dim3_slice=0, t_slice=14.75, resolution=[640, 240])

PINN.train(epochs=1000, print_freq=100, plot_freq=500)
```

Each run creates files for the loss history, the defined plots and more.  
The trained PINN is saved in `*_predictable.pt` and can be used as follows:

```python
import torch
from DaFlowPINN.model.core import load_predictable

PINN_trained = load_predictable("example_predictable.pt")

X = torch.tensor([2, 1, 0.2, 14.7]) #X,Y,Z,T

Y = PINN_trained(X)

```

---


## 📄 License

MIT License. See `LICENSE` file for details.

---

## References
[1] https://cgl.ethz.ch/research/visualization/data.php
