# DaFlowPINN

**DaFlowPINN** is a modular and extensible Python framework for building and training Physics-Informed Neural Networks (PINNs) for data assimilation of incompressible, time-dependend 3D fluid flows. It is primaly developed for postprocessing of Particle Tracking Velocimetry (PTV) data during a master's thesis. It includes components for model definition, training customization, boundary condition handling, and postprocessing utilities.

---

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

## 🚀 Features

- **Modular Architecture**: Swap in/out different architectures.
- **Parameterizable**: Configure the PINN using .yaml-files and run parameter sweeps to compare different settings.
- **Boundary Handling**: Tools for domain-specific boundary conditions. Option between soft and hard boundary conditions (using approximate distance functions).
- **Training Framework**: Includes autoweighting of losses and multiple optimizers (e.g., L-BFGS, SOAP).
- **Postprocessing Utilities**: Evaluate and visualize model results during training.
- **Example Scripts**: Jump-start your experiments using `examples/basic_example.py`.

---

## 🛠 Installation

From the project root:

```bash
pip install -e .
````

This installs the package in **editable mode**.

---

## 📂 Examples

Run basic example:

```bash
python examples/basic_example.py
```

---

## 📚 Usage

Here’s a minimal example of how to import and use core components:

```python
from DaFlowPINN import PINN_3D
from DaFlowPINN.config.config import load_config

config = load_config("config.yaml")
model = PINN_3D(config)
model.train()
```

---

## 🧪 Testing

*Coming soon*: Add tests under a `tests/` directory and use `pytest` for running unit tests.

---

## 📄 License

MIT License. See `LICENSE` file for details.

---