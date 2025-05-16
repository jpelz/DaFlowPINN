## Third-Party Code and Licenses

### 1. training/optim/soap.py
Originally from https://github.com/nikhilvyas/SOAP  
Licensed under MIT License (© 2024 Nikhil Vyas)

---

### 2. training/optim/lbfgs.py
Originally from https://github.com/hjmshi/PyTorch-LBFGS  
Licensed under MIT License (© 2018 Hao-Jun Michael Shi)

---

### 3. training/utils.py
Originally from https://github.com/tum-pbs/ConFIG/  
Licensed under MIT License (© 2024 TUM Physics-based Simulation)

**Modifications:**  

`get_gradient_vector()`   
Modified to return a flat vector on demand → used by LBFGS-optimizer.