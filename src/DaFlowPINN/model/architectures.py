import torch
from torch import nn
from typing import Callable

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"

    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int):
        """
        Initializes the fully-connected network.

        Args:
            N_INPUT (int): Number of input features.
            N_OUTPUT (int): Number of output features.
            N_HIDDEN (int): Number of hidden units in each layer.
            N_LAYERS (int): Number of hidden layers.
        """

        super().__init__()
        #activation = nn.Tanh  # Tanh activation function
        activation = nn.SiLU

        def init_weights(m):
          if isinstance(m, nn.Linear):
              torch.nn.init.xavier_uniform_(m.weight)

        # Input layer
        self.in_layer = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )

        # Hidden layers
        self.hidden = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ) for _ in range(N_LAYERS - 1)
        ])

        # Output layer
        self.out_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

        self.in_layer.apply(init_weights)
        self.hidden.apply(init_weights)
        self.out_layer.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.in_layer(x)
        x = self.hidden(x)
        x = self.out_layer(x)
        return x

class HardBC(nn.Module):
    """Defines a hard boundary condition layer in PyTorch using a user-defined distance function. The distance function is applied only to the first 3 dimensions of the input tensor. The fourth dimension is treated as pressure and is not modified."""

    def __init__(self, bc_fun: Callable[[torch.Tensor], torch.Tensor], values=[0,0,0,0]):
        """
        Initializes the HardBC layer.
        Args:
            bc_fun (callable): A function that computes the distance to the boundary. It should take a tensor of shape (N, d) as input and return a tensor of shape (N, 1).
            values (list): A list of defined boundary values for each dimension. Default is [0, 0, 0, 0].
        """
        super().__init__()
        self.bc_fun = bc_fun
        self.offset = nn.Parameter(torch.tensor(values), requires_grad=False) #Defining as parameter saves the values of into the state_dict of the model

    def forward(self, x, x0):
        self.offset= nn.Parameter(self.offset.to(x.device), requires_grad=False)
        sdf_values=torch.maximum(self.bc_fun(x0),torch.tensor(0, device=x0.device))
        l=torch.stack((sdf_values, sdf_values, sdf_values,(1+max(sdf_values)-sdf_values)),dim=1) #stack to multiply u,v,w --> p*1    torch.ones_like(sdf_values)
        x = x*l+(1-l)*self.offset 
        return x



class RFF(nn.Module):
    """Defines a random Fourier feature layer in PyTorch. This layer is used to project the input tensor into a higher-dimensional space using random Fourier features. The projection is done using a linear transformation followed by sine and cosine transformations."""
    def __init__(self, n_freqs: int = 16, scales: list = [1, 1, 1, 1]):
        """
        Initializes the Random Fourier Feature (RFF) layer.

        Args:
            n_freqs (int): Number of random frequencies.
            scales (list): Scaling factors / gaussian standard deviations for each input dimension (x, y, z, t).
        """
        super().__init__()

        self.mapping_size = 2 * n_freqs

        # Random frequency matrices for each input dimension (not trainable)
        self.B_x = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)
        self.B_y = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)
        self.B_z = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)
        self.B_t = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)

        # Scaling factors for each input dimension (not trainable)
        self.scales = nn.Parameter(torch.tensor(scales), requires_grad=False)

    def rff(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the random Fourier feature mapping to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 4).

        Returns:
            torch.Tensor: Transformed tensor with random Fourier features.
        """
        # Concatenate scaled frequency matrices for all input dimensions
        B = torch.cat([
            self.B_x * self.scales[0],
            self.B_y * self.scales[1],
            self.B_z * self.scales[2],
            self.B_t * self.scales[3]
        ], dim=1).T  # Shape: (4, n_freqs)

        # Linear transformation and projection
        x_proj = 2 * torch.pi * x @ B  # Shape: (N, n_freqs)
        # Apply sine and cosine, then concatenate
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # Shape: (N, 2*n_freqs)

        return x_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RFF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 4).

        Returns:
            torch.Tensor: Output tensor with random Fourier features.
        """
        x = self.rff(x)
        return x

#Additional architectures:
        
    
class WangResNet(nn.Module):
    """https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs"""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, **kwargs):
        super().__init__()

        self.activation = nn.Tanh

        self.encoder_1 = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), self.activation())
        self.encoder_2 = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), self.activation())

        self.input_layer = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), self.activation())

        self.hidden_layers = nn.ModuleList(nn.Sequential(
            nn.Linear(N_HIDDEN, N_HIDDEN), self.activation()) for _ in range(N_LAYERS - 2))

        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, X):

        U = self.encoder_1(X)
        V = self.encoder_2(X)

        Z = self.input_layer(X)
        H = (1 - Z) * U + Z * V

        for layer in self.hidden_layers:
            Z = layer(H)
            H = (1 -Z) * U + Z * V

        return self.output_layer(H)

