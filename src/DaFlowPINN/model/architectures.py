import torch
from torch import nn

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
    "Defines a standard fully-connected network in PyTorch with hard boundary constraints using sdf"

    def __init__(self, bc_fun, values=[0,0,0,0]):
        super().__init__()
        self.bc_fun = bc_fun
        self.offset = nn.Parameter(torch.tensor(values), requires_grad=False)

    def forward(self, x, x0):
        self.offset= nn.Parameter(self.offset.to(x.device), requires_grad=False)
        sdf_values=torch.maximum(self.bc_fun(x0),torch.tensor(0, device=x0.device))
        #print(max(sdf_values), min(sdf_values))
        l=torch.stack((sdf_values, sdf_values, sdf_values,(1+max(sdf_values)-sdf_values)),dim=1) #stack to multiply u,v,w --> p*1    torch.ones_like(sdf_values)
        x = x*l+(1-l)*self.offset #
        return x



class RFF(nn.Module):
    """Defines a fully-connected network with random Fourier features in PyTorch"""
    def __init__(self, N_INPUT, n_freqs = 16, scales = [1, 1, 1, 1]):
        super().__init__()

        self.mapping_size = 2 * n_freqs

        # B_x = torch.randn(n_freqs, 1) * scales[0]
        # B_y = torch.randn(n_freqs, 1) * scales[1]
        # B_z = torch.randn(n_freqs, 1) * scales[2]
        # B_t = torch.randn(n_freqs, 1) * scales[3]
        # self.B = nn.Parameter(torch.cat([B_x, B_y, B_z, B_t], dim=1).T, requires_grad=False)

        self.B_x = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)
        self.B_y = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)
        self.B_z = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)
        self.B_t = nn.Parameter(torch.randn(n_freqs, 1), requires_grad=False)

        self.scales = nn.Parameter(torch.tensor(scales), requires_grad=False)


    def rff(self, x):
        B = torch.cat([self.B_x * self.scales[0], 
                        self.B_y * self.scales[1],
                        self.B_z * self.scales[2], 
                        self.B_t * self.scales[3]], dim=1).T

        #B = self.B.to(x.device)

        x_proj = 2 * torch.pi * x @ B  # Linear transformation for all dimensions
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


        return x_proj
    
    def forward(self, x):
        x = self.rff(x)
        return x
    
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

