# In smartgd/model/common/positional_encoders.py (new file or existing common module)
import torch
import torch.nn as nn
import math


class GaussianRBFEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 16, basis_min: float = -1.5, basis_max: float = 1.5,
                 learnable_basis: bool = True):
        """
        Encodes input coordinates using Gaussian Radial Basis Functions.
        Args:
            in_dim (int): Input dimension of coordinates (e.g., 2 for (x,y)).
            out_dim (int): Output dimension of the encoded features.
            num_basis (int): Number of basis functions per input dimension.
            basis_min (float): Estimated minimum value for coordinate range for initializing centers.
            basis_max (float): Estimated maximum value for coordinate range for initializing centers.
            learnable_basis (bool): If True, centers and widths of RBFs are learnable parameters.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis

        centers = torch.linspace(basis_min, basis_max, num_basis).unsqueeze(0).repeat(in_dim,
                                                                                      1)  # Shape: [in_dim, num_basis]

        # Heuristic for widths: cover the range between centers, ensure it's not too small
        width_val = abs(basis_max - basis_min) / (num_basis - 1 if num_basis > 1 else 1.0)
        width_val = max(width_val, 0.1)  # Ensure a minimum width
        widths = torch.ones(in_dim, num_basis) * width_val

        if learnable_basis:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

        # Projection layer to map RBF features to the desired output dimension
        self.projection = nn.Linear(in_dim * num_basis, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Node coordinates, shape [num_nodes, in_dim]
        num_nodes = x.shape[0]
        x_unsqueezed = x.unsqueeze(-1)  # [num_nodes, in_dim, 1]

        # Ensure widths are positive during forward pass
        current_widths = torch.abs(self.widths.unsqueeze(0)) + 1e-6  # [1, in_dim, num_basis]
        current_centers = self.centers.unsqueeze(0)  # [1, in_dim, num_basis]

        # diff shape: [num_nodes, in_dim, num_basis]
        diff = x_unsqueezed - current_centers
        # rbf_features shape: [num_nodes, in_dim, num_basis]
        rbf_features = torch.exp(-(diff ** 2) / (2 * current_widths ** 2))

        # Flatten RBF features for each node
        # flattened_rbf shape: [num_nodes, in_dim * num_basis]
        flattened_rbf = rbf_features.view(num_nodes, -1)

        output = self.projection(flattened_rbf)
        return output

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, '
                f'num_basis={self.num_basis})')