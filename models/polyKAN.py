import torch
import torch.nn as nn
import numpy as np

class polyKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer implementing various polynomial basis expansions.
    
    This layer transforms input features using non-linear polynomial basis functions 
    and learnable coefficients, allowing for adaptive function approximation.
    
    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        degree (int): Polynomial expansion degree
        poly_type (str): Type of polynomial basis used for expansion ('chebyshev', 'bessel', 'fibonacci', 'gegenbauer', 'hermite', 'jacobi', 'laguerre', 'legendre')
        alpha (float, optional): Parameter for certain polynomial types (Gegenbauer, Jacobi, Laguerre)
        beta (float, optional): Parameter for Jacobi polynomials
        coeffs (nn.Parameter): Learnable coefficients for polynomial basis transformation
    """
    def __init__(self, input_dim, output_dim, degree, poly_type, alpha=None, beta=None):
        super(polyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.poly_type = poly_type
        self.alpha = alpha
        self.beta = beta
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

        if poly_type == 'chebyshev':
            self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.view(-1, self.input_dim)
        x = torch.tanh(x)

        if self.poly_type == 'chebyshev':
            x = x.view((-1, self.input_dim, 1)).expand(-1, -1, self.degree + 1)
            x = x.acos()
            x *= self.arange
            x = x.cos()
        else:
            poly = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
            if self.poly_type == 'bessel':
                if self.degree > 0:
                    poly[:, :, 1] = x + 1
                for i in range(2, self.degree + 1):
                    poly[:, :, i] = (2 * i - 1) * x * poly[:, :, i - 1].clone() + poly[:, :, i - 2].clone()
            elif self.poly_type == 'fibonacci':
                poly[:, :, 0] = 0
                if self.degree > 0:
                    poly[:, :, 1] = 1
                for i in range(2, self.degree + 1):
                    poly[:, :, i] = x * poly[:, :, i - 1].clone() + poly[:, :, i - 2].clone()
            elif self.poly_type == 'gegenbauer':
                if self.degree > 0:
                    poly[:, :, 1] = 2 * self.alpha * x
                for n in range(1, self.degree):
                    term1 = 2 * (n + self.alpha) * x * poly[:, :, n].clone()
                    term2 = (n + 2 * self.alpha - 1) * poly[:, :, n - 1].clone()
                    poly[:, :, n + 1] = (term1 - term2) / (n + 1)
            elif self.poly_type == 'hermite':
                if self.degree > 0:
                    poly[:, :, 1] = 2 * x
                for i in range(2, self.degree + 1):
                    poly[:, :, i] = 2 * x * poly[:, :, i - 1].clone() - 2 * (i - 1) * poly[:, :, i - 2].clone()
            elif self.poly_type == 'jacobi':
                if self.degree > 0:
                    poly[:, :, 1] = (0.5 * (self.alpha - self.beta) + (self.alpha + self.beta + 2) * x / 2)
                for n in range(2, self.degree + 1):
                    A_n = 2 * n * (n + self.alpha + self.beta) * (2 * n + self.alpha + self.beta - 2)
                    term1 = (2 * n + self.alpha + self.beta - 1) * (2 * n + self.alpha + self.beta) * \
                            (2 * n + self.alpha + self.beta - 2) * x * poly[:, :, n-1].clone()
                    term2 = (2 * n + self.alpha + self.beta - 1) * (self.alpha ** 2 - self.beta ** 2) * poly[:, :, n-1].clone()
                    term3 = (n + self.alpha + self.beta - 1) * (n + self.alpha - 1) * (n + self.beta - 1) * \
                            (2 * n + self.alpha + self.beta) * poly[:, :, n-2].clone()
                    poly[:, :, n] = (term1 - term2 - term3) / A_n
            elif self.poly_type == 'laguerre':
                poly[:, :, 0] = 1
                if self.degree > 0:
                    poly[:, :, 1] = 1 + self.alpha - x
                for k in range(2, self.degree + 1):
                    term1 = ((2 * (k-1) + 1 + self.alpha - x) * poly[:, :, k - 1].clone())
                    term2 = (k - 1 + self.alpha) * poly[:, :, k - 2].clone()
                    poly[:, :, k] = (term1 - term2) / k
            elif self.poly_type == 'legendre':
                poly[:, :, 0] = 1
                if self.degree > 0:
                    poly[:, :, 1] = x
                for n in range(2, self.degree + 1):
                    poly[:, :, n] = ((2 * (n-1) + 1) / n) * x * poly[:, :, n-1].clone() - ((n-1) / n) * poly[:, :, n-2].clone()
            x = poly
        
        y = torch.einsum('bid,iod->bo', x, self.coeffs)
        return y.view(*original_shape, self.output_dim)
            
def combine_layers(layers, method='sum', weights=None, dim=0):
    """
    Unified function for combining neural network layer outputs with tensor support.
    
    Args:
        layers (list): List of layer outputs (torch.Tensor)
        method (str): Combination method 
            - 'sum': Element-wise summation
            - 'average': Element-wise averaging
            - 'quadratic': Quadratic combination (requires 2 layers)
            - 'weighted': Weighted combination
        weights (list, optional): Weights for weighted combination
        dim (int, optional): Dimension for combination operations
    
    Returns:
        torch.Tensor: Combined layer outputs
    """
    # Input validation
    if not layers:
        raise ValueError("At least one layer must be provided")
    
    # Ensure all inputs are tensors
    layers = [layer if isinstance(layer, torch.Tensor) else torch.tensor(layer) for layer in layers]

    # Ensure all layers have the same shape
    if not all(layer.shape == layers[0].shape for layer in layers):
        raise ValueError("All layers must have the same shape")
    
    # Combination methods with gradient propagation
    if method == 'sum':
        return torch.sum(torch.stack(layers), dim=dim)
    
    elif method == 'average':
        return torch.mean(torch.stack(layers), dim=dim)
    
    elif method == 'product':
        return torch.prod(torch.stack(layers), dim=dim)
    
    elif method == 'quadratic':
        # Sum + Product + Squared layers
        if len(layers) != 2:
            raise ValueError("Quadratic combination requires exactly 2 layers")
        return torch.sum(torch.stack(layers), dim=dim) + torch.prod(torch.stack(layers), dim=dim) + torch.sum(torch.stack([layer ** 2 for layer in layers]), dim=dim)
    
    elif method == 'weighted':
        if weights is None:
            weights = torch.ones(len(layers)) / len(layers)
        
        if len(weights) != len(layers):
            raise ValueError("Number of weights must match number of layers")
        
        # Weighted combination with gradient support
        stacked_layers = torch.stack(layers)
        weights_tensor = torch.tensor(weights, dtype=stacked_layers.dtype, device=stacked_layers.device)
        return torch.sum(stacked_layers * weights_tensor.view(-1, 1, 1), dim=0)
    
    else:
        raise ValueError("Invalid combination method")
    
class polynet(nn.Module):
    def __init__(self,
                 layers_hidden,
                 poly_type,
                 degree,
                 alpha=3,
                 beta=3):
        super(polynet, self).__init__()
        self.poly_type = poly_type
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                polyKANLayer(
                    in_features,
                    out_features,
                    degree=degree,
                    poly_type=poly_type,
                    alpha=alpha,
                    beta=beta
                )
            )
            self.layers.append(
                nn.LayerNorm(out_features)
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x