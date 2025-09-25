import torch

class Knet(torch.nn.Module):
    """Linear neural net to approximate the Koopman matrix.
    
    Contains identically sized input and output layers, no hidden layers, no bias vector, and no activation function.

    ## Parameters
    - **size** (*int*) - Dimension of the input and output layer.

    ## Attributes
    - **net** (*torch.nn.ModuleList*) - The neural net.
    """
    def __init__(self, size):
        """ """
        super().__init__()
        self.net = torch.nn.Linear(
            in_features = size,
            out_features = size,
            bias = False
        )

    def forward(self, X) -> torch.Tensor:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*, size)*) - Input data to net.

        ## Returns 
        - **X** (*torch.Tensor, shape=(\\*, size)*) - Output data from net.
        """
        return self.net(X)
