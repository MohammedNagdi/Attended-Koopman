"""Neural nets used inside the models."""


import torch
from .efficient_kan import *
from .polyKAN import *

__pdoc__ = {
    'MLP': False
}


class MLP(torch.nn.Module):
    """Multi-layer perceptron neural net.

    ## Parameters
    - **input_size** (*int*) - Dimension of the input layer.

    - **output_size** (*int*) - Dimension of the output layer.

    - **hidden_sizes** (*list[int], optional*) - Dimensions of hidden layers, if any.

    - **batch_norm** (*bool, optional*) - Whether to use batch normalization.

    ## Attributes
    - **net** (*torch.nn.ModuleList*) - List of layers
    """
    def __init__(self, input_size, output_size, hidden_sizes=[], batch_norm=False):
        """ """
        super().__init__()
        self.net = torch.nn.ModuleList([])
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers)-1):
            self.net.append(torch.nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2: #all layers except last
                if batch_norm:
                    self.net.append(torch.nn.BatchNorm1d(layers[i+1]))
                self.net.append(torch.nn.LeakyReLU())

    def forward(self, X) -> torch.Tensor:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to net.

        ## Returns 
        - **X** (*torch.Tensor, shape=(\\*,output_size)*) - Output data from net.
        """
        for layer in self.net:
            X = layer(X)
        return X


class AutoEncoder(torch.nn.Module):
    """AutoEncoder neural net. Contains an encoder connected to a decoder, with configurable network types (MLP, KAN, PolyKan).

    ## Parameters
    - **input_size** (*int*) - Number of dimensions in original data (encoder input) and reconstructed data (decoder output).

    - **encoded_size** (*int*) - Number of dimensions in encoded data (encoder output and decoder input).

    - **encoder_hidden_layers** (*list[int], optional*) - Encoder will have layers = `[input_size, *encoder_hidden_layers, encoded_size]`. If not set, defaults to reverse of `decoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **decoder_hidden_layers** (*list[int], optional*) - Decoder has layers = `[encoded_size, *decoder_hidden_layers, input_size]`. If not set, defaults to reverse of `encoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **network_type** (*str, optional*) - Type of network to use: 'MLP', 'KAN', or 'PolyKan'. Defaults to 'MLP'.

    - **batch_norm** (*bool, optional*) - Whether to use batch normalization (applicable for MLP only).

    ## Attributes
    - **encoder** - Encoder neural net.

    - **decoder** - Decoder neural net.
    """
    def __init__(self, input_size, encoded_size, encoder_hidden_layers=[], decoder_hidden_layers=[], network_type='MLP', batch_norm=False, **kwargs):
        """ """
        super().__init__()

        if not decoder_hidden_layers and encoder_hidden_layers:
            decoder_hidden_layers = encoder_hidden_layers[::-1]
        elif not encoder_hidden_layers and decoder_hidden_layers:
            encoder_hidden_layers = decoder_hidden_layers[::-1]

        # Select network type for encoder
        if network_type == 'MLP':
            
            self.encoder = nn.Sequential(
                MLP(
                input_size=input_size,
                output_size=encoded_size,
                hidden_sizes=encoder_hidden_layers,
                batch_norm=batch_norm
            ),
                torch.nn.Dropout(p=0.3)
            )            
            self.decoder = MLP(
                input_size=encoded_size,
                output_size=input_size,
                hidden_sizes=decoder_hidden_layers,
                batch_norm=batch_norm
            )
        elif network_type == 'KAN':
            spline_knots = kwargs.get('spline_knots', 5)

            encoder_layers = [input_size] + encoder_hidden_layers + [encoded_size] # input, hidden, output
            self.encoder = KAN(
                layers_hidden=encoder_layers,
                grid_size=spline_knots
            )

            # inverse the encoder layers for decoder ; output, hidden, input
            decoder_layers = encoder_layers[::-1]
            self.decoder = KAN(
                layers_hidden=decoder_layers,
                grid_size=spline_knots
            )

        elif network_type == 'PolyKan':
            poly_type = kwargs.get('poly_type', 'legendre')
            degree = kwargs.get('degree', 3)

            encoder_layers = [input_size] + encoder_hidden_layers + [encoded_size]
            self.encoder = polynet(
                layers_hidden=encoder_layers,
                poly_type=poly_type,
                degree=degree
            )

            decoder_layers = encoder_layers[::-1]
            self.decoder = polynet(
                layers_hidden=decoder_layers,
                poly_type=poly_type,
                degree=degree
            )
        else:
            raise ValueError("network_type must be one of 'MLP', 'KAN', or 'PolyKan'")

    def forward(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to encoder.

        ## Returns 
        - **Y** (*torch.Tensor, shape=(\\*,encoded_size)*) - Encoded data, i.e. output from encoder, input to decoder.

        - **Xr** (*torch.Tensor, shape=(\\*,input_size)*) - Reconstructed data, i.e. output of decoder.
        """
        Y = self.encoder(X)  # encoder complete output
        Xr = self.decoder(Y)  # final reconstructed output
        return Y, Xr
    
    def encode(self, X) -> torch.Tensor:
        """Encode data using encoder.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to encoder.

        ## Returns 
        - **Y** (*torch.Tensor, shape=(\\*,encoded_size)*) - Encoded data.
        """
        return self.encoder(X)
    
    def decode(self, Y) -> torch.Tensor:
        """Decode data using decoder.

        ## Parameters
        - **Y** (*torch.Tensor, shape=(\\*,encoded_size)*) - Encoded data.

        ## Returns 
        - **Xr** (*torch.Tensor, shape=(\\*,input_size)*) - Reconstructed data.
        """
        return self.decoder(Y)
    
    def reconstruct(self, X) -> torch.Tensor:
        """Reconstruct data using encoder and decoder.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to encoder.

        ## Returns 
        - **Xr** (*torch.Tensor, shape=(\\*,input_size)*) - Reconstructed data.
        """
        Y = self.encoder(X)
        return self.decoder(Y)

