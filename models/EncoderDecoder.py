import copy
import math
import torch
import einops
import random
import numpy as np
from torch import nn
from typing import Optional
import torch.nn.functional as F
from .pytorch_polynomial_features import PolynomialFeatures

class EncoderDecoder(nn.Module):
    """
    Main function to generate mixes of models for EncoderDecoder. Specify an encoder and a decoder.
    
    ## Attributes
    - **encoder** (*nn.Module*) - The encoder model
    - **decoder** (*nn.Module*) - The decoder model
    """
    def __init__(self,
            name,
            input_size,
            input_length,
            hidden_size,
            output_size,
            encoder,
            decoder,
            encoder_depth,
            decoder_depth,
            dropout,
            n_heads,
            dim_feedforward,
            device
        ):
        super().__init__()
        
        # Class variables
        self.name = name
        self.input_size = input_size
        self.input_length = input_length

        # Construct EncoderDecoder
        if encoder == "gru":
            self.encoder = GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=encoder_depth,
                dropout=dropout,
                device=device
            )
        elif encoder == "transformer":
            self.encoder = Transformer(
                d_model=input_size,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=nn.GELU(),
                hidden_size=hidden_size,
                input_length=input_length,
                num_layers=encoder_depth,
                layer_norm_eps=1e-5,
                bias=True,
                device=device
            )
        else:
            raise NotImplementedError(f"Encoder {encoder} not implemented")
        
        if decoder == "mlp":
            self.decoder = MLP(
                in_dim = hidden_size,
                out_dim = output_size,
                n_layers = decoder_depth,
                dropout=dropout,
                device=device
            )
        else:
            raise NotImplementedError(f"Decoder {decoder} not implemented")

        # Model encoder and decoder
        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    def forward(self, src: torch.Tensor):
        """
        Forward pass.

        ## Parameters
        - **src** (*torch.Tensor, shape=(batch_size,sequence_length,d_model)*) - Input tensor

        ## Returns
        - **src_decoded** (*dict*) - Dictionary containing the output of the decoder
            - **output** (*torch.Tensor, shape=(batch_size,\\*,output_size)*) - Output tensor
        """

        src_encoded = self.encoder(src)
        src_decoded = self.decoder(src_encoded)

        return src_decoded

class MLP(nn.Module):
    """
    Creates a simple MLP.

    ## Parameters
    - **in_dim** (*int*) - Input dimension `[*, in_dim]`
    - **out_dim** (*int*) - Output dimension `[*, out_dim]`
    - **n_layers** (*int*) - Number of layers in the model (number of matrix multiplies)
    - **dropout** (*float*) - Dropout rate
    - **device** (*str*) - Which device to put the model on

    ## Attributes
    - **model** (*nn.Sequential*) - The MLP model
    - **dropout** (*nn.Dropout*) - Dropout layer

    ## Notes
    - Size of intermediate layers is logarithmic (base 2) from `in_dim` to `out_dim`
    """
    def __init__(self,
            in_dim:int,
            out_dim:int,
            n_layers:int,
            dropout:float,
            device:str
        ):
        super(MLP, self).__init__()

        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(np.logspace(np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers+1, dtype=int).tolist())
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx+1]))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        # Define model
        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.dropout = nn.Dropout(dropout)
        self.model = model

    def forward(self, x):
        """
        Forward pass through the MLP.

        ## Parameters
        - **x** (*dict*) - Dictionary containing the output of the encoder
            - **output** (*torch.Tensor, shape=(batch_size,\\*,in_dim)*) - Final hidden state of the encoder

        ## Returns 
        - **d** (*dict*) - Dictionary containing the output
            - **output** (*torch.Tensor, shape=(batch_size,\\*,out_dim)*) - Output of the MLP
        """

        # Pass final state through MLP
        x = x["output"]
        out = self.model(x)
        out = self.dropout(out)

        # output dictionary
        d = {
            "output": out,
        }

        return d

class GRU(nn.Module):
    """
    GRU encoder.

    ## Parameters
    - **input_size** (*int*) - Input dimension `[*, input_size]`
    - **hidden_size** (*int*) - Hidden dimension `[*, hidden_size]`
    - **num_layers** (*int*) - Number of layers in the GRU
    - **dropout** (*float*) - Dropout rate
    - **device** (*str*) - Which device to put the model on

    ## Attributes
    - **gru** (*nn.GRU*) - The GRU model
    """
    def __init__(self, input_size:int,
            hidden_size:int,
            num_layers:int,
            dropout:float,
            device:str
        ):
        super().__init__()

        # Class variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Define GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass through the GRU model.

        ## Parameters
        - **x** (*torch.Tensor, shape=(batch_size,sequence_length,input_size)*) - Input data to encoder.

        ## Returns 
        - **d** (*dict*) - Dictionary containing the output
            - **sequence_output** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Sequence output of the GRU
            - **final_hidden_state** (*torch.Tensor, shape=(batch_size,1,hidden_size)*) - Final hidden state of the GRU
            - **output** (*torch.Tensor, shape=(batch_size,1,hidden_size)*) - Output of the GRU
        """
        # Initialize hidden state and output state
        out, _ = self.gru(x)

        # Apply dropout
        out = self.dropout(out)

        # Extract last hidden state
        final_out = out[:,-1:,:]

        # output dictionary
        d = {
            "sequence_output": out,
            "final_hidden_state": final_out,
            "output": final_out,
        }

        return d

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

class TransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer.

    ## Parameters
    - **d_model** (*int*) - The dimension of the model
    - **nhead** (*int*) - The number of heads
    - **dim_feedforward** (*int*) - The dimension of the feedforward network
    - **dropout** (*float*) - The dropout rate
    - **activation** (*nn.Module*) - The activation function
    - **layer_norm_eps** (*float*) - The epsilon for the layer normalization
    - **norm_first** (*bool*) - Whether to apply layer normalization before the attention
    - **bias** (*bool*) - Whether to use bias in the linear layers
    - **device** (*str*) - The device to put the model on
    - **dtype** (*torch.dtype*) - The data type to put the model on

    ## Notes
    - Copied from pytorch: https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation:nn.Module,
        layer_norm_eps,
        norm_first,
        bias,
        device,
        dtype,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Define self-attention layer
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )

        # Define feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        # Define layer normalizations
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        # Define dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        
    def _sa_block(self, x, attn_mask, is_causal):
        """
        Self-attention block.

        ## Parameters
        - **x** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Input tensor
        - **attn_mask** (*torch.Tensor, shape=(batch_size,sequence_length,sequence_length)*) - Attention mask
        - **is_causal** (*bool*) - Whether to apply causal mask

        ## Returns
        - **x** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Output tensor
        """
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        """
        Feedforward block.

        ## Parameters
        - **x** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Input tensor

        ## Returns
        - **x** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Output tensor
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=True):
        """
        Forward pass.

        ## Parameters
        - **src** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Input tensor
        - **src_mask** (*torch.Tensor, shape=(batch_size,sequence_length,sequence_length)*) - Attention mask
        - **is_causal** (*bool*) - Whether to apply causal mask

        ## Returns
        - **x** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Output tensor
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

class TransformerEncoder(nn.Module):
    """
    TransformerEncoder.

    ## Parameters
    - **encoder_layer** (*TransformerEncoderLayer*) - The encoder layer
    - **num_layers** (*int*) - The number of layers
    - **norm** (*nn.Module*) - The layer normalization
    - **device** (*str*) - The device to put the model on
    - **dtype** (*torch.dtype*) - The data type to put the model on

    ## Returns
    - **output** (*torch.Tensor*) - The output of the encoder

    ## Notes
    - Copied from pytorch: https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Class variables
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
            src: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            is_causal: bool = True 
        ):
        """
        Forward pass through the TransformerEncoder.

        ## Parameters
        - **src** (*torch.Tensor*, shape=(batch_size,sequence_length,hidden_size)*) - The input tensor
        - **mask** (*torch.Tensor*, shape=(batch_size,sequence_length,sequence_length)*) - The mask tensor (optional)
        - **is_causal** (*bool*) - Whether to apply causal mask

        ## Returns
        - **output** (*torch.Tensor*) - The output of the encoder
        """
        output = src
        for mod in self.layers:
            output = mod(output, mask, is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class Transformer(nn.Module):
    """
    Transformer.

    ## Parameters
    - **d_model** (*int*) - The dimension of the model
    - **nhead** (*int*) - The number of heads
    - **num_layers** (*int*) - The number of encoder layers
    - **dim_feedforward** (*int*) - The dimension of the feedforward network
    - **dropout** (*float*) - The dropout rate
    - **activation** (*nn.Module*) - The activation function
    - **layer_norm_eps** (*float*) - The epsilon for the layer normalization
    - **norm_first** (*bool*) - Whether to apply layer normalization before the attention
    - **bias** (*bool*) - Whether to use bias in the linear layers
    - **input_length** (*int*) - The length of the input sequence
    - **hidden_size** (*int*) - The dimension of the hidden state
    - **device** (*str*) - The device to put the model on
    """
    def __init__(
        self,
        d_model,
        nhead,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        bias=True,
        input_length=10,
        hidden_size=10,
        device='cpu',
    ):
        super().__init__()

        # Define input embedding
        self.input_embedding = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Define positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_size,
            sequence_length=input_length + 10, # Provide some buffer
            dropout=dropout,
            device=device
        )

        # Define encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=torch.float
        )

        # Define layer normalization
        encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=bias, device=device)

        # Define transformer encoder
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )

    def forward(
        self,
        src,
        src_mask=None,
        is_causal=True,
    ):
        """
        Forward pass through the Transformer.

        ## Parameters
        - **src** (*torch.Tensor, shape=(batch_size,sequence_length,input_size)*) - Input data to encoder.

        ## Returns
        - **d** (*dict*) - Dictionary containing the output
            - **sequence_output** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Sequence output of the Transformer
            - **final_hidden_state** (*torch.Tensor, shape=(batch_size,1,hidden_size)*) - Final hidden state of the Transformer
            - **output** (*torch.Tensor, shape=(batch_size,sequence_length,hidden_size)*) - Output of the Transformer
        """
        # Embed input data using GRU
        x_embedded, _ = self.input_embedding(src)

        # Positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded)

        # Transformer
        transformer_output = self.encoder(
            x_pos_encoded,
            mask=src_mask,
            is_causal=is_causal,
        )

        return {
            "sequence_output": transformer_output,
            "final_hidden_state": transformer_output[:, -1, :],
            "output": transformer_output,
        }

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the input data to the transformer.

    ## Parameters
    - **d_model** (*int*) - The dimension of the model
    - **sequence_length** (*int*) - The length of the sequence
    - **dropout** (*float*) - The dropout rate
    - **device** (*str*) - The device to put the module

    ## Notes
    - source: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """
    def __init__(self,
            d_model: int,
            sequence_length: int,
            dropout: float,
            device: str
        ):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(1, sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x

def _get_clones(module, N):
    """
    Get a list of clones of the module.

    ## Parameters
    - **module** (*nn.Module*) - The module to clone
    - **N** (*int*) - Number of clones

    ## Returns
    - **clones** (*nn.ModuleList*) - List of clones

    ## Notes
    - We use this for exact parity with the PyTorch transformer implementation, having the same init for every layer might not be necessary.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
