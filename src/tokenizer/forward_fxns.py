from torch import nn, Tensor
from typing import Tuple, Optional
import math

### Source foward processing functions ###


def src_fwd_fxn_basic(src: Tensor,
                      d_model: int,
                      src_embed: nn.Module,
                      src_pad_token: int,
                      pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for source tensor in Transformer for substructure to structure problem
    Args:
        src: The unembedded source tensor, raw input into the forward() method, 
            shape (batch_size, seq_len)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index
        pos_encoder: The positional encoder layer
    """
    if not isinstance(src_embed, nn.Embedding):
        src_key_pad_mask = None
    elif src_pad_token is not None:
        src_key_pad_mask = (src == src_pad_token).bool().to(src.device)
    src = src_embed(src) * math.sqrt(d_model)
    src = pos_encoder(src, None)
    return src, src_key_pad_mask
