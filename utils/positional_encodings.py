##copy from https://github.com/tatp22/multidim-positional-encoding

import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        return tensor + penc
