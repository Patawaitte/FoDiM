from typing import Optional, Any

import torch
from torch.nn.modules.transformer import TransformerEncoder, Dropout
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from utils.positional_encodings import PositionalEncoding1D, Summer
from utils.relativeposition import MultiHeadAttentionLayer

"""

Tranformer model from :
https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TransformerModel.py

Transformed fo multilabel/multi target segmentation.
With the addition of relative position encoding

"""



__all__ = ['TransformerModel']

class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_type, n_sev, n_date,  window_size, d_model=64, n_head=1, n_layers=3,
                 d_inner=64, activation="relu", dropout=0.1):

        super(TransformerModel, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_n_type={n_type}_n_sev={n_sev}_n_date={n_date}_window_size={window_size}" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout} "


        encoder_layer = TransformerEncoderLayer(d_model, n_head, window_size, d_inner, dropout, activation)
        self.pos_encoder = Summer(PositionalEncoding1D(d_model))
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.flatten = Flatten()


        self.type_ = nn.Sequential(Linear(d_model, n_type))

        self.date_ = nn.Sequential(Linear(d_model, n_date))


        """
        self.sequential = Sequential(
            ,
            ,
            ,
            ,
            ReLU(),

        )
        """

    def forward(self,x):
        x = self.inlinear(x)
        x = self.relu(x)
        #x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        #x = x.transpose(0, 1) # T x N x D -> N x T x D

        #x = self.pos_encoder(x)
        x = x.max(1)[0]

        x = self.relu(x)
        #logits = self.outlinear(x)

	xtype = self.type_(x)
        xdate= self.date_(x)
        
        logprobabilitiestype = F.log_softmax(xtype, dim=-1)
        logprobabilitiesdate = F.log_softmax(xdate, dim=-1)

        return {
            'type': logprobabilitiestype,
            'date': logprobabilitiesdate
        }



class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)



class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, window_size, dim_feedforward=2048, dropout=0.1,  activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.window_size = window_size
        self.self_attn = MultiHeadAttentionLayer(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        nhead))  # 2*Wh-1 * 2*Ww-1, nH
        Wh, Ww = self.window_size

        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)

        rel_position_index = rel_index_coords + rel_index_coords.T

        rel_position_index = rel_position_index.flip(1).contiguous()

        self.register_buffer('relative_position_index', rel_position_index)


        #self.activation = _get_activation_fn(activation)
        self.softmax = nn.Softmax(dim=-1)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        #print("self.relative_position_bias_table", self.relative_position_bias_table.size())
        #print("self.relative_position_index", self.relative_position_index.size())
        #print("self.relative_position_index.view(-1)", self.relative_position_index.view(-1).size())
        #print("self.relative_position_index.view(-1).view", self.relative_position_index.view(-1).view(
             #   self.window_size[0] * self.window_size[1],
             #   self.window_size[0] * self.window_size[1],
             #   -1).size())



        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #print('relative_position_bias2', relative_position_bias.size())

        src2 = self.self_attn(src, src, src)[0]
        #print('src2', src2.size())

        #src2 = src2 + relative_position_bias.unsqueeze(0)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)




        return src

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
