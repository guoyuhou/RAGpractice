import torch
from torch import nn
import copy

def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module)
                          for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        features: 特征维度的大小,对应layer.size
        eps: 一个很小的数,加在分母上防止除以0
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        向前传播
        x: 输入张量,通常是(batch_size, sequence_length, features)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return (self.a_2 * (x - mean) / (std + self.eps) + self.b_2)

class Encoder(nn.Module):
    "Core encoder is a stack of layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input/mask through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A layer norm followed by a residual connection,
    Note norm is not applied to residual x
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to sublayer fn"
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn,
                 feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        sublayer = SublayerConnection(size, dropout)
        self.sublayer = clones(sublayer, 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder calls self-attn, src-attn, and feed forward"
    def __init__(self, size, self_attn,
                 src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        sublayer = SublayerConnection(size, dropout)
        self.sublayer = clones(sublayer, 3)
        self.size = size

    def forward(self, x, memory, s_mask, t_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, t_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, s_mask))
        return self.sublayer[2](x, self.feed_forward)