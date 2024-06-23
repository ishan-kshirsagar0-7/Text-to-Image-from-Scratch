# THIS IS THE LOGIC FOR IMPLEMENTING ATTENTION IN OUR DIFFUSION MODEL.
# There are two types of Attention used in our project : Self-Attention & Cross-Attention.
# Self-Attention mainly is used to calculate how important or relevant one token is with respect to other tokens, within a same block.
# Cross-Attention mainly is used to calculate how important or relevant one token of one block is with respect to tokens of some other block.
# In our case, since we're dealing with text-to-image, we apply Self-Attention individually within tokens (of the prompt) and within the pixels (of the image or the noise) separately. However, we apply Cross-Attention between token-pixel pairs.

import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x : (batch_size, seq_len, dim)
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # Now we split the input matrices into Query, Key and Value pairs
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, H, dim/H) -> (batch_size, H, seq_len, dim/H)
        # where, H is the number of heads or n_heads as defined in the interim_shape
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # We then calculate the attention values over all pixels with the formula [w = q @ Transpose of (k)]
        # (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle of the matrix is populated with Ones
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (dim/H). 
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1) 

        # (batch_size, H, seq_len, seq_len) @ (batch_size, H, seq_len, dim/H) -> (batch_size, H, seq_len, dim/H)
        output = weight @ v

        # (batch_size, H, seq_len, dim/H) -> (batch_size, seq_len, H, dim/H)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape) 
        output = self.out_proj(output) 
        
        # (batch_size, seq_len, dim)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent) : (batch_size, seq_len_q, dim_q)
        # y (context) : (batch_size, seq_len_kv, dim_kv) = (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # First we multiply q with Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output