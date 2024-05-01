import torch
from torch import nn
from torch.nn import functional as F
import math


#self attention used to model the relationship between the features of the same image
#used in encoder and decoder for better understanding of the features
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        # merge wq, wk, wv 
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        
        #output projection
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):

        
        # store shape
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # q,k,v 
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        
        #masking the upper triangle
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        
        # scale the weight
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 

        #get output
        output = weight @ v
        output = output.transpose(1, 2) 
        output = output.reshape(input_shape) 
        output = self.out_proj(output) 
        
        return output


#cross attention used to model the relationship between the features of the different images
#used in decoder and encoder to understand the relationship between the features of the image and the text
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        # merge wq, wk, wv  
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        
        # store shape
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # q,k,v reshaping
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        
        #calculate weights
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        #get output
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output