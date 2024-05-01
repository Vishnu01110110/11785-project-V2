import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        batch, length, dimensions = x.shape
        reshaped_dim = (batch, length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(reshaped_dim).transpose(1, 2)
        k = k.view(reshaped_dim).transpose(1, 2)
        v = v.view(reshaped_dim).transpose(1, 2)

        attention_scores = q @ k.transpose(-2, -1)

        if causal_mask:
            sequence_mask = torch.triu(torch.ones_like(attention_scores, dtype=torch.bool), diagonal=1)
            attention_scores.masked_fill_(sequence_mask, float('-inf'))

        attention_scores /= math.sqrt(self.d_head)
        attention_probabilities = F.softmax(attention_scores, dim=-1)

        attended_output = attention_probabilities @ v
        attended_output = attended_output.transpose(1, 2).contiguous()
        attended_output = attended_output.reshape(batch, length, dimensions)
        final_output = self.out_proj(attended_output)

        return final_output

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
        batch, length, dim = x.shape
        reshaped_dim = (batch, -1, self.n_heads, self.d_head)

        q = self.q_proj(x).view(reshaped_dim).transpose(1, 2)
        k = self.k_proj(y).view(reshaped_dim).transpose(1, 2)
        v = self.v_proj(y).view(reshaped_dim).transpose(1, 2)

        cross_attention_scores = q @ k.transpose(-2, -1)
        cross_attention_scores /= math.sqrt(self.d_head)
        cross_attention_probabilities = F.softmax(cross_attention_scores, dim=-1)

        cross_attended_output = cross_attention_probabilities @ v
        cross_attended_output = cross_attended_output.transpose(1, 2).contiguous()
        cross_attended_output = cross_attended_output.view(batch, length, dim)
        final_cross_output = self.out_proj(cross_attended_output)

        return final_cross_output
