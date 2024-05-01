import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class ClipTokenEmbedder(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        position_embed_init = torch.zeros((n_token, n_embd))
        self.position_embedding = nn.Parameter(position_embed_init)

    def forward(self, tokens):
        embedded_tokens = self.token_embedding(tokens)
        positionally_encoded = embedded_tokens + self.position_embedding

        return positionally_encoded


class ClipTransformerLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residual_connection = x
        x = self.layernorm_1(x)

        attended = self.attention(x, causal_mask=True)
        x = attended

        x += residual_connection

        residual_connection = x
        x = self.layernorm_2(x)

        x = self.linear_1(x)

        gelu_scale = 1.702 * x
        gelu_activation = torch.sigmoid(gelu_scale)
        x = x * gelu_activation  # QuickGELU activation function

        x = self.linear_2(x)

        x += residual_connection

        return x


class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ClipTokenEmbedder(49408, 768, 77)

        self.layers = nn.ModuleList([
            ClipTransformerLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        embedding_output = self.embedding(tokens)

        for clip_layer in self.layers:
            embedding_output = clip_layer(embedding_output)
        normed_output = self.layernorm(embedding_output)

        return normed_output
