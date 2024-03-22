import torch
from torch import nn
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_length: int):
        """
        Embedding layer for CLIP model.

        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimensionality of the token embeddings.
        :param max_seq_length: Maximum sequence length for position embeddings.
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # A learnable weight matrix for encoding position information for each token.
        self.position_embedding = nn.Parameter(torch.zeros((max_seq_length, embedding_dim)))

    def forward(self, tokens):
        """
        Forward pass for generating token embeddings with added position information.

        :param tokens: Input token IDs.
        :return: Embedded tokens with positional embeddings.
        """
        token_embeddings = self.token_embedding(tokens)
        position_embeddings = self.position_embedding[:tokens.size(1), :]
        return token_embeddings + position_embeddings


class CLIPLayer(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int):
        """
        A single layer of the CLIP model, consisting of self-attention and a feedforward network.

        :param num_heads: Number of attention heads.
        :param embedding_dim: Dimensionality of the embeddings.
        """
        super().__init__()

        self.layernorm_pre = nn.LayerNorm(embedding_dim)
        self.self_attention = SelfAttention(num_heads, embedding_dim)
        self.layernorm_post = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            # QuickGELU: an approximate but faster version of the GELU activation function.
            lambda x: x * torch.sigmoid(1.702 * x),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        """
        Forward pass for the CLIP layer.

        :param x: Input embeddings.
        :return: Output embeddings after self-attention and feedforward network.
        """
        # Self-attention block
        x = x + self.self_attention(self.layernorm_pre(x))

        # Feedforward block
        x = x + self.feedforward(self.layernorm_post(x))
        return x


class CLIPModel(nn.Module):
    def __init__(self, vocab_size: int = 49408, embedding_dim: int = 768, max_seq_length: int = 77,
                 num_layers: int = 12, num_heads: int = 12):
        """
        Constructor for the CLIP model.

        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimensionality of the embeddings.
        :param max_seq_length: Maximum sequence length.
        :param num_layers: Number of layers in the encoder.
        :param num_heads: Number of attention heads in each layer.
        """
        super().__init__()
        self.embedding = CLIPEmbedding(vocab_size, embedding_dim, max_seq_length)
        self.layers = nn.ModuleList([CLIPLayer(num_heads, embedding_dim) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass for the CLIP model.

        :param tokens: Input token IDs.
        :return: Output embeddings from the CLIP model.
        """
        tokens = tokens.type(torch.long)
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layernorm(x)
        return x
