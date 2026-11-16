"""Character-level GPT: a decoder-only Transformer written from scratch in PyTorch."""

import math

import torch
import torch.nn as nn


class NewGELU(nn.Module):
    """GELU activation (tanh approximation), as used in BERT and GPT-2.

    Reference: Gaussian Error Linear Units, https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention followed by an output projection."""

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.d_head = n_embd // n_head

        # A single linear layer produces queries, keys and values for every head at once.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Additive causal mask: -inf above the diagonal so a token never attends to the future.
        self.register_buffer("mask", torch.full((block_size, block_size), float("-inf")).triu(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, n_head, T, d_head)

        # Scaled dot-product attention with the causal mask.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_head, T, T)
        scores = scores + self.mask[:T, :T]
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = weights @ v  # (B, n_head, T, d_head)

        # Concatenate the heads back together and project.
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class Block(nn.Module):
    """Pre-norm Transformer block: attention and MLP, each with a residual connection."""

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            NewGELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT language model over a character vocabulary."""

    def __init__(self, n_layer: int, n_head: int, n_embd: int, vocab_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_head, n_embd, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        pos = torch.arange(t, device=idx.device).unsqueeze(0)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` characters after the prompt `idx` (shape (1, T))."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            logits = self(context)[:, -1, :] / temperature
            if top_k is not None:
                kth_value = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_value, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
