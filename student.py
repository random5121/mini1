"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
from model import GPT
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict, Counter


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight


def apply_rope(x):
    # x: [batch, heads, time, head_dim]; rotate adjacent even/odd channels.
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos, sin = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width=128, heads=4, gate_enabled=False, gate_init=0.1,
                 local_attention=False, attention_window=128, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.gate_enabled = gate_enabled
        self.local_attention = local_attention
        self.attention_window = attention_window
        # TUNE: dropout probability; configure this in configs/baseline.json.
        self.dropout = nn.Dropout(float(dropout))
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.up = nn.Linear(width, 8 * width)
        self.down = nn.Linear(4 * width, width)
        if gate_enabled:
            # Separate learned per-channel scales for attention and FFN residuals.
            self.attn_gate = nn.Parameter(torch.full((width,), float(gate_init)))
            self.ffn_gate = nn.Parameter(torch.full((width,), float(gate_init)))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q), apply_rope(k)
        # Local layers see only the current token and the preceding window - 1 tokens.
        if self.local_attention:
            positions = torch.arange(length, device=x.device)
            allowed = (positions[:, None] - positions[None, :]) < self.attention_window
            allowed &= positions[None, :] <= positions[:, None]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_update = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        attn_update = self.dropout(attn_update)
        if self.gate_enabled:
            attn_update = attn_update * self.attn_gate
        x = x + attn_update
        gate, value = self.up(self.norm2(x)).chunk(2, dim=-1)
        ffn_update = self.down(F.silu(gate) * value)
        ffn_update = self.dropout(ffn_update)
        if self.gate_enabled:
            ffn_update = ffn_update * self.ffn_gate
        return x + ffn_update


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        gate_enabled = bool(config.get('residual_gates', False))
        gate_init = float(config.get('residual_gate_init', 0.1))
        attention_mode = config.get('attention_mode', 'global')
        if attention_mode not in ('global', 'hybrid'):
            raise ValueError("attention_mode must be 'global' or 'hybrid'")
        attention_window = int(config.get('attention_window', 128))
        dropout = float(config.get('dropout', 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in the range [0, 1)')
        global_every = int(config.get('global_attention_every', 4))
        self.ngram_enabled = bool(config.get('ngram_enabled', False))
        self.ngram_mix = float(config.get('ngram_mix', 0.1))
        self.ngram_slots = int(config.get('ngram_slots', 32768))
        self.temperature_scaling = bool(config.get('temperature_scaling', False))
        self.temperature = float(config.get('temperature', 1.0))
        if self.temperature <= 0.0:
            raise ValueError('temperature must be positive')
        if not 0.0 <= self.ngram_mix <= 1.0:
            raise ValueError('ngram_mix must be between 0 and 1')
        if attention_window < 1 or global_every < 1:
            raise ValueError('attention_window and global_attention_every must be positive')
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([
            Block(
                width, config['heads'], gate_enabled, gate_init,
                local_attention=(attention_mode == 'hybrid' and (i + 1) % global_every != 0),
                attention_window=attention_window,
                dropout=dropout,
            )
            for i in range(config['depth'])
        ])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        # Compact training-derived bigram-context table; saved in the checkpoint.
        self.register_buffer('ngram_keys', torch.full((self.ngram_slots,), -1, dtype=torch.long))
        self.register_buffer('ngram_next', torch.zeros(self.ngram_slots, dtype=torch.long))
        self.register_buffer('ngram_prob', torch.zeros(self.ngram_slots, dtype=torch.float32))
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        Override this for a strictly causal, within-window memory mechanism.
        A prediction at position t can use ids[:, :t+1] and nothing later.
        Reset all temporary state on every call; each evaluation window starts fresh.
        """
        logits = self(ids).float()
        if self.temperature_scaling:
            logits = logits / self.temperature
        logp = F.log_softmax(logits, dim=-1)
        if not self.ngram_enabled or self.ngram_mix == 0:
            return logp

        # Match the last two observed tokens. Position zero has no two-token context.
        keys = ids[:, :-1] * 2048 + ids[:, 1:]
        slots = torch.remainder(keys, self.ngram_slots)
        matched = self.ngram_keys[slots] == keys
        prob = self.ngram_prob[slots].to(logp.dtype) * matched
        nxt = self.ngram_next[slots]
        background = (1.0 - prob.unsqueeze(-1)) / (logp.shape[-1] - 1)
        ng = torch.ones_like(logp[:, 1:]) * background
        ng.scatter_(-1, nxt.unsqueeze(-1), prob.unsqueeze(-1))
        mix = self.ngram_mix
        mixed = torch.logaddexp(logp[:, 1:] + torch.log1p(logp.new_tensor(-mix)),
                                ng.clamp_min(1e-30).log() + logp.new_tensor(mix).log())
        return torch.cat((logp[:, :1], mixed), dim=1)

    @torch.no_grad()
    def fit_ngram(self, tokens):
        """Build a bounded bigram-context -> next-token table from training tokens only."""
        if not self.ngram_enabled:
            return
        counts = defaultdict(Counter)
        seq = tokens.detach().cpu().tolist()
        for a, b, nxt in zip(seq, seq[1:], seq[2:]):
            counts[a * 2048 + b][nxt] += 1
        # Keep frequent contexts and discard hash collisions; exact key checks at
        # inference ensure no wrong context can contribute probability.
        rows = sorted(counts.items(), key=lambda item: sum(item[1].values()), reverse=True)
        keys = torch.full((self.ngram_slots,), -1, dtype=torch.long)
        nxts = torch.zeros(self.ngram_slots, dtype=torch.long)
        probs = torch.zeros(self.ngram_slots, dtype=torch.float32)
        for key, successors in rows:
            slot = key % self.ngram_slots
            if keys[slot] != -1:
                continue
            token, count = successors.most_common(1)[0]
            keys[slot] = key
            nxts[slot] = token
            probs[slot] = count / sum(successors.values())
        self.ngram_keys.copy_(keys.to(self.ngram_keys.device))
        self.ngram_next.copy_(nxts.to(self.ngram_next.device))
        self.ngram_prob.copy_(probs.to(self.ngram_prob.device))


def build_model(config):
    """Keep the classroom baseline runnable with --implementation model."""
    return GPT(config)
