from __future__ import annotations

import torch
import torch.nn as nn

try:
    from time_series_transformer.learnable_time2vec import LearnableTime2VecSinCos
except ImportError:
    from learnable_time2vec import LearnableTime2VecSinCos


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


class HTTransformerRegressor(nn.Module):
    """HT-style transformer for one-step autoregressive forecasting."""

    def __init__(
        self,
        *,
        t2v_dim: int = 16,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 1024,
        observed_dim: int = 1,
        tfx_dim: int = 4,
        tfy_dim: int = 4,
        use_rollout_depth_for_horizon: bool = False,
    ):
        super().__init__()

        if observed_dim < 1:
            raise ValueError("observed_dim must be >= 1")

        self.observed_dim = observed_dim
        self.tfy_dim = tfy_dim
        self.use_rollout_depth_for_horizon = use_rollout_depth_for_horizon

        self.tfx_t2v = LearnableTime2VecSinCos(in_dim=tfx_dim, out_dim=t2v_dim)
        self.tfy_t2v = LearnableTime2VecSinCos(in_dim=tfy_dim, out_dim=t2v_dim)

        t2v_out = 1 + 2 * (t2v_dim - 1)
        self.input_proj = nn.Linear(observed_dim + t2v_out, model_dim)
        self.tfy_proj = nn.Linear(t2v_out, model_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.horizon_embed = nn.Embedding(512, model_dim)
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, src: torch.Tensor, tfy: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        src: (B, seq_len, observed_dim + tfx_dim)
        tfy: (B, tfy_dim)
        """
        batch_size, seq_len, _ = src.shape

        vals = src[..., : self.observed_dim]
        tfx_raw = src[..., self.observed_dim :]

        tfx_embed = self.tfx_t2v(tfx_raw)
        x = torch.cat([vals, tfx_embed], dim=-1)
        x = self.input_proj(x)

        mask = generate_causal_mask(seq_len, src.device)
        x = self.transformer_encoder(x, mask=mask)

        tfy_embed = self.tfy_t2v(tfy)
        q = self.tfy_proj(tfy_embed).unsqueeze(1)

        if self.use_rollout_depth_for_horizon and self.tfy_dim >= 5:
            rollout_norm = tfy[..., -1].clamp(0.0, 1.0)
            horizon_idx = torch.round(rollout_norm * 511.0).to(torch.long)
        else:
            horizon_idx = torch.zeros(batch_size, dtype=torch.long, device=src.device)
        q = q + self.horizon_embed(horizon_idx).unsqueeze(1)

        attended, _ = self.cross_attention(q, x, x)
        return self.decoder(attended.squeeze(1)).squeeze(-1)
