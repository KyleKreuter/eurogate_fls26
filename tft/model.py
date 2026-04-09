# tft_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class TFTConfig:
    static_dim: int
    observed_dim: int
    known_future_dim: int
    output_dim: int = 2                  # [point, p90] by default
    hidden_dim: int = 128
    lstm_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.1
    max_encoder_length: int = 168
    max_decoder_length: int = 24

    # Timestamp embedding config
    use_timestamp_embedding: bool = True
    timestamp_num_embeddings: int = 0    # set >0 if using learned discrete embedding
    timestamp_embedding_dim: int = 32
    timestamp_continuous_dim: int = 0    # number of continuous timestamp channels already provided
    fuse_timestamp_into_known: bool = True


# ============================================================
# Utility blocks
# ============================================================

class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GateAddNorm(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.glu(x)
        return self.norm(x + residual)


class GRN(nn.Module):
    """
    Gated Residual Network.
    Supports optional context.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, context_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim is not None else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.elu = nn.ELU()
        self.gate_norm = GateAddNorm(output_dim, dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            # Broadcast context when needed
            while context.dim() < h.dim():
                context = context.unsqueeze(1)
            h = h + self.context_fc(context)
        h = self.elu(h)
        h = self.fc2(h)
        return self.gate_norm(h, residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection over last dimension variables.
    Input:
      x: [B, T, N, D] or [B, N, D]
      context: optional [B, C]
    Output:
      selected: [B, T, H] or [B, H]
      weights:  [B, T, N] or [B, N]
    """
    def __init__(self, num_vars: int, var_dim: int, hidden_dim: int, context_dim: Optional[int], dropout: float):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        self.var_grns = nn.ModuleList(
            [GRN(var_dim, hidden_dim, output_dim=hidden_dim, context_dim=context_dim, dropout=dropout) for _ in range(num_vars)]
        )
        self.weight_grn = GRN(num_vars * var_dim, hidden_dim, output_dim=num_vars, context_dim=context_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        original_rank = x.dim()

        if original_rank == 3:
            # [B, N, D] -> [B, 1, N, D]
            x = x.unsqueeze(1)

        b, t, n, d = x.shape
        assert n == self.num_vars, f"Expected {self.num_vars} vars, got {n}"

        processed = []
        for i in range(n):
            xi = x[:, :, i, :]  # [B, T, D]
            processed.append(self.var_grns[i](xi, context=context))  # [B, T, H]
        processed = torch.stack(processed, dim=2)  # [B, T, N, H]

        flat = x.reshape(b, t, n * d)
        weights = self.weight_grn(flat, context=context)  # [B, T, N]
        weights = torch.softmax(weights, dim=-1)

        selected = (processed * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, H]

        if original_rank == 3:
            selected = selected.squeeze(1)   # [B, H]
            weights = weights.squeeze(1)     # [B, N]

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Simplified interpretable multi-head attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, tq, h = q.shape
        tk = k.size(1)

        q = self.q_proj(q).view(b, tq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, Hh, Tq, Hd]
        k = self.k_proj(k).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, Hh, Tk, Hd]
        v = self.v_proj(v).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, Hh, Tk, Hd]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, Hh, Tq, Tk]

        if attn_mask is not None:
            # attn_mask expected [Tq, Tk] or [B, Tq, Tk], with True for allowed positions
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,Tq,Tk]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)               # [B,1,Tq,Tk]
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, v)  # [B, Hh, Tq, Hd]
        out = out.transpose(1, 2).contiguous().view(b, tq, h)
        out = self.out_proj(out)

        # Mean attention over heads for interpretability
        attn_mean = weights.mean(dim=1)  # [B, Tq, Tk]
        return out, attn_mean


# ============================================================
# Timestamp embedding
# ============================================================

class TimestampEmbedding(nn.Module):
    """
    Easily expandable timestamp encoder.

    Supports:
    - continuous timestamp channels (e.g. hour_sin, hour_cos, weekday_sin, ...)
    - learned discrete embeddings (e.g. hour_id, weekday_id bucketized externally)

    Expected inputs:
      ts_cont: [B, T, Cc] or None
      ts_index: [B, T] long or None
    Output:
      [B, T, E]
    """
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.use_timestamp_embedding = config.use_timestamp_embedding
        self.has_cont = config.timestamp_continuous_dim > 0
        self.has_disc = config.timestamp_num_embeddings > 0

        pieces = []
        if self.has_cont:
            self.cont_proj = nn.Sequential(
                nn.Linear(config.timestamp_continuous_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.timestamp_embedding_dim),
            )
            pieces.append(config.timestamp_embedding_dim)
        else:
            self.cont_proj = None

        if self.has_disc:
            self.disc_emb = nn.Embedding(config.timestamp_num_embeddings, config.timestamp_embedding_dim)
            pieces.append(config.timestamp_embedding_dim)
        else:
            self.disc_emb = None

        final_in = sum(pieces)
        if final_in == 0:
            self.fuse = None
            self.out_dim = 0
        else:
            self.fuse = nn.Sequential(
                nn.Linear(final_in, config.timestamp_embedding_dim),
                nn.ReLU(),
                nn.Linear(config.timestamp_embedding_dim, config.timestamp_embedding_dim),
            )
            self.out_dim = config.timestamp_embedding_dim

    def forward(
        self,
        ts_cont: Optional[torch.Tensor] = None,
        ts_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_timestamp_embedding or self.fuse is None:
            return None

        parts = []
        if self.cont_proj is not None:
            if ts_cont is None:
                raise ValueError("Timestamp continuous features expected but ts_cont is None")
            parts.append(self.cont_proj(ts_cont))

        if self.disc_emb is not None:
            if ts_index is None:
                raise ValueError("Timestamp discrete indices expected but ts_index is None")
            parts.append(self.disc_emb(ts_index))

        x = torch.cat(parts, dim=-1)
        return self.fuse(x)


# ============================================================
# Main model
# ============================================================

class ReeferTFT(nn.Module):
    """
    TFT-style model for aggregate reefer load forecasting.

    Inputs
    ------
    static_x:
        [B, static_dim] or None

    observed_x:
        [B, encoder_len, observed_dim]
        Historical observed features, including lagged target history if desired.

    known_future_x:
        [B, encoder_len + decoder_len, known_future_dim]
        Known covariates over full horizon (calendar features, allowed future weather, etc.)

    timestamp_cont:
        [B, encoder_len + decoder_len, timestamp_continuous_dim] or None

    timestamp_index:
        [B, encoder_len + decoder_len] long or None

    Returns
    -------
    dict with:
      "forecast": [B, decoder_len, output_dim]
      "point":    [B, decoder_len]
      "p90":      [B, decoder_len]
      "attn":     [B, decoder_len, encoder_len + decoder_len]
      "encoder_var_weights": optional var selection weights
      "decoder_var_weights": optional var selection weights
    """
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # Static
        self.static_proj = nn.Linear(config.static_dim, h) if config.static_dim > 0 else None
        self.static_context_grn = GRN(h, h, output_dim=h, dropout=config.dropout) if config.static_dim > 0 else None

        # Per-variable projections so future expansion is easy
        self.observed_var_proj = nn.ModuleList([nn.Linear(1, h) for _ in range(config.observed_dim)])
        self.known_var_proj = nn.ModuleList([nn.Linear(1, h) for _ in range(config.known_future_dim)])

        self.timestamp_embedding = TimestampEmbedding(config)
        self.timestamp_to_hidden = (
            nn.Linear(self.timestamp_embedding.out_dim, h)
            if self.timestamp_embedding.out_dim > 0 and config.fuse_timestamp_into_known
            else None
        )

        # Variable selection
        self.encoder_vsn = VariableSelectionNetwork(
            num_vars=config.observed_dim + config.known_future_dim + (1 if self.timestamp_to_hidden is not None else 0),
            var_dim=h,
            hidden_dim=h,
            context_dim=h if config.static_dim > 0 else None,
            dropout=config.dropout,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            num_vars=config.known_future_dim + (1 if self.timestamp_to_hidden is not None else 0),
            var_dim=h,
            hidden_dim=h,
            context_dim=h if config.static_dim > 0 else None,
            dropout=config.dropout,
        )

        # Sequence modeling
        self.encoder_lstm = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )

        self.post_lstm_gate_encoder = GateAddNorm(h, config.dropout)
        self.post_lstm_gate_decoder = GateAddNorm(h, config.dropout)

        # Static enrichment
        self.static_enrichment = GRN(h, h, output_dim=h, context_dim=h if config.static_dim > 0 else None, dropout=config.dropout)

        # Attention over full temporal context
        self.attention = InterpretableMultiHeadAttention(h, config.num_attention_heads, config.dropout)
        self.post_attn_gate = GateAddNorm(h, config.dropout)

        # Position-wise feedforward
        self.pos_ff = GRN(h, h, output_dim=h, dropout=config.dropout)
        self.post_ff_gate = GateAddNorm(h, config.dropout)

        # Output projection
        self.output_layer = nn.Linear(h, config.output_dim)

    @staticmethod
    def _split_time(
        x: torch.Tensor,
        encoder_len: int,
        decoder_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, :encoder_len], x[:, encoder_len:encoder_len + decoder_len]

    def _project_per_variable(self, x: torch.Tensor, projectors: nn.ModuleList) -> torch.Tensor:
        """
        x: [B, T, N]
        returns: [B, T, N, H]
        """
        vars_out = []
        for i, proj in enumerate(projectors):
            xi = x[:, :, i:i+1]
            vars_out.append(proj(xi))
        return torch.stack(vars_out, dim=2)

    def _build_static_context(self, static_x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.static_proj is None or static_x is None:
            return None
        s = self.static_proj(static_x)
        s = self.static_context_grn(s)
        return s

    def _build_timestamp_token(
        self,
        timestamp_cont: Optional[torch.Tensor],
        timestamp_index: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.timestamp_to_hidden is None:
            return None
        ts_emb = self.timestamp_embedding(timestamp_cont, timestamp_index)
        if ts_emb is None:
            return None
        return self.timestamp_to_hidden(ts_emb)  # [B, T, H]

    def _causal_mask(self, tq: int, tk: int, device: torch.device) -> torch.Tensor:
        """
        Allow each decoder position to attend only to past/full encoder and past decoder positions.
        Returns [tq, tk] boolean mask.
        """
        # We assume keys are [encoder positions..., decoder positions...]
        mask = torch.ones(tq, tk, dtype=torch.bool, device=device)
        enc_len = tk - tq
        for i in range(tq):
            # future decoder positions beyond current are masked out
            allowed_decoder_upto = enc_len + i
            if allowed_decoder_upto + 1 < tk:
                mask[i, allowed_decoder_upto + 1:] = False
        return mask

    def forward(
        self,
        observed_x: torch.Tensor,
        known_future_x: torch.Tensor,
        static_x: Optional[torch.Tensor] = None,
        timestamp_cont: Optional[torch.Tensor] = None,
        timestamp_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, enc_len, obs_dim = observed_x.shape
        _, total_len, known_dim = known_future_x.shape
        dec_len = total_len - enc_len

        if obs_dim != self.config.observed_dim:
            raise ValueError(f"observed_dim mismatch: got {obs_dim}, expected {self.config.observed_dim}")
        if known_dim != self.config.known_future_dim:
            raise ValueError(f"known_future_dim mismatch: got {known_dim}, expected {self.config.known_future_dim}")
        if dec_len <= 0:
            raise ValueError("known_future_x must cover encoder_len + decoder_len")

        static_ctx = self._build_static_context(static_x)

        # Project observed and known variables
        observed_proj = self._project_per_variable(observed_x, self.observed_var_proj)        # [B, Enc, ObsN, H]
        known_proj = self._project_per_variable(known_future_x, self.known_var_proj)          # [B, Enc+Dec, KnownN, H]
        known_enc, known_dec = self._split_time(known_proj, enc_len, dec_len)

        # Timestamp token as an extra "variable"
        ts_token = self._build_timestamp_token(timestamp_cont, timestamp_index)  # [B, T, H] or None
        if ts_token is not None:
            ts_enc, ts_dec = self._split_time(ts_token, enc_len, dec_len)
            ts_enc = ts_enc.unsqueeze(2)  # [B, Enc, 1, H]
            ts_dec = ts_dec.unsqueeze(2)  # [B, Dec, 1, H]

        # Encoder variable selection over [observed vars + known vars (+ timestamp token)]
        enc_vars = [observed_proj, known_enc]
        if ts_token is not None:
            enc_vars.append(ts_enc)
        enc_input = torch.cat(enc_vars, dim=2)  # [B, Enc, N_enc_vars, H]

        enc_selected, enc_weights = self.encoder_vsn(enc_input, context=static_ctx)  # [B, Enc, H]

        # Decoder variable selection over [known vars (+ timestamp token)]
        dec_vars = [known_dec]
        if ts_token is not None:
            dec_vars.append(ts_dec)
        dec_input = torch.cat(dec_vars, dim=2)  # [B, Dec, N_dec_vars, H]

        dec_selected, dec_weights = self.decoder_vsn(dec_input, context=static_ctx)  # [B, Dec, H]

        # LSTM encoder/decoder
        enc_lstm_out, (h_n, c_n) = self.encoder_lstm(enc_selected)
        enc_lstm_out = self.post_lstm_gate_encoder(enc_lstm_out, enc_selected)

        dec_lstm_out, _ = self.decoder_lstm(dec_selected, (h_n, c_n))
        dec_lstm_out = self.post_lstm_gate_decoder(dec_lstm_out, dec_selected)

        # Static enrichment
        full_seq = torch.cat([enc_lstm_out, dec_lstm_out], dim=1)   # [B, Enc+Dec, H]
        enriched = self.static_enrichment(full_seq, context=static_ctx)

        # Decoder queries attend over full sequence
        decoder_query = enriched[:, enc_len:]   # [B, Dec, H]
        keys_values = enriched                  # [B, Enc+Dec, H]

        attn_mask = self._causal_mask(dec_len, enc_len + dec_len, device=observed_x.device)
        attn_out, attn_weights = self.attention(decoder_query, keys_values, keys_values, attn_mask=attn_mask)
        attn_out = self.post_attn_gate(attn_out, decoder_query)
        ff = self.pos_ff(attn_out)
        ff = self.post_ff_gate(ff, attn_out)

        out = self.output_layer(ff)  # [B, Dec, output_dim]

        result = {
            "attn": attn_weights,
            "encoder_var_weights": enc_weights,
            "decoder_var_weights": dec_weights,
        }

        if self.config.output_dim < 1:
            raise ValueError("output_dim must be at least 1")

        point = out[:, :, 0]
        result["point"] = point

        if self.config.output_dim >= 2:
            raw_p90_margin = out[:, :, 1]
            p90 = point + F.softplus(raw_p90_margin)
            result["p90"] = p90
            result["forecast"] = torch.stack([point, p90], dim=-1)
        else:
            result["forecast"] = point.unsqueeze(-1)

        return result