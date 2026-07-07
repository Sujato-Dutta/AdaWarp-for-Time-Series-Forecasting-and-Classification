"""AdaWarp-MVPF+: ablation-pruned MVPF with an adaptive linear field skip.

This file intentionally does not modify ``adawarp_mvpf.py``. The original
AdaWarp-MVPF results remain reproducible. MVPF+ keeps the same multiscale
variate-patch field backbone, defaults off the components that were neutral or
slightly harmful in matched ablations (prototype memory and frequency gates),
and adds a global prefix-only linear field skip for smooth long-horizon regimes
such as ETTh1.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from adawarp_mvpf import AdaWarpMVPFForecaster


class AdaptiveLinearFieldHead(nn.Module):
    """Global linear latent-field candidate bank."""

    def __init__(self, seq_len: int, pred_len: int, dropout: float = 0.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.center_head = nn.Linear(self.seq_len, self.pred_len)
        self.trend_head = nn.Linear(self.seq_len, self.pred_len)
        self.residual_head = nn.Linear(self.seq_len, self.pred_len)
        self.gate = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4),
        )
        self._init_conservative()

    def _init_conservative(self) -> None:
        nn.init.zeros_(self.center_head.weight)
        nn.init.zeros_(self.center_head.bias)
        nn.init.xavier_uniform_(self.trend_head.weight, gain=0.20)
        nn.init.zeros_(self.trend_head.bias)
        nn.init.xavier_uniform_(self.residual_head.weight, gain=0.20)
        nn.init.zeros_(self.residual_head.bias)
        final_gate = self.gate[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            with torch.no_grad():
                final_gate.bias.copy_(torch.tensor([0.75, -0.75, 0.25, 0.25], dtype=final_gate.bias.dtype))

    def _moving_average(self, values: torch.Tensor) -> torch.Tensor:
        short_kernel = 7
        long_kernel = min(25, self.seq_len if self.seq_len % 2 == 1 else self.seq_len - 1)
        long_kernel = max(3, long_kernel)

        def avg(kernel: int) -> torch.Tensor:
            pad = kernel // 2
            channel_first = values.permute(0, 2, 1)
            padded = F.pad(channel_first, (pad, pad), mode="replicate")
            return F.avg_pool1d(padded, kernel_size=kernel, stride=1).permute(0, 2, 1).contiguous()

        return 0.35 * avg(short_kernel) + 0.65 * avg(long_kernel)

    def _linear_per_channel(self, head: nn.Linear, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        flat = values.permute(0, 2, 1).reshape(batch * channels, length)
        forecast = head(flat)
        return forecast.reshape(batch, channels, self.pred_len).permute(0, 2, 1).contiguous()

    def _analytic_trend(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time = torch.linspace(-1.0, 1.0, self.seq_len, device=values.device, dtype=values.dtype)
        centered_time = time - time.mean()
        denom = centered_time.pow(2).sum().clamp_min(1e-6)
        mean = values.mean(dim=1, keepdim=True)
        slope = ((values - mean) * centered_time[None, :, None]).sum(dim=1, keepdim=True) / denom
        step = 2.0 / max(1, self.seq_len - 1)
        future_time = 1.0 + step * torch.arange(1, self.pred_len + 1, device=values.device, dtype=values.dtype)
        forecast = mean + slope * future_time[None, :, None]
        fitted = mean + slope * time[None, :, None]
        fit_error = (values - fitted).pow(2).mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return forecast, fit_error

    def _features(self, values: torch.Tensor, fit_error: torch.Tensor) -> torch.Tensor:
        slope = (values[:, -1, :] - values[:, 0, :]).mean(dim=1, keepdim=True)
        volatility = values.std(dim=(1, 2), unbiased=False).unsqueeze(1)
        roughness = (values[:, 1:, :] - values[:, :-1, :]).abs().mean(dim=(1, 2)).unsqueeze(1)
        level = values[:, -1, :].abs().mean(dim=1, keepdim=True)
        return torch.cat([slope, volatility, roughness, level, fit_error], dim=1)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last = values[:, -1:, :]
        centered = values - last
        nlinear = self._linear_per_channel(self.center_head, centered) + last
        trend = self._moving_average(values)
        residual = values - trend
        dlinear = self._linear_per_channel(self.trend_head, trend) + self._linear_per_channel(self.residual_head, residual)
        analytic, fit_error = self._analytic_trend(values)
        persistence = last.expand(-1, self.pred_len, -1)
        components = torch.stack([nlinear, dlinear, analytic, persistence], dim=1)
        weights = torch.softmax(self.gate(self._features(values, fit_error)), dim=-1)
        forecast = (components * weights[:, :, None, None]).sum(dim=1)
        return forecast, weights, fit_error


class AdaWarpMVPFPlusForecaster(nn.Module):
    """Ablation-pruned MVPF plus adaptive linear field skip."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        patch_lens: Sequence[int] = (8, 16, 32),
        width: int = 128,
        depth: int = 2,
        dropout: float = 0.05,
        num_prototypes: int = 8,
        max_shift: int = 2,
        reconstruction_weight: float = 0.03,
        use_prototype_memory: bool = False,
        use_frequency_gate: bool = False,
        use_adaptive_shifts: bool = True,
        use_adaptive_radius: bool = True,
        use_component_gate: bool = True,
        use_trend_decomposition: bool = True,
        use_linear_field: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.reconstruction_weight = float(reconstruction_weight)
        self.use_linear_field = bool(use_linear_field)
        self.backbone = AdaWarpMVPFForecaster(
            seq_len,
            pred_len,
            patch_lens=patch_lens,
            width=width,
            depth=depth,
            dropout=dropout,
            num_prototypes=num_prototypes,
            max_shift=max_shift,
            reconstruction_weight=reconstruction_weight,
            use_prototype_memory=use_prototype_memory,
            use_adaptive_shifts=use_adaptive_shifts,
            use_frequency_gate=use_frequency_gate,
            use_adaptive_radius=use_adaptive_radius,
            use_component_gate=use_component_gate,
            use_trend_decomposition=use_trend_decomposition,
        )
        self.linear_field = AdaptiveLinearFieldHead(seq_len, pred_len, dropout=dropout)
        self.mean_mixer = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        final = self.mean_mixer[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            with torch.no_grad():
                final.bias.copy_(torch.tensor([0.65, 0.15], dtype=final.bias.dtype))

    def _summary_features(
        self,
        normalized: torch.Tensor,
        residual: torch.Tensor,
        linear_confidence: torch.Tensor,
        fit_error: torch.Tensor,
    ) -> torch.Tensor:
        slope = (normalized[:, -1, :] - normalized[:, 0, :]).mean(dim=1, keepdim=True)
        volatility = residual.std(dim=(1, 2), unbiased=False).unsqueeze(1)
        roughness = (normalized[:, 1:, :] - normalized[:, :-1, :]).abs().mean(dim=(1, 2)).unsqueeze(1)
        level = normalized[:, -1, :].abs().mean(dim=1, keepdim=True)
        return torch.cat([slope, volatility, roughness, level, linear_confidence, fit_error], dim=1)

    def forward_with_aux(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        backbone_mean = self.backbone(x_enc, None, None, None)
        normalized, mean, std = self.backbone._normalize(x_enc)
        _, residual = self.backbone._decompose(normalized)
        backbone_norm = (backbone_mean - mean) / std.clamp_min(1e-5)
        if self.use_linear_field:
            linear_norm, linear_component_weights, fit_error = self.linear_field(normalized)
            linear_confidence = linear_component_weights.max(dim=1, keepdim=True).values
        else:
            linear_norm = backbone_norm
            linear_component_weights = backbone_norm.new_zeros(backbone_norm.shape[0], 4)
            fit_error = backbone_norm.new_zeros(backbone_norm.shape[0], 1)
            linear_confidence = backbone_norm.new_zeros(backbone_norm.shape[0], 1)
        mix_logits = self.mean_mixer(self._summary_features(normalized, residual, linear_confidence, fit_error))
        mix_weights = torch.softmax(mix_logits, dim=-1)
        forecast_norm = mix_weights[:, 0, None, None] * backbone_norm + mix_weights[:, 1, None, None] * linear_norm
        forecast = forecast_norm * std + mean
        aux = {
            "backbone_gate": mix_weights[:, 0].mean(),
            "linear_field_gate": mix_weights[:, 1].mean(),
            "linear_component_confidence": linear_confidence.mean(),
            "linear_fit_error": fit_error.mean(),
        }
        return forecast, aux

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        forecast, _ = self.forward_with_aux(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return forecast

    def auxiliary_loss(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.backbone.auxiliary_loss(x_enc)


__all__ = ["AdaWarpMVPFPlusForecaster", "AdaptiveLinearFieldHead"]