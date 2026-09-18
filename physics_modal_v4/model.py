"""LatentFlow.

The model follows the requested order exactly:

    observed prefix X
      -> explicit causal structural latent Z = E_phi(X)
      -> additive sparse stochastic-process posterior {Z_q}
      -> process-preserving MVPF fields {H_q^ell}
      -> late process decoding and finalized MVPF continuation
      -> probabilistic forecast with validation-selected point training.

The process axis is never collapsed before field evolution. Every process is
continued independently by the shared MVPF field operators, and fusion occurs
only after future process fields have been decoded. A frozen copy of the
finalized variate-patch backbone is an exact validation fallback and the
matched probabilistic reference.
"""

from __future__ import annotations

import copy
import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster


PROCESS_SPECS = {
    "ETT": (
        ("load_daily", "periodic", 1.0, 0.65),
        ("load_weekly", "periodic", 7.0, 1.25),
        ("thermal", "matern", 0.0, 2.0),
        ("innovation", "ou", 0.0, 0.20),
    ),
    "Weather": (
        ("solar_daily", "periodic", 1.0, 0.65),
        ("thermodynamic", "matern", 0.0, 2.0),
        ("humidity", "matern", 0.0, 0.75),
        ("wind", "ou", 0.0, 0.20),
        ("precipitation", "ou", 0.0, 0.06),
        ("slow_state", "matern", 0.0, 7.0),
    ),
    "Electricity": (
        ("daily", "periodic", 1.0, 0.75),
        ("weekly", "periodic", 7.0, 1.25),
        ("baseline", "matern", 0.0, 7.0),
        ("irregular", "ou", 0.0, 0.20),
    ),
    "Traffic": (
        ("daily", "periodic", 1.0, 0.75),
        ("weekly", "periodic", 7.0, 1.25),
        ("baseline", "matern", 0.0, 7.0),
        ("congestion", "ou", 0.0, 0.15),
    ),
}


# Sensor classes come from the repository's dataset audit. ETT and Weather
# expose physical semantics, so their membership is fixed. Electricity and
# Traffic expose anonymous instances of one measurement type; their soft
# behavioural membership is inferred from each observed prefix instead.
SENSOR_GROUP_SPECS = {
    'ETT': {
        'names': ('electrical_load', 'oil_temperature'),
        'indices': ((0, 1, 2, 3, 4, 5), (6,)),
        'process_prior': (
            (0.48, 0.27, 0.05, 0.20),
            (0.18, 0.07, 0.65, 0.10),
        ),
    },
    'Weather': {
        'names': (
            'pressure_density',
            'thermodynamic',
            'humidity',
            'wind',
            'precipitation',
            'radiation',
            'atmospheric_composition',
        ),
        'indices': (
            (0, 10),
            (1, 2, 3, 19),
            (4, 5, 6, 7, 8, 9),
            (11, 12, 13),
            (14, 15),
            (16, 17, 18),
            (20,),
        ),
        # Columns follow PROCESS_SPECS[Weather]. The bounded learnable
        # adjustment below can refine, but cannot erase, this audit prior.
        'process_prior': (
            (0.04, 0.30, 0.04, 0.03, 0.01, 0.58),
            (0.24, 0.46, 0.06, 0.03, 0.01, 0.20),
            (0.14, 0.14, 0.48, 0.05, 0.04, 0.15),
            (0.08, 0.06, 0.04, 0.66, 0.08, 0.08),
            (0.05, 0.03, 0.10, 0.07, 0.70, 0.05),
            (0.72, 0.10, 0.03, 0.03, 0.07, 0.05),
            (0.18, 0.12, 0.08, 0.04, 0.03, 0.55),
        ),
    },
    'Electricity': {
        'names': (
            'strongly_periodic',
            'irregular_high_variability',
            'low_activity',
        ),
        'process_prior': (
            (0.48, 0.30, 0.17, 0.05),
            (0.17, 0.10, 0.13, 0.60),
            (0.15, 0.10, 0.60, 0.15),
        ),
        'group_prior': (0.626168, 0.355140, 0.018692),
    },
    'Traffic': {
        'names': (
            'strongly_periodic',
            'irregular_high_variability',
            'low_activity',
        ),
        'process_prior': (
            (0.48, 0.30, 0.17, 0.05),
            (0.17, 0.10, 0.13, 0.60),
            (0.15, 0.10, 0.60, 0.15),
        ),
        'group_prior': (0.406032, 0.402552, 0.191415),
    },
}


def domain_for_dataset(dataset: str) -> str:
    if dataset.startswith("ETT"):
        return "ETT"
    if dataset not in PROCESS_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return dataset


def samples_per_day(dataset: str) -> int:
    if dataset.startswith("ETTm"):
        return 96
    if dataset == "Weather":
        return 144
    return 24


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return value if value > 20.0 else math.log(math.expm1(value))


def student_t_log_prob(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
) -> torch.Tensor:
    """Elementwise Student-t log density."""

    standardized = (target - location) / scale
    half_df = 0.5 * degrees_of_freedom
    negative_log_prob = (
        torch.lgamma(half_df)
        - torch.lgamma(half_df + 0.5)
        + 0.5 * (degrees_of_freedom.log() + math.log(math.pi))
        + scale.log()
        + (half_df + 0.5) * torch.log1p(standardized.square() / degrees_of_freedom)
    )
    return -negative_log_prob


class SparseStochasticKernel(nn.Module):
    """Trainable covariance with ordered Motion-Code-style inducing times."""

    def __init__(
        self,
        name: str,
        family: str,
        *,
        seq_len: int,
        samples_day: int,
        nominal_period_days: float,
        nominal_length_days: float,
        inducing_points: int,
        initial_amplitude: float,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.family = str(family)
        self.seq_len = int(seq_len)
        self.samples_day = int(samples_day)
        self.inducing_points = min(int(inducing_points), self.seq_len)
        self.nominal_period = max(2.0, nominal_period_days * samples_day)
        nominal_length = (
            max(0.10, nominal_length_days)
            if self.family == "periodic"
            else max(0.75, nominal_length_days * samples_day)
        )

        self.raw_amplitude = nn.Parameter(torch.tensor(inverse_softplus(initial_amplitude)))
        self.raw_length = nn.Parameter(torch.tensor(inverse_softplus(nominal_length)))
        if self.family == "periodic":
            self.raw_decay = nn.Parameter(
                torch.tensor(inverse_softplus(max(self.nominal_period * 2.0, samples_day)))
            )
            self.raw_period_adjustment = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("raw_decay", None)
            self.register_parameter("raw_period_adjustment", None)

        # m ordered interior inducing timestamps are generated from m+1
        # positive gaps, so collisions and reversed time are impossible.
        self.raw_inducing_gaps = nn.Parameter(torch.full((self.inducing_points + 1,), 0.50))

    def inducing_times(self) -> torch.Tensor:
        gaps = F.softplus(self.raw_inducing_gaps) + 1e-4
        cumulative = torch.cumsum(gaps[:-1], dim=0)
        return (self.seq_len - 1.0) * cumulative / gaps.sum()

    def parameters_physical(self) -> dict[str, torch.Tensor]:
        amplitude = F.softplus(self.raw_amplitude).clamp_min(1e-6)
        length = F.softplus(self.raw_length).clamp_min(0.25)
        result = {"amplitude": amplitude, "length": length}
        if self.family == "periodic":
            result["decay"] = F.softplus(self.raw_decay).clamp_min(1.0)
            result["period"] = self.nominal_period * (
                0.20 * torch.tanh(self.raw_period_adjustment)
            ).exp()
        return result

    def kernel(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distance = (left[:, None] - right[None, :]).abs()
        parameters = self.parameters_physical()
        length = parameters["length"]
        if self.family == "periodic":
            phase = torch.sin(math.pi * distance / parameters["period"])
            covariance = torch.exp(-2.0 * phase.square() / length.square())
            covariance = covariance * torch.exp(-distance / parameters["decay"])
        elif self.family == "matern":
            scaled = math.sqrt(3.0) * distance / length
            covariance = (1.0 + scaled) * torch.exp(-scaled)
        elif self.family == "ou":
            covariance = torch.exp(-distance / length)
        else:
            raise ValueError(f"Unknown kernel family: {self.family}")
        return parameters["amplitude"] * covariance

    def sparse_covariances(
        self,
        past_times: torch.Tensor,
        future_times: torch.Tensor,
        jitter: float,
    ) -> dict[str, torch.Tensor]:
        inducing = self.inducing_times().to(device=past_times.device, dtype=past_times.dtype)
        kuu = self.kernel(inducing, inducing)
        kuu = kuu + float(jitter) * torch.eye(
            self.inducing_points, device=kuu.device, dtype=kuu.dtype
        )
        chol = torch.linalg.cholesky(kuu)
        ktu = self.kernel(past_times, inducing)
        kuf = self.kernel(inducing, future_times)
        solved_ut = torch.cholesky_solve(ktu.transpose(0, 1), chol)
        qtt = ktu @ solved_ut
        qtt = 0.5 * (qtt + qtt.transpose(0, 1))
        qft = kuf.transpose(0, 1) @ solved_ut
        solved_uf = torch.cholesky_solve(kuf, chol)
        qff_diagonal = (kuf * solved_uf).sum(dim=0)
        full_past = self.kernel(past_times, past_times)
        full_future_diagonal = self.kernel(
            future_times, future_times
        ).diagonal()
        trace_gap = (full_past.diagonal() - qtt.diagonal()).clamp_min(0.0).sum()
        return {
            "past": qtt,
            "past_diagonal": full_past.diagonal(),
            "future_past": qft,
            "future_diagonal": full_future_diagonal,
            "nystrom_future_diagonal": qff_diagonal,
            "trace_gap": trace_gap,
            "inducing_times": inducing,
        }

    @torch.no_grad()
    def project(self) -> None:
        self.raw_amplitude.clamp_(-8.0, 5.0)
        self.raw_length.clamp_(-8.0, max(32.0, 14.0 * self.samples_day))
        self.raw_inducing_gaps.clamp_(-8.0, 8.0)
        if self.raw_decay is not None:
            self.raw_decay.clamp_(-8.0, max(64.0, 28.0 * self.samples_day))
        if self.raw_period_adjustment is not None:
            self.raw_period_adjustment.clamp_(-5.0, 5.0)


class SensorProcessGeometry(nn.Module):
    '''Audit-grounded sensor-to-process geometry.'''

    def __init__(
        self,
        dataset: str,
        channels: int,
        num_processes: int,
        *,
        neutral_soft_assignment: bool = False,
        uninformed_assignment: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = str(dataset)
        self.domain = domain_for_dataset(dataset)
        self.channels = int(channels)
        self.num_processes = int(num_processes)
        self.neutral_soft_assignment = bool(neutral_soft_assignment)
        self.uninformed_assignment = bool(uninformed_assignment)
        if self.neutral_soft_assignment and self.uninformed_assignment:
            raise ValueError("Assignment controls are mutually exclusive.")
        spec = SENSOR_GROUP_SPECS[self.domain]
        if self.neutral_soft_assignment:
            # Each neutral group names one stochastic process. Sensor-to-group
            # membership is learned without audited labels or domain priors.
            self.group_names = tuple(
                f'latent_process_{index}' for index in range(self.num_processes)
            )
            self.num_groups = self.num_processes
            prior = torch.eye(self.num_processes, dtype=torch.float32)
        else:
            self.group_names = tuple(spec['names'])
            self.num_groups = len(self.group_names)
            prior = (
                torch.ones(self.num_groups, self.num_processes)
                if self.uninformed_assignment
                else torch.tensor(spec['process_prior'], dtype=torch.float32)
            )
        if prior.shape != (self.num_groups, self.num_processes):
            raise ValueError('Invalid sensor/process prior shape.')
        self.register_buffer('log_process_prior', prior.clamp_min(1e-6).log())
        self.raw_process_adjustment = nn.Parameter(torch.zeros_like(prior))
        self.semantic_membership = (
            not self.neutral_soft_assignment
            and not self.uninformed_assignment
            and self.domain in {'ETT', 'Weather'}
        )
        self._configure_membership(spec)

    def _configure_membership(self, spec: dict) -> None:
        if self.neutral_soft_assignment:
            self.register_buffer('fixed_membership', torch.empty(0))
            self.register_buffer(
                'log_group_prior',
                torch.full((self.num_groups,), -math.log(self.num_groups)),
            )
            self.behaviour_router = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, 24),
                nn.GELU(),
                nn.Linear(24, self.num_groups),
            )
            # Near-uniform, but not exactly symmetric, so the processes can
            # specialize without encoding a semantic assignment by hand.
            nn.init.normal_(self.behaviour_router[-1].weight, std=1e-3)
            nn.init.zeros_(self.behaviour_router[-1].bias)
            return
        if self.uninformed_assignment:
            self.register_buffer('fixed_membership', torch.empty(0))
            self.register_buffer(
                'log_group_prior',
                torch.full((self.num_groups,), -math.log(self.num_groups)),
            )
            self.behaviour_router = None
            self.channel_group_logits = nn.Parameter(
                torch.empty(self.channels, self.num_groups)
            )
            nn.init.normal_(self.channel_group_logits, std=1e-3)
            return
        if self.semantic_membership:
            membership = torch.zeros(self.channels, self.num_groups)
            assigned = set()
            for group, indices in enumerate(spec['indices']):
                for index in indices:
                    if index >= self.channels:
                        raise ValueError(
                            f'{self.dataset} is missing audited sensor {index}.'
                        )
                    membership[index, group] = 1.0
                    assigned.add(index)
            if assigned != set(range(self.channels)):
                raise ValueError(
                    f'{self.dataset} does not match its audited sensor schema.'
                )
            self.register_buffer('fixed_membership', membership)
            self.register_buffer(
                'log_group_prior',
                torch.full((self.num_groups,), -math.log(self.num_groups)),
            )
            self.behaviour_router = None
            return
        self.register_buffer('fixed_membership', torch.empty(0))
        group_prior = torch.tensor(spec['group_prior'], dtype=torch.float32)
        self.register_buffer(
            'log_group_prior', group_prior.clamp_min(1e-6).log()
        )
        self.behaviour_router = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 24),
            nn.GELU(),
            nn.Linear(24, self.num_groups),
        )
        nn.init.zeros_(self.behaviour_router[-1].weight)
        with torch.no_grad():
            self.behaviour_router[-1].bias.copy_(self.log_group_prior)

    def process_weights(self) -> torch.Tensor:
        if self.neutral_soft_assignment:
            # Membership is a direct channel-to-process assignment.
            return torch.eye(
                self.num_processes,
                device=self.log_process_prior.device,
                dtype=self.log_process_prior.dtype,
            )
        logits = self.log_process_prior + 1.25 * torch.tanh(
            self.raw_process_adjustment
        )
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _lag_correlation(values: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0 or values.shape[1] <= lag + 1:
            return values.new_zeros(values.shape[0], values.shape[2])
        left = values[:, :-lag]
        right = values[:, lag:]
        left = left - left.mean(dim=1, keepdim=True)
        right = right - right.mean(dim=1, keepdim=True)
        numerator = (left * right).mean(dim=1)
        denominator = (
            left.square().mean(dim=1) * right.square().mean(dim=1)
        ).clamp_min(1e-8).sqrt()
        return numerator / denominator

    def _behaviour_features(self, values: torch.Tensor) -> torch.Tensor:
        day = min(samples_per_day(self.dataset), max(2, values.shape[1] // 2))
        half_day = max(1, day // 2)
        deviation = values - values.mean(dim=1, keepdim=True)
        scale = deviation.square().mean(dim=1).clamp_min(1e-8).sqrt()
        features = torch.stack(
            [
                values.abs().mean(dim=1),
                scale,
                (values[:, 1:] - values[:, :-1]).abs().mean(dim=1),
                values.abs().amax(dim=1) / scale,
                self._lag_correlation(values, day),
                self._lag_correlation(values, half_day),
            ],
            dim=-1,
        )
        center = features.mean(dim=1, keepdim=True)
        spread = features.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        return (features - center) / spread

    def forward(
        self, sensor_prefix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = sensor_prefix.shape[0]
        if sensor_prefix.shape[2] != self.channels:
            raise ValueError('Unexpected channel count in sensor geometry.')
        if self.semantic_membership:
            membership = self.fixed_membership[None].expand(batch, -1, -1)
            prior_kl = sensor_prefix.new_tensor(0.0)
        elif self.uninformed_assignment:
            membership = torch.softmax(
                self.channel_group_logits, dim=-1
            )[None].expand(batch, -1, -1)
            prior_kl = sensor_prefix.new_tensor(0.0)
        else:
            membership = torch.softmax(
                self.behaviour_router(self._behaviour_features(sensor_prefix)),
                dim=-1,
            )
            prior = self.log_group_prior.exp()[None, None]
            prior_kl = (
                membership
                * (
                    membership.clamp_min(1e-8).log()
                    - prior.clamp_min(1e-8).log()
                )
            ).sum(dim=-1).mean()
        process_weights = self.process_weights()
        diagnostics = {
            'sensor_group_prior_kl': prior_kl,
            'sensor_group_entropy': -(
                membership * membership.clamp_min(1e-8).log()
            ).sum(dim=-1).mean(),
        }
        for group, name in enumerate(self.group_names):
            diagnostics[f'sensor_group_mass_{name}'] = (
                membership[:, :, group].mean()
            )
            for process in range(process_weights.shape[1]):
                diagnostics[f'sensor_group_{name}_process_{process}'] = (
                    process_weights[group, process]
                )
        return membership, process_weights, diagnostics

    @torch.no_grad()
    def project(self) -> None:
        self.raw_process_adjustment.clamp_(-4.0, 4.0)


class CausalStructuralEncoder(nn.Module):
    """Explicit X -> Z dynamical encoder with an identity-safe initialization."""

    def __init__(self, latent_dim: int = 8, hidden: int = 32) -> None:
        super().__init__()
        if latent_dim < 2:
            raise ValueError("latent_dim must be at least two.")
        self.latent_dim = int(latent_dim)
        self.input_projection = nn.Linear(3, self.latent_dim, bias=False)
        self.temporal_in = nn.Conv1d(self.latent_dim, hidden, kernel_size=5)
        self.temporal_out = nn.Conv1d(hidden, self.latent_dim, kernel_size=1)
        self.decoder = nn.Linear(self.latent_dim, 1, bias=False)
        self.residual_gain = nn.Parameter(torch.tensor(-2.0))
        with torch.no_grad():
            self.input_projection.weight.zero_()
            self.input_projection.weight[0, 0] = 1.0
            nn.init.kaiming_uniform_(self.temporal_in.weight, a=math.sqrt(5.0))
            if self.temporal_in.bias is not None:
                self.temporal_in.bias.zero_()
            self.temporal_out.weight.zero_()
            self.temporal_out.bias.zero_()
            self.decoder.weight.zero_()
            self.decoder.weight[0, 0] = 1.0

    @staticmethod
    def _features(values: torch.Tensor) -> torch.Tensor:
        first = F.pad(values[:, 1:] - values[:, :-1], (0, 0, 1, 0))
        second = F.pad(first[:, 1:] - first[:, :-1], (0, 0, 1, 0))
        return torch.stack([values, first, second], dim=-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        base = self.input_projection(self._features(values))
        temporal = base.permute(0, 2, 3, 1).reshape(
            batch * channels, self.latent_dim, length
        )
        temporal = F.pad(temporal, (4, 0))
        temporal = self.temporal_out(F.gelu(self.temporal_in(temporal)))
        temporal = temporal.reshape(
            batch, channels, self.latent_dim, length
        ).permute(0, 3, 1, 2)
        gain = 0.25 * torch.sigmoid(self.residual_gain)
        return base + gain * torch.tanh(temporal)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent).squeeze(-1)

    @torch.no_grad()
    def project(self) -> None:
        self.residual_gain.clamp_(-8.0, 4.0)
        self.decoder.weight.clamp_(-4.0, 4.0)

class MultiStochasticLatentSeparator(nn.Module):
    """Sparse variational decomposition of an explicit structural latent Z."""

    def __init__(
        self,
        dataset: str,
        *,
        channels: int,
        seq_len: int,
        pred_len: int,
        latent_dim: int,
        inducing_points: int = 16,
        jitter: float = 1e-4,
        neutral_soft_assignment: bool = False,
        uninformed_assignment: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = str(dataset)
        self.channels = int(channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.latent_dim = int(latent_dim)
        self.jitter = float(jitter)
        specs = PROCESS_SPECS[domain_for_dataset(dataset)]
        self.process_names = tuple(spec[0] for spec in specs)
        day = samples_per_day(dataset)
        self.kernels = nn.ModuleList(
            [
                SparseStochasticKernel(
                    name,
                    family,
                    seq_len=seq_len,
                    samples_day=day,
                    nominal_period_days=period_days,
                    nominal_length_days=length_days,
                    inducing_points=inducing_points,
                    initial_amplitude=1.0,
                )
                for name, family, period_days, length_days in specs
            ]
        )
        self.sensor_geometry = SensorProcessGeometry(
            dataset,
            channels,
            len(self.kernels),
            neutral_soft_assignment=neutral_soft_assignment,
            uninformed_assignment=uninformed_assignment,
        )
        self.raw_noise_std = nn.Parameter(torch.tensor(inverse_softplus(0.10)))

    @property
    def num_processes(self) -> int:
        return len(self.kernels)

    def _sparse_terms(self, values: torch.Tensor):
        past = torch.arange(
            self.seq_len, device=values.device, dtype=values.dtype
        )
        future = torch.arange(
            self.seq_len,
            self.seq_len + self.pred_len,
            device=values.device,
            dtype=values.dtype,
        )
        sparse = [
            kernel.sparse_covariances(past, future, self.jitter)
            for kernel in self.kernels
        ]
        return (
            sparse,
            torch.stack([item["past"] for item in sparse]),
            torch.stack([item["past_diagonal"] for item in sparse]),
            torch.stack([item["future_past"] for item in sparse]),
            torch.stack([item["future_diagonal"] for item in sparse]),
        )

    def _group_posterior(
        self,
        rhs: torch.Tensor,
        weights: torch.Tensor,
        past_cov: torch.Tensor,
        past_diag: torch.Tensor,
        cross: torch.Tensor,
        future_diag: torch.Tensor,
        noise_std: torch.Tensor,
        batch: int,
        channels: int,
    ) -> dict[str, torch.Tensor]:
        component_past = weights[:, None, None] * past_cov
        component_cross = weights[:, None, None] * cross
        component_diag = weights[:, None] * future_diag
        identity = torch.eye(
            self.seq_len, device=rhs.device, dtype=rhs.dtype
        )
        observation = component_past.sum(dim=0) + (
            noise_std.square() + self.jitter
        ) * identity
        cholesky = torch.linalg.cholesky(observation)
        alpha = torch.cholesky_solve(rhs, cholesky)

        latent = torch.einsum("qlm,mn->qln", component_past, alpha)
        latent = latent.reshape(
            self.num_processes,
            self.seq_len,
            batch,
            channels,
            self.latent_dim,
        ).permute(2, 1, 0, 3, 4)

        past_variance = []
        process_future_variance = []
        for process in range(self.num_processes):
            covariance = component_past[process]
            solved = torch.cholesky_solve(covariance, cholesky)
            reduction = (covariance * solved.transpose(0, 1)).sum(dim=1)
            past_variance.append(
                (weights[process] * past_diag[process] - reduction).clamp_min(1e-8)
            )
            process_cross = component_cross[process]
            solved_cross = torch.cholesky_solve(
                process_cross.transpose(0, 1), cholesky
            )
            reduction = (
                process_cross * solved_cross.transpose(0, 1)
            ).sum(dim=1)
            process_future_variance.append(
                (component_diag[process] - reduction).clamp_min(1e-8)
            )
        past_variance = torch.stack(past_variance, dim=-1)
        process_future_variance = torch.stack(process_future_variance, dim=-1)

        total_cross = component_cross.sum(dim=0)
        future_mean = (total_cross @ alpha).reshape(
            self.pred_len, batch, channels, self.latent_dim
        ).permute(1, 0, 2, 3)
        total_solved = torch.cholesky_solve(
            total_cross.transpose(0, 1), cholesky
        )
        future_reduction = (
            total_cross * total_solved.transpose(0, 1)
        ).sum(dim=1)
        future_variance = (
            component_diag.sum(dim=0) - future_reduction
        ).clamp_min(1e-8)

        quadratic = (rhs * alpha).sum(dim=0).reshape(
            batch, channels, self.latent_dim
        ).mean(dim=-1) / self.seq_len
        log_det = 2.0 * cholesky.diagonal().log().sum() / self.seq_len
        nll = 0.5 * (quadratic + log_det + math.log(2.0 * math.pi))
        return {
            "latent": latent,
            "past_variance": past_variance,
            "process_future_variance": process_future_variance,
            "future_mean": future_mean,
            "future_variance": future_variance,
            "nll": nll,
        }

    def forward(
        self,
        structural_latent: torch.Tensor,
        sensor_prefix: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, length, channels, latent_dim = structural_latent.shape
        if (length, channels, latent_dim) != (
            self.seq_len,
            self.channels,
            self.latent_dim,
        ):
            raise ValueError(
                "Unexpected structural latent shape: "
                f"{tuple(structural_latent.shape)}."
            )
        membership, group_weights, geometry_diagnostics = (
            self.sensor_geometry(sensor_prefix)
        )
        sparse, past_cov, past_diag, cross, future_diag = self._sparse_terms(
            structural_latent
        )
        noise_std = F.softplus(self.raw_noise_std).clamp_min(1e-4)
        rhs = structural_latent.permute(1, 0, 2, 3).reshape(
            self.seq_len, batch * channels * latent_dim
        )
        latent = structural_latent.new_zeros(
            batch,
            self.seq_len,
            self.num_processes,
            channels,
            self.latent_dim,
        )
        past_second = torch.zeros_like(latent)
        future_mean = structural_latent.new_zeros(
            batch, self.pred_len, channels, self.latent_dim
        )
        future_second = torch.zeros_like(future_mean)
        process_future_variance_summary = structural_latent.new_zeros(
            self.num_processes
        )
        group_nll = []
        # Stream sensor groups. Materializing every group's
        # B x H x Q x C x D posterior is prohibitive on Traffic; only total
        # future moments and per-process variance summaries are needed.
        for group in range(self.sensor_geometry.num_groups):
            item = self._group_posterior(
                rhs,
                group_weights[group],
                past_cov,
                past_diag,
                cross,
                future_diag,
                noise_std,
                batch,
                channels,
            )
            channel_weight = membership[:, :, group]
            latent_weight = channel_weight[:, None, None, :, None]
            latent = latent + latent_weight * item["latent"]
            past_second = past_second + latent_weight * (
                item["latent"].square()
                + item["past_variance"][None, :, :, None, None]
            )
            future_weight = channel_weight[:, None, :, None]
            future_mean = future_mean + future_weight * item["future_mean"]
            future_second = future_second + future_weight * (
                item["future_mean"].square()
                + item["future_variance"][None, :, None, None]
            )
            process_future_variance_summary = (
                process_future_variance_summary
                + channel_weight.mean()
                * item["process_future_variance"].mean(dim=0)
            )
            group_nll.append(item["nll"])

        posterior_past_variance = (past_second - latent.square()).clamp_min(1e-8)
        future_variance = (future_second - future_mean.square()).clamp_min(1e-8)

        reconstruction = latent.sum(dim=2)
        reconstruction_mse = (reconstruction - structural_latent).square().mean()
        signal_power = structural_latent.square().mean().clamp_min(1e-8)
        reconstruction_ratio = reconstruction_mse / signal_power
        group_nll = torch.stack(group_nll, dim=-1)
        gaussian_nll = (membership * group_nll).sum(dim=-1).mean()
        trace_gaps = torch.stack([item["trace_gap"] for item in sparse])
        group_trace_gap = group_weights @ trace_gaps
        group_occupancy = membership.mean(dim=(0, 1))
        trace_correction = (
            group_occupancy * group_trace_gap
        ).sum() / (
            2.0 * noise_std.square().clamp_min(1e-8) * self.seq_len
        )
        representation_objective = (
            gaussian_nll
            + trace_correction
            + 0.01 * geometry_diagnostics["sensor_group_prior_kl"]
        )

        process_energy = latent.square().mean(dim=(1, 3, 4))
        energy_probability = process_energy / process_energy.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        process_entropy = -(
            energy_probability * energy_probability.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        inducing = torch.stack(
            [item["inducing_times"] for item in sparse], dim=0
        )
        diagnostics = {
            **geometry_diagnostics,
            "latent_reconstruction_mse": reconstruction_mse,
            "latent_reconstruction_ratio": reconstruction_ratio,
            "representation_gaussian_nll": gaussian_nll,
            "representation_trace_correction": trace_correction,
            "representation_objective": representation_objective,
            "posterior_past_variance": posterior_past_variance.mean(),
            "posterior_future_variance": future_variance.mean(),
            "process_energy_entropy": process_entropy,
            "inducing_minimum_gap": (
                inducing[:, 1:] - inducing[:, :-1]
            ).min(),
            "observation_noise_std": noise_std,
        }
        for index, name in enumerate(self.process_names):
            diagnostics[f"process_energy_{name}"] = (
                energy_probability[:, index].mean()
            )
            diagnostics[f"process_future_variance_{name}"] = (
                process_future_variance_summary[index]
            )
        return {
            "latent_mean": latent,
            "reconstruction": reconstruction,
            "posterior_past_variance": posterior_past_variance,
            "future_variance": future_variance,
            "sensor_group_membership": membership,
            "group_process_weights": group_weights,
            "inducing_times": inducing,
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def project(self) -> None:
        self.raw_noise_std.clamp_(-8.0, 4.0)
        for kernel in self.kernels:
            kernel.project()
        self.sensor_geometry.project()


class CrossProcessExchange(nn.Module):
    """Bounded low-rank exchange while retaining the complete process axis."""

    def __init__(self, width: int, processes: int, rank: int = 24) -> None:
        super().__init__()
        self.width = int(width)
        self.processes = int(processes)
        rank = min(int(rank), self.width)
        self.in_projection = nn.Linear(self.width, rank, bias=False)
        self.out_projection = nn.Linear(rank, self.width, bias=False)
        self.router = nn.Sequential(
            nn.LayerNorm(2 * self.processes + 1),
            nn.Linear(2 * self.processes + 1, 32),
            nn.GELU(),
            nn.Linear(32, self.processes * self.processes),
        )
        self.raw_gate = nn.Parameter(torch.full((self.processes,), -4.0))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.out_projection.weight)

    def forward(
        self,
        field: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, processes, width, channels, patches = field.shape
        if (processes, width) != (self.processes, self.width):
            raise ValueError("Cross-process field shape mismatch.")
        logits = self.router(context).reshape(batch, processes, processes)
        diagonal = torch.eye(processes, device=field.device, dtype=torch.bool)[None]
        coupling = torch.softmax(logits.masked_fill(diagonal, -1e4), dim=-1)

        values = field.permute(0, 1, 3, 4, 2)
        projected = self.out_projection(self.in_projection(values))
        projected = projected.permute(0, 1, 4, 2, 3)
        message = torch.einsum("bqr,brdcp->bqdcp", coupling, projected)
        field_rms = (
            field.square().mean(dim=(2, 3, 4), keepdim=True) + 1e-12
        ).sqrt()
        message_rms = (
            message.square().mean(dim=(2, 3, 4), keepdim=True) + 1e-12
        ).sqrt()
        # Smoothly cap the message RMS by the field RMS. Near zero this is
        # locally the identity map; unlike direct RMS division it cannot create
        # a large first-step gradient from a zero-initialized output projection.
        normalized = message * field_rms / (field_rms + message_rms)
        gate = torch.sigmoid(self.raw_gate)[None, :, None, None, None]
        correction = gate * normalized
        # Epsilon-stabilized RMS normalization is differentiable at the
        # zero-correction initialization.
        output = field + correction
        effective_ratio = (
            (correction.square().mean() + 1e-12).sqrt()
            / (field.square().mean() + 1e-12).sqrt()
        )
        effective_matrix = self.out_projection.weight @ self.in_projection.weight
        # The Frobenius norm is a smooth upper bound on the spectral norm and
        # therefore a conservative stability constraint. The exact norm is
        # detached because its SVD derivative is undefined at the zero matrix.
        spectral_norm = (effective_matrix.square().sum() + 1e-12).sqrt()
        exact_spectral_norm = torch.linalg.matrix_norm(
            effective_matrix.detach(), ord=2
        )
        off_diagonal_entropy = -(
            coupling * coupling.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        return output, {
            "cross_effective_ratio": effective_ratio,
            "cross_spectral_norm": spectral_norm,
            "cross_exact_spectral_norm": exact_spectral_norm,
            "cross_gate": gate.mean(),
            "cross_coupling_entropy": off_diagonal_entropy,
        }

    @torch.no_grad()
    def project(self, maximum_spectral_norm: float = 1.0) -> None:
        effective = self.out_projection.weight @ self.in_projection.weight
        norm = torch.linalg.matrix_norm(effective, ord=2).clamp_min(1e-8)
        if norm > maximum_spectral_norm:
            self.out_projection.weight.mul_(maximum_spectral_norm / norm)
        self.raw_gate.clamp_(-8.0, 0.0)


class ProcessPreservingMVPFContinuation(nn.Module):
    """MVPF continuation that preserves Q through every field block."""

    def __init__(
        self,
        base: AdaWarpMVPFPlusForecaster,
        *,
        num_processes: int,
        latent_dim: int,
        fuse_before_field: bool = False,
        disable_cross_exchange: bool = True,
    ) -> None:
        super().__init__()
        self.mvpff = base
        self.num_processes = int(num_processes)
        self.latent_dim = int(latent_dim)
        self.fuse_before_field = bool(fuse_before_field)
        self.disable_cross_exchange = bool(disable_cross_exchange)
        self.process_router = nn.Sequential(
            nn.LayerNorm(2 * self.num_processes + 1),
            nn.Linear(2 * self.num_processes + 1, 32),
            nn.GELU(),
            nn.Linear(32, self.num_processes),
        )
        nn.init.zeros_(self.process_router[-1].weight)
        nn.init.zeros_(self.process_router[-1].bias)

        self.patch_adapters = nn.ModuleList()
        self.exchange_blocks = nn.ModuleList()
        self.film_scales = nn.ParameterList()
        self.film_shifts = nn.ParameterList()
        for branch in self.mvpff.backbone.branches:
            adapter = nn.Linear(
                branch.patch_len * self.latent_dim,
                branch.patch_len,
                bias=False,
            )
            with torch.no_grad():
                adapter.weight.zero_()
                for position in range(branch.patch_len):
                    adapter.weight[position, position * self.latent_dim] = 1.0
            self.patch_adapters.append(adapter)
            width = branch.encoder_norm.normalized_shape[0]
            depth = len(branch.blocks)
            # Exchange is a legacy experimental option. The finalized model
            # does not allocate these modules, reducing parameters and field
            # compute while preserving each process until late decoding.
            self.exchange_blocks.append(
                nn.ModuleList()
                if self.disable_cross_exchange
                else nn.ModuleList(
                    [CrossProcessExchange(width, self.num_processes) for _ in range(depth)]
                )
            )
            self.film_scales.append(
                nn.Parameter(torch.zeros(depth, self.num_processes, width))
            )
            self.film_shifts.append(
                nn.Parameter(torch.zeros(depth, self.num_processes, width))
            )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _weights(
        self,
        latent: torch.Tensor,
        posterior_past_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = latent.shape[0]
        energy = torch.log1p(latent.square().mean(dim=(1, 3, 4)))
        uncertainty = torch.log1p(
            posterior_past_variance.mean(dim=(1, 3, 4))
        )
        horizon = latent.new_full(
            (batch, 1),
            math.log1p(self.mvpff.pred_len) / math.log(721.0),
        )
        context = torch.cat([energy, uncertainty, horizon], dim=-1)
        return torch.softmax(self.process_router(context), dim=-1), context

    def _patchify_latent(self, latent: torch.Tensor, branch: nn.Module) -> torch.Tensor:
        batch, length, processes, channels, latent_dim = latent.shape
        target = branch.input_patches * branch.patch_len
        if length < target:
            latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, target - length))
        elif length > target:
            latent = latent[:, :target]
        patches = latent.reshape(
            batch,
            branch.input_patches,
            branch.patch_len,
            processes,
            channels,
            latent_dim,
        ).permute(0, 3, 4, 1, 2, 5)
        return patches.reshape(
            batch,
            processes,
            channels,
            branch.input_patches,
            branch.patch_len * latent_dim,
        )

    def _process_branch(
        self,
        branch_index: int,
        branch: nn.Module,
        latent: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        batch, _, processes, channels, _ = latent.shape
        patches = self._patchify_latent(latent, branch)
        adapted = self.patch_adapters[branch_index](patches)
        encoded = branch._encode(adapted)
        width = encoded.shape[-1]
        field = encoded.permute(0, 1, 4, 2, 3).contiguous()
        flattened = field.reshape(
            batch * processes, width, channels, branch.input_patches
        )
        flattened = branch.memory(flattened)
        field = flattened.reshape(
            batch, processes, width, channels, branch.input_patches
        )

        diagnostics = []
        for layer, (frequency_gate, field_block) in enumerate(
            zip(branch.freq_gates, branch.blocks)
        ):
            flattened = field.reshape(
                batch * processes, width, channels, branch.input_patches
            )
            flattened = field_block(frequency_gate(flattened))
            field = flattened.reshape(
                batch, processes, width, channels, branch.input_patches
            )
            scale = 0.10 * torch.tanh(self.film_scales[branch_index][layer])
            shift = 0.10 * torch.tanh(self.film_shifts[branch_index][layer])
            field = field * (1.0 + scale[None, :, :, None, None])
            field = field + shift[None, :, :, None, None]
            if self.disable_cross_exchange:
                zero = field.new_tensor(0.0)
                one = field.new_tensor(1.0)
                layer_diagnostics = {
                    "cross_effective_ratio": zero,
                    "cross_spectral_norm": zero,
                    "cross_exact_spectral_norm": zero,
                    "cross_gate": zero,
                    "cross_coupling_entropy": one,
                }
            else:
                exchange = self.exchange_blocks[branch_index][layer]
                field, layer_diagnostics = exchange(field, context)
            diagnostics.append(layer_diagnostics)

        future = branch.patch_predictor(
            field.reshape(batch * processes, width, channels, branch.input_patches)
        ).reshape(
            batch, processes, width, channels, branch.output_patches
        )
        future_embeddings = future.permute(0, 1, 3, 4, 2).reshape(
            batch * processes, channels, branch.output_patches, width
        )
        decoded = branch._decode(future_embeddings, branch.pred_len)
        decoded = decoded.reshape(batch, processes, branch.pred_len, channels)
        return decoded, diagnostics

    def forward(
        self,
        latent: torch.Tensor,
        posterior_past_variance: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if latent.ndim != 5 or latent.shape[2] != self.num_processes:
            raise ValueError("The complete process axis must enter MVPF continuation.")
        backbone = self.mvpff.backbone
        trend, residual = backbone._decompose(reconstructed)
        if backbone.use_trend_decomposition:
            trend_forecast = backbone._linear_per_channel(backbone.trend_head, trend)
        else:
            trend_forecast = reconstructed.new_zeros(
                reconstructed.shape[0], self.mvpff.pred_len, reconstructed.shape[2]
            )
        direct = backbone._linear_per_channel(backbone.direct_residual_head, residual)
        process_weights, context = self._weights(latent, posterior_past_variance)
        if self.fuse_before_field:
            fused = (latent * process_weights[:, None, :, None, None]).sum(dim=2, keepdim=True)
            latent = fused.expand(-1, -1, self.num_processes, -1, -1)
            process_weights = process_weights.new_full(
                process_weights.shape, 1.0 / self.num_processes
            )

        process_branch_forecasts = []
        exchange_diagnostics = []
        for index, branch in enumerate(backbone.branches):
            process_forecast, branch_diagnostics = self._process_branch(
                index, branch, latent, context
            )
            process_branch_forecasts.append(process_forecast)
            exchange_diagnostics.extend(branch_diagnostics)
        # This is the first process reduction in the field path. It occurs only
        # after every process has reached and been decoded from the future field.
        branch_forecasts = [
            (forecast * process_weights[:, :, None, None]).sum(dim=1)
            for forecast in process_branch_forecasts
        ]

        components = torch.stack([direct, *branch_forecasts], dim=1)
        if backbone.use_component_gate:
            component_weights = torch.softmax(
                backbone.component_gate(
                    backbone._summary_features(reconstructed, residual)
                ),
                dim=-1,
            )
        else:
            component_weights = components.new_full(
                (components.shape[0], components.shape[1]),
                1.0 / components.shape[1],
            )
        field_forecast = trend_forecast + torch.sigmoid(
            backbone.residual_strength
        ) * (components * component_weights[:, :, None, None]).sum(dim=1)

        if self.mvpff.use_linear_field:
            linear, linear_weights, fit_error = self.mvpff.linear_field(reconstructed)
            confidence = linear_weights.max(dim=1, keepdim=True).values
        else:
            linear = field_forecast
            fit_error = field_forecast.new_zeros(field_forecast.shape[0], 1)
            confidence = field_forecast.new_zeros(field_forecast.shape[0], 1)
        outer = torch.softmax(
            self.mvpff.mean_mixer(
                self.mvpff._summary_features(
                    reconstructed, residual, confidence, fit_error
                )
            ),
            dim=-1,
        )
        forecast = (
            outer[:, 0, None, None] * field_forecast
            + outer[:, 1, None, None] * linear
        )

        entropy = -(
            process_weights * process_weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        ratios = torch.stack(
            [entry["cross_effective_ratio"] for entry in exchange_diagnostics]
        )
        spectra = torch.stack(
            [entry["cross_spectral_norm"] for entry in exchange_diagnostics]
        )
        exact_spectra = torch.stack(
            [entry["cross_exact_spectral_norm"] for entry in exchange_diagnostics]
        )
        gates = torch.stack(
            [entry["cross_gate"] for entry in exchange_diagnostics]
        )
        self.last_diagnostics = {
            "continuation_process_entropy": entropy,
            "continuation_process_information_bits": (
                math.log(self.num_processes) - entropy
            ) / math.log(2.0),
            "continuation_field_weight": outer[:, 0].mean(),
            "continuation_linear_weight": outer[:, 1].mean(),
            "cross_effective_ratio": ratios.mean(),
            "cross_effective_ratio_max": ratios.max(),
            "cross_spectral_norm": spectra.max(),
            "cross_exact_spectral_norm": exact_spectra.max(),
            "cross_gate": gates.mean(),
            "process_axis_size_at_field_exit": forecast.new_tensor(
                float(self.num_processes)
            ),
            "process_fusion_after_future_decode": forecast.new_tensor(1.0),
        }
        for index in range(self.num_processes):
            self.last_diagnostics[f"continuation_process_weight_{index}"] = (
                process_weights[:, index].mean()
            )
        return forecast, self.last_diagnostics

    @torch.no_grad()
    def project(self) -> None:
        for exchanges in self.exchange_blocks:
            for exchange in exchanges:
                exchange.project(1.0)
        for scale, shift in zip(self.film_scales, self.film_shifts):
            scale.clamp_(-4.0, 4.0)
            shift.clamp_(-4.0, 4.0)

class StudentTDistributionHead(nn.Module):
    """Student-t mixture scales with explicit latent-posterior variance input."""

    def __init__(self, pred_len: int, channels: int, components: int) -> None:
        super().__init__()
        self.components = int(components)
        initial = inverse_softplus(0.50)
        self.horizon_scale = nn.Parameter(torch.full((components, pred_len, 1), initial))
        self.channel_scale = nn.Parameter(torch.zeros(components, 1, channels))
        self.context_trunk = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, 32), nn.GELU())
        self.context_scale = nn.Linear(32, components)
        self.router = nn.Linear(32, components)
        self.raw_df = nn.Parameter(torch.full((components,), -2.0))
        self.raw_latent_variance_gain = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.context_scale.weight)
        nn.init.zeros_(self.context_scale.bias)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    @staticmethod
    def features(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=1, keepdim=True)
        scale = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = centered / scale
        slope = (normalized[:, -1] - normalized[:, 0]).mean(dim=1, keepdim=True)
        volatility = normalized.std(dim=(1, 2), unbiased=False).unsqueeze(1)
        roughness = (normalized[:, 1:] - normalized[:, :-1]).abs().mean(dim=(1, 2)).unsqueeze(1)
        level = normalized[:, -1].abs().mean(dim=1, keepdim=True)
        maximum = normalized.abs().amax(dim=(1, 2)).unsqueeze(1)
        return torch.cat([slope, volatility, roughness, level, maximum], dim=1)

    def forward(
        self,
        values: torch.Tensor,
        latent_future_variance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_trunk(self.features(values))
        context_scale = self.context_scale(context)[:, :, None, None]
        raw_scale = self.horizon_scale[None] + self.channel_scale[None] + context_scale
        scale_variance = F.softplus(raw_scale).clamp_min(1e-4).square()
        if latent_future_variance is not None and self.components > 1:
            gain = F.softplus(self.raw_latent_variance_gain)
            scale_variance = scale_variance.clone()
            scale_variance[:, 1] = scale_variance[:, 1] + gain * latent_future_variance
        scales = scale_variance.clamp_min(1e-8).sqrt()
        degrees = 2.1 + 27.9 * torch.sigmoid(self.raw_df)
        weights = torch.softmax(self.router(context), dim=-1)
        return weights, scales, degrees

    @torch.no_grad()
    def copy_from_single_component(self, reference: "StudentTDistributionHead") -> None:
        if reference.components != 1:
            raise ValueError("Reference head must have one component.")
        self.horizon_scale.copy_(reference.horizon_scale.expand_as(self.horizon_scale))
        self.channel_scale.copy_(reference.channel_scale.expand_as(self.channel_scale))
        self.context_trunk.load_state_dict(reference.context_trunk.state_dict())
        self.context_scale.weight.copy_(reference.context_scale.weight.expand_as(self.context_scale.weight))
        self.context_scale.bias.copy_(reference.context_scale.bias.expand_as(self.context_scale.bias))
        self.raw_df.copy_(reference.raw_df.expand_as(self.raw_df))
        self.router.weight.zero_()
        self.router.bias.copy_(self.router.bias.new_tensor([2.0, -2.0]))

    @torch.no_grad()
    def project(self) -> None:
        self.horizon_scale.clamp_(-10.0, 8.0)
        self.channel_scale.clamp_(-8.0, 8.0)
        self.raw_df.clamp_(-8.0, 8.0)
        self.raw_latent_variance_gain.clamp_(-8.0, 5.0)


class LatentFlow(nn.Module):
    """Explicit latent encoding, stochastic separation, and preserved MVPF fields."""

    CONSTRAINT_NAMES = (
        "latent_fidelity",
        "point_regret",
        "variance_adequacy",
        "cross_effective_size",
        "cross_spectral_norm",
        "soft_miscoverage",
    )

    def __init__(
        self,
        base: AdaWarpMVPFPlusForecaster,
        *,
        dataset: str,
        channels: int,
        inducing_points: int = 16,
        latent_dim: int = 8,
        horizon_groups: int = 4,
        reconstruction_tolerance: float = 0.10,
        regret_tolerance: float = 0.002,
        variance_tolerance: float = 0.15,
        cross_tolerance: float = 0.15,
        spectral_tolerance: float = 1.0,
        miscoverage_tolerance: float = 0.03,
        ablation_mode: str = "final",
        continuation_unfreeze_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.reference_base = copy.deepcopy(base)
        self.reference_base.eval()
        for parameter in self.reference_base.parameters():
            parameter.requires_grad_(False)
        self.ablation_mode = str(ablation_mode)
        self.continuation_unfreeze_blocks = int(continuation_unfreeze_blocks)
        if self.continuation_unfreeze_blocks not in {-1, 0, 1, 2}:
            raise ValueError("continuation_unfreeze_blocks must be -1 (all), 0, 1, or 2")
        # The finalized architecture was selected without primal-dual updates.
        self.primal_dual_enabled = False
        # Evaluation-only intervention; None preserves the exact final model.
        self.disabled_process: int | None = None
        self.structural_only = self.ablation_mode == "x_to_z"
        fuse_before_field = self.ablation_mode in {
            "x_to_z", "single_stochastic", "multi_stochastic", "learned_inducing_elbo"
        }
        # Cross-process exchange was removed from the finalized architecture:
        # its measured contribution was negligible and inconsistent by seed.
        # The old ``full`` mode remains opt-in only for historical diagnostics.
        disable_cross_exchange = self.ablation_mode != "full"
        self.encoder = CausalStructuralEncoder(latent_dim=latent_dim)
        self.separator = MultiStochasticLatentSeparator(
            dataset,
            channels=channels,
            seq_len=base.seq_len,
            pred_len=base.pred_len,
            latent_dim=latent_dim,
            inducing_points=inducing_points,
            neutral_soft_assignment=self.ablation_mode in {
                "learned_soft_geometry"
            },
            uninformed_assignment=self.ablation_mode == "no_sensor_group_priors",
        )
        # Control-only deterministic separator. It is instantiated only for the
        # matched non-GP multibranch experiment and cannot affect the released
        # ``final`` architecture or its state dictionary.
        self.deterministic_process_filters = nn.ModuleList()
        self.register_parameter("deterministic_process_gain", None)
        if self.ablation_mode == "deterministic_multibranch":
            kernel_sizes = (3, 5, 9, 15, 23, 31)
            for process in range(self.separator.num_processes):
                width = kernel_sizes[process]
                convolution = nn.Conv1d(
                    latent_dim, latent_dim, kernel_size=width,
                    padding=width // 2, groups=latent_dim, bias=False,
                )
                nn.init.constant_(convolution.weight, 1.0 / width)
                self.deterministic_process_filters.append(convolution)
            self.deterministic_process_gain = nn.Parameter(
                torch.full((self.separator.num_processes,), 0.10)
            )
        self.continuation = ProcessPreservingMVPFContinuation(
            base,
            num_processes=self.separator.num_processes,
            latent_dim=latent_dim,
            fuse_before_field=fuse_before_field,
            disable_cross_exchange=disable_cross_exchange,
        )
        self.reference_head = StudentTDistributionHead(
            base.pred_len, channels, components=1
        )
        self.mixture_head = StudentTDistributionHead(
            base.pred_len, channels, components=2
        )
        self.dataset = str(dataset)
        self.channels = int(channels)
        self.seq_len = int(base.seq_len)
        self.pred_len = int(base.pred_len)
        self.latent_dim = int(latent_dim)
        self.horizon_groups = min(int(horizon_groups), self.pred_len)
        self.reconstruction_tolerance = float(reconstruction_tolerance)
        self.regret_tolerance = float(regret_tolerance)
        self.variance_tolerance = float(variance_tolerance)
        self.cross_tolerance = float(cross_tolerance)
        self.spectral_tolerance = float(spectral_tolerance)
        self.miscoverage_tolerance = float(miscoverage_tolerance)
        self.register_buffer(
            "constraint_duals", torch.zeros(len(self.CONSTRAINT_NAMES))
        )
        self.register_buffer(
            "robust_weights",
            torch.full((self.horizon_groups,), 1.0 / self.horizon_groups),
        )
        self.register_buffer("regret_ema", torch.zeros(self.horizon_groups))
        self._stage = "A"
        self.reference_only = False
        self._apply_ablation_configuration()
        self.configure_stage("A")
    _ABLATION_MODES = {
        "full",
        "x_to_z",
        "single_stochastic",
        "multi_stochastic",
        "learned_inducing_elbo",
        "process_preserving",
        "fixed_inducing",
        "random_process_families",
        "permuted_process_families",
        "no_sensor_priors",
        "no_sensor_group_priors",
        "neutral_process_scales",
        "deterministic_multibranch",
        "learned_soft_geometry",
        "no_primal_dual",
        "final",
    }

    def _apply_ablation_configuration(self) -> None:
        if self.ablation_mode not in self._ABLATION_MODES:
            raise ValueError(f"Unknown LatentFlow ablation mode: {self.ablation_mode}")
        if self.ablation_mode in {"single_stochastic", "multi_stochastic", "fixed_inducing"}:
            with torch.no_grad():
                for kernel in self.separator.kernels:
                    kernel.raw_inducing_gaps.zero_()
        if self.ablation_mode == "single_stochastic":
            with torch.no_grad():
                prior = self.separator.sensor_geometry.log_process_prior
                prior.fill_(-20.0)
                prior[:, 0] = 0.0
                self.separator.sensor_geometry.raw_process_adjustment.zero_()
        elif self.ablation_mode in {
            "random_process_families", "permuted_process_families"
        }:
            self.separator.kernels = nn.ModuleList(
                list(reversed(list(self.separator.kernels)))
            )
        elif self.ablation_mode == "neutral_process_scales":
            # Dataset-agnostic initialization in sample-index units. Kernel
            # families and their count are unchanged; only domain-specific
            # daily/weekly/thermal starting scales are removed.
            with torch.no_grad():
                for kernel in self.separator.kernels:
                    kernel.raw_amplitude.fill_(inverse_softplus(1.0))
                    neutral_length = 1.0 if kernel.family == "periodic" else 12.0
                    kernel.raw_length.fill_(inverse_softplus(neutral_length))
                    if kernel.family == "periodic":
                        kernel.nominal_period = 24.0
                        kernel.raw_decay.fill_(inverse_softplus(48.0))
                        kernel.raw_period_adjustment.zero_()
        elif self.ablation_mode == "no_sensor_priors":
            with torch.no_grad():
                self.separator.sensor_geometry.log_process_prior.zero_()
                self.separator.sensor_geometry.log_group_prior.zero_()

    def _enforce_ablation_freezes(self) -> None:
        if self.ablation_mode in {"single_stochastic", "multi_stochastic", "fixed_inducing"}:
            for kernel in self.separator.kernels:
                kernel.raw_inducing_gaps.requires_grad_(False)
        if self.ablation_mode == "single_stochastic":
            self.separator.sensor_geometry.raw_process_adjustment.requires_grad_(False)
        if self.ablation_mode == "learned_soft_geometry":
            # Direct channel-to-process routing replaces this unused tensor.
            self.separator.sensor_geometry.raw_process_adjustment.requires_grad_(False)

    def _structural_posterior(
        self, structural_latent: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, _, channels, latent_dim = structural_latent.shape
        processes = self.separator.num_processes
        latent = structural_latent[:, :, None].expand(
            -1, -1, processes, -1, -1
        ) / processes
        epsilon = structural_latent.new_tensor(1e-8)
        membership = structural_latent.new_full(
            (batch, channels, self.separator.sensor_geometry.num_groups),
            1.0 / self.separator.sensor_geometry.num_groups,
        )
        group_weights = structural_latent.new_full(
            (self.separator.sensor_geometry.num_groups, processes),
            1.0 / processes,
        )
        inducing = torch.stack(
            [kernel.inducing_times() for kernel in self.separator.kernels], dim=0
        )
        diagnostics = {
            "sensor_group_prior_kl": epsilon,
            "sensor_group_entropy": epsilon,
            "latent_reconstruction_mse": epsilon,
            "latent_reconstruction_ratio": epsilon,
            "representation_gaussian_nll": epsilon,
            "representation_trace_correction": epsilon,
            "representation_objective": epsilon,
            "posterior_past_variance": epsilon,
            "posterior_future_variance": epsilon,
            "process_energy_entropy": structural_latent.new_tensor(math.log(processes)),
            "inducing_minimum_gap": (inducing[:, 1:] - inducing[:, :-1]).min(),
            "observation_noise_std": epsilon,
        }
        for name in self.separator.process_names:
            diagnostics[f"process_energy_{name}"] = structural_latent.new_tensor(
                1.0 / processes
            )
            diagnostics[f"process_future_variance_{name}"] = epsilon
        return {
            "latent_mean": latent,
            "reconstruction": structural_latent,
            "posterior_past_variance": torch.full_like(latent, 1e-8),
            "future_variance": structural_latent.new_full(
                (batch, self.pred_len, channels, latent_dim), 1e-8
            ),
            "sensor_group_membership": membership,
            "group_process_weights": group_weights,
            "inducing_times": inducing,
            "diagnostics": diagnostics,
        }
    def _deterministic_multibranch_posterior(
        self, structural_latent: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Exactly reconstructing non-GP process-axis control."""
        batch, length, channels, latent_dim = structural_latent.shape
        processes = self.separator.num_processes
        flat = structural_latent.permute(0, 2, 3, 1).reshape(
            batch * channels, latent_dim, length
        )
        filtered = []
        for convolution in self.deterministic_process_filters:
            item = convolution(flat).reshape(
                batch, channels, latent_dim, length
            ).permute(0, 3, 1, 2)
            filtered.append(item)
        filtered = torch.stack(filtered, dim=2)
        centered = filtered - filtered.mean(dim=2, keepdim=True)
        gain = 0.05 * torch.tanh(self.deterministic_process_gain)
        latent = structural_latent[:, :, None] / processes
        latent = latent + centered * gain[None, None, :, None, None]
        variance = torch.full_like(latent, 1e-8)
        epsilon = structural_latent.new_tensor(1e-8)
        membership = structural_latent.new_full(
            (batch, channels, self.separator.sensor_geometry.num_groups),
            1.0 / self.separator.sensor_geometry.num_groups,
        )
        group_weights = structural_latent.new_full(
            (self.separator.sensor_geometry.num_groups, processes),
            1.0 / processes,
        )
        inducing = torch.stack(
            [kernel.inducing_times() for kernel in self.separator.kernels], dim=0
        )
        reconstruction = latent.sum(dim=2)
        reconstruction_mse = (reconstruction - structural_latent).square().mean()
        energy = latent.square().mean(dim=(1, 3, 4))
        energy_probability = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        diagnostics = {
            "sensor_group_prior_kl": epsilon,
            "sensor_group_entropy": epsilon,
            "latent_reconstruction_mse": reconstruction_mse,
            "latent_reconstruction_ratio": reconstruction_mse
            / structural_latent.square().mean().clamp_min(1e-8),
            "representation_gaussian_nll": epsilon,
            "representation_trace_correction": epsilon,
            "representation_objective": epsilon,
            "posterior_past_variance": epsilon,
            "posterior_future_variance": epsilon,
            "process_energy_entropy": -(
                energy_probability * energy_probability.clamp_min(1e-8).log()
            ).sum(dim=-1).mean(),
            "inducing_minimum_gap": (inducing[:, 1:] - inducing[:, :-1]).min(),
            "observation_noise_std": epsilon,
            "deterministic_multibranch": structural_latent.new_tensor(1.0),
        }
        for index, name in enumerate(self.separator.process_names):
            diagnostics[f"process_energy_{name}"] = energy_probability[:, index].mean()
            diagnostics[f"process_future_variance_{name}"] = epsilon
        return {
            "latent_mean": latent,
            "reconstruction": reconstruction,
            "posterior_past_variance": variance,
            "future_variance": structural_latent.new_full(
                (batch, self.pred_len, channels, latent_dim), 1e-8
            ),
            "sensor_group_membership": membership,
            "group_process_weights": group_weights,
            "inducing_times": inducing,
            "diagnostics": diagnostics,
        }
    def reference_distribution(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.reference_base.eval()
        mean = self.reference_base(inputs, None, None, None)
        _, scales, degrees = self.reference_head(inputs)
        variance = degrees[0] / (degrees[0] - 2.0) * scales[:, 0].square()
        return {
            "mean": mean,
            "variance": variance,
            "second_moment": variance + mean.square(),
            "scale": scales[:, 0],
            "df": degrees[0],
        }

    def forward_distribution(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        reference = self.reference_distribution(inputs)
        if self.reference_only:
            return {
                **reference,
                "within_variance": reference["variance"],
                "between_variance": torch.zeros_like(reference["variance"]),
                "component_means": reference["mean"][:, None],
                "component_scales": reference["scale"][:, None],
                "component_df": reference["df"][None],
                "weights": inputs.new_ones(inputs.shape[0], 1),
                "base_mean": reference["mean"],
                "diagnostics": {"selected_reference_only": inputs.new_tensor(1.0)},
            }

        backbone = self.continuation.mvpff.backbone
        normalized, window_mean, window_std = backbone._normalize(inputs)
        structural_latent = self.encoder(normalized)
        if self.structural_only:
            posterior = self._structural_posterior(structural_latent)
        elif self.ablation_mode == "deterministic_multibranch":
            posterior = self._deterministic_multibranch_posterior(structural_latent)
        else:
            posterior = self.separator(structural_latent, inputs)
        if self.disabled_process is not None:
            process = int(self.disabled_process)
            if not 0 <= process < self.separator.num_processes:
                raise ValueError(f"Invalid process intervention index: {process}.")
            posterior["latent_mean"] = posterior["latent_mean"].clone()
            posterior["posterior_past_variance"] = posterior["posterior_past_variance"].clone()
            posterior["latent_mean"][:, :, process] = 0.0
            posterior["posterior_past_variance"][:, :, process] = 0.0
            posterior["reconstruction"] = posterior["latent_mean"].sum(dim=2)
            posterior["diagnostics"]["disabled_process"] = inputs.new_tensor(float(process))
        structural_reconstruction = self.encoder.decode(structural_latent)
        process_reconstruction = self.encoder.decode(posterior["reconstruction"])
        structural_ratio = (
            structural_reconstruction - normalized
        ).square().mean() / normalized.square().mean().clamp_min(1e-8)
        process_signal_ratio = (
            process_reconstruction - normalized
        ).square().mean() / normalized.square().mean().clamp_min(1e-8)
        posterior["diagnostics"]["structural_reconstruction_ratio"] = structural_ratio
        posterior["diagnostics"]["process_signal_reconstruction_ratio"] = process_signal_ratio
        posterior["diagnostics"]["representation_objective"] = (
            posterior["diagnostics"]["representation_objective"]
            + structural_ratio
            + process_signal_ratio
        )
        latent_forecast_normalized, continuation_diagnostics = self.continuation(
            posterior["latent_mean"],
            posterior["posterior_past_variance"],
            process_reconstruction,
        )
        latent_forecast = latent_forecast_normalized * window_std + window_mean
        component_means = torch.stack([reference["mean"], latent_forecast], dim=1)
        decoder_squared = self.encoder.decoder.weight[0].square()
        future_variance_normalized = torch.einsum(
            "bhcd,d->bhc", posterior["future_variance"], decoder_squared
        )
        future_variance_raw = future_variance_normalized * window_std.square()
        weights, scales, degrees = self.mixture_head(inputs, future_variance_raw)
        weights_bc = weights[:, :, None, None]
        mean = (weights_bc * component_means).sum(dim=1)
        component_variance = (
            degrees[None, :, None, None] / (degrees[None, :, None, None] - 2.0)
        ) * scales.square()
        within = (weights_bc * component_variance).sum(dim=1)
        between = (weights_bc * (component_means - mean[:, None]).square()).sum(dim=1)
        variance = (within + between).clamp_min(1e-8)
        second_moment = variance + mean.square()

        mixture_entropy = -(
            weights * weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        diagnostics = {
            **posterior["diagnostics"],
            **continuation_diagnostics,
            "mixture_weight_entropy": mixture_entropy,
            "mixture_routing_information_bits": (
                math.log(2.0) - mixture_entropy
            ) / math.log(2.0),
            "mixture_base_weight": weights[:, 0].mean(),
            "mixture_latent_weight": weights[:, 1].mean(),
            "predictive_mean": mean.mean(),
            "predictive_variance": variance.mean(),
            "predictive_second_moment": second_moment.mean(),
            "within_component_variance": within.mean(),
            "between_component_variance": between.mean(),
            "latent_variance_gain": F.softplus(
                self.mixture_head.raw_latent_variance_gain
            ),
            "selected_reference_only": inputs.new_tensor(0.0),
        }
        return {
            "mean": mean,
            "variance": variance,
            "second_moment": second_moment,
            "within_variance": within,
            "between_variance": between,
            "component_means": component_means,
            "component_scales": scales,
            "component_df": degrees,
            "weights": weights,
            "base_mean": reference["mean"],
            "latent_posterior": posterior,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def mixture_nll(target: torch.Tensor, distribution: dict[str, torch.Tensor]) -> torch.Tensor:
        log_prob = student_t_log_prob(
            target[:, None],
            distribution["component_means"],
            distribution["component_scales"],
            distribution["component_df"][None, :, None, None],
        )
        log_weights = distribution["weights"].clamp_min(1e-8).log()[:, :, None, None]
        return -torch.logsumexp(log_prob + log_weights, dim=1).mean()

    def reference_nll(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reference = self.reference_distribution(inputs)
        return -student_t_log_prob(
            target, reference["mean"], reference["scale"], reference["df"]
        ).mean()

    def _group_losses(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                (prediction.index_select(1, indices) - target.index_select(1, indices))
                .square()
                .mean()
                for indices in torch.tensor_split(
                    torch.arange(self.pred_len, device=prediction.device), self.horizon_groups
                )
            ]
        )

    def training_objective(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_dual: torch.Tensor | None,
        fenchel_weight: float,
        nll_weight: float,
        representation_weight: float,
        augmented_weight: float,
        reconstruction_weight: float,
    ):
        distribution = self.forward_distribution(inputs)
        group_mse = self._group_losses(distribution["mean"], target)
        base_group_mse = self._group_losses(distribution["base_mean"], target).detach()
        point_mse = (distribution["mean"] - target).square().mean()
        base_mse = (distribution["base_mean"] - target).square().mean().detach()
        robust_point = (self.robust_weights.detach() * group_mse).sum()
        nll = self.mixture_nll(target, distribution)

        residual = distribution["mean"] - target
        residual_bias = torch.stack(
            [
                residual.index_select(1, indices).mean(dim=1)
                for indices in torch.tensor_split(
                    torch.arange(self.pred_len, device=target.device), self.horizon_groups
                )
            ],
            dim=1,
        )
        if sample_dual is None or not self.primal_dual_enabled:
            fenchel_primal = point_mse.new_tensor(0.0)
            sample_dual_norm = point_mse.new_tensor(0.0)
        else:
            fenchel_primal = (sample_dual.detach() * residual_bias).mean()
            sample_dual_norm = sample_dual.detach().square().mean().sqrt()

        diagnostics = distribution["diagnostics"]
        reconstruction_ratio = torch.stack(
            [
                diagnostics["latent_reconstruction_ratio"],
                diagnostics["structural_reconstruction_ratio"],
                diagnostics["process_signal_reconstruction_ratio"],
            ]
        ).max()
        point_regret = point_mse / base_mse.clamp_min(1e-8) - 1.0
        standardized_error = (
            residual.square() / distribution["variance"].clamp_min(1e-6)
        ).mean()
        standardized_magnitude = residual.abs() / distribution[
            "variance"
        ].clamp_min(1e-6).sqrt()
        soft_miscoverage = torch.sigmoid(
            8.0 * (standardized_magnitude - 1.6448536269514722)
        ).mean()
        violations = torch.stack(
            [
                reconstruction_ratio - self.reconstruction_tolerance,
                point_regret - self.regret_tolerance,
                (standardized_error - 1.0).abs() - self.variance_tolerance,
                diagnostics["cross_effective_ratio_max"] - self.cross_tolerance,
                diagnostics["cross_spectral_norm"] - self.spectral_tolerance,
                (soft_miscoverage - 0.10).abs() - self.miscoverage_tolerance,
            ]
        )
        if self.primal_dual_enabled:
            optimization_point = robust_point
            lagrangian = (self.constraint_duals.detach() * violations).sum()
            augmented = (
                0.5
                * float(augmented_weight)
                * F.relu(violations).square().sum()
            )
        else:
            # Plain primal control: no Fenchel/sample dual, constraint dual,
            # augmented constraint penalty, or adversarial horizon weighting.
            optimization_point = point_mse
            lagrangian = point_mse.new_tensor(0.0)
            augmented = point_mse.new_tensor(0.0)
        representation = distribution["diagnostics"]["representation_objective"]
        reconstruction = (
            self.continuation.mvpff.auxiliary_loss(inputs)
            if reconstruction_weight > 0.0
            else point_mse.new_tensor(0.0)
        )
        total = (
            optimization_point
            + float(fenchel_weight) * fenchel_primal
            + float(nll_weight) * nll
            + float(representation_weight) * representation
            + float(reconstruction_weight) * reconstruction
            + lagrangian
            + augmented
        )
        group_regret = group_mse.detach() / base_group_mse.clamp_min(1e-8) - 1.0
        parts = {
            "total": total.detach(),
            "point_mse": point_mse.detach(),
            "base_mse": base_mse.detach(),
            "point_regret": point_regret.detach(),
            "mixture_nll": nll.detach(),
            "fenchel_primal": fenchel_primal.detach(),
            "sample_dual_norm": sample_dual_norm.detach(),
            "standardized_error_second_moment": standardized_error.detach(),
            "soft_miscoverage_90": soft_miscoverage.detach(),
            **{name: value.detach() for name, value in distribution["diagnostics"].items()},
        }
        return (
            distribution,
            total,
            parts,
            violations.detach(),
            group_regret,
            residual_bias.detach(),
        )

    @torch.no_grad()
    def initialize_mixture_from_reference(self) -> None:
        self.mixture_head.copy_from_single_component(self.reference_head)

    @torch.no_grad()
    def update_sample_dual(
        self,
        sample_dual: torch.Tensor,
        residual_bias: torch.Tensor,
        learning_rate: float,
    ) -> torch.Tensor:
        # Proximal ascent for max_alpha <alpha,r> - ||alpha||^2 / 2.
        rate = float(learning_rate)
        return (sample_dual + rate * residual_bias) / (1.0 + rate)

    @torch.no_grad()
    def update_constraint_duals(
        self, violations: torch.Tensor, learning_rate: float, maximum: float
    ) -> None:
        if not self.primal_dual_enabled:
            return
        self.constraint_duals.copy_(
            (self.constraint_duals + float(learning_rate) * violations)
            .clamp(0.0, float(maximum))
        )

    @torch.no_grad()
    def update_robust_weights(
        self,
        group_regret: torch.Tensor,
        *,
        learning_rate: float,
        decay: float = 0.90,
    ) -> None:
        if not self.primal_dual_enabled:
            return
        self.regret_ema.mul_(decay).add_(group_regret, alpha=1.0 - decay)
        self.robust_weights.copy_(
            torch.softmax(float(learning_rate) * self.regret_ema, dim=0)
        )

    @torch.no_grad()
    def project_primal(self) -> None:
        self.encoder.project()
        self.separator.project()
        self.continuation.project()
        self.reference_head.project()
        self.mixture_head.project()

    def _adapted_backbone_modules(self, depth: int) -> list[nn.Module]:
        if depth == -1:
            return [self.continuation.mvpff]
        if depth <= 0:
            return []
        base = self.continuation.mvpff
        modules: list[nn.Module | None] = [
            base.mean_mixer,
            base.backbone.component_gate,
        ]
        for branch in base.backbone.branches:
            modules.extend(branch.blocks[-depth:])
            modules.extend([branch.patch_predictor, branch.decoder])
        return [module for module in modules if module is not None]

    def configure_stage(self, stage: str) -> None:
        stage = stage.upper()
        if stage not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unknown stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "B":
            for parameter in self.reference_head.parameters():
                parameter.requires_grad_(True)
        elif stage in {"C", "D"}:
            stage_modules = [self.encoder, self.mixture_head]
            if self.ablation_mode == "deterministic_multibranch":
                stage_modules.append(self.deterministic_process_filters)
                self.deterministic_process_gain.requires_grad_(True)
            else:
                stage_modules.append(self.separator)
            for module in stage_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for name, parameter in self.continuation.named_parameters():
                if not name.startswith("mvpff."):
                    parameter.requires_grad_(True)
        adaptation_depth = self.continuation_unfreeze_blocks
        if stage == "D":
            adaptation_depth = max(1, adaptation_depth)
        if stage in {"C", "D"}:
            for module in self._adapted_backbone_modules(adaptation_depth):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self._enforce_ablation_freezes()
        self._stage = stage
        self.reference_only = stage == "A"
        self.reference_base.eval()
        self.continuation.mvpff.eval()

    def select_reference_only(self, enabled: bool) -> None:
        self.reference_only = bool(enabled)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference_base.eval()
        self.continuation.mvpff.eval()
        if mode and self._stage in {"C", "D"}:
            adaptation_depth = self.continuation_unfreeze_blocks
            if self._stage == "D":
                adaptation_depth = max(1, adaptation_depth)
            for module in self._adapted_backbone_modules(adaptation_depth):
                module.train(True)
        if self._stage != "B":
            self.reference_head.eval()
        return self


__all__ = [
    "PROCESS_SPECS",
    "SENSOR_GROUP_SPECS",
    "CausalStructuralEncoder",
    "MultiStochasticLatentSeparator",
    "ProcessPreservingMVPFContinuation",
    "LatentFlow",
    "SensorProcessGeometry",
    "domain_for_dataset",
    "student_t_log_prob",
]


