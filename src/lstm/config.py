"""Typed configuration container for experiment hyperparameters.

All tuneable values live in named TOML files under ``scripts/``.  The
``Config`` dataclass is the single source of truth consumed by every
other module — no magic numbers should appear elsewhere.

Key exports
-----------
Config
    Dataclass holding every hyperparameter; path fields are resolved to
    absolute ``Path`` objects automatically on load.
load_config(path)
    Parse a TOML file and return a validated ``Config`` instance.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.paths import TRAINING_OUTPUT_DIR

_PATH_FIELDS = frozenset(
    ["data_dir", "climate_zarr", "flow_zarr", "static_basin_atlas", "static_climate", "output_dir"]
)


@dataclass
class Config:
    # ----- Paths -----
    data_dir: Path
    climate_zarr: Path           # data/training/climate/<scope>.zarr
    flow_zarr: Path              # data/training/flow.zarr
    static_basin_atlas: Path
    static_climate: Path
    output_dir: Path

    # ----- Sequence / windows -----
    seq_len: int

    # ----- Model architecture -----
    model_type: str
    dropout: float
    static_dropout: float

    # ----- Training -----
    batch_size: int
    learning_rate: float
    weight_decay: float
    input_noise_std: float
    warmup_epochs: int
    num_epochs: int
    patience: int
    min_delta: float
    num_workers: int
    grad_clip: float
    use_swa: bool
    swa_lr: float
    swa_patience: int

    # ----- Validation -----
    n_folds: int
    seed: int

    # ----- Feature lists -----
    dynamic_features: List[str]
    static_features: List[str]
    log_transform_static: List[str]

    # ----- Validation selection -----
    validation_selection_metric: str = "loss"  # "loss", "nse", or "kge"

    # ----- Blended loss -----
    log_loss_lambda: float = 0.05
    log_loss_epsilon: float = 0.001

    # ----- Dual-pathway defaults (not needed for single/moe configs) -----
    single_hidden_size: int = 128
    fast_window: int = 28
    info_gap: bool = False
    fast_hidden_size: int = 64
    slow_hidden_size: int = 128
    aux_loss_weight: float = 0.4
    baseflow_alpha: float = 0.925
    static_embedding_dim: int = 32   # flat encoder only; ignored in grouped mode
    static_hidden_size: int = 64     # flat encoder only; ignored in grouped mode

    # ----- Basin domain / filtering -----
    training_manifest: str = "gages"
    include_cdec_basins: bool = True

    # ----- Flow normalisation -----
    # When normalize_by_precip=True (default): scale_map[b] = precip_mean[b];
    # targets are dimensionless runoff ratios (flow / precip_mean).
    # When normalize_by_precip=False: scale_map[b] = 1.0; targets stay in
    # mm/day and the model must learn absolute amplitude from static embedding.
    normalize_by_precip: bool = True

    # ----- Optional climate-static handling -----
    exclude_climate_statics: bool = False
    climate_static_features: List[str] = field(default_factory=lambda: [
        "precip_mean", "pet_mean", "aridity_index",
        "snow_fraction", "low_precip_dur",
    ])
    use_window_snow_fraction: bool = False

    # ----- Static attribute representation -----
    # ``static_attribute_mode='flat'`` uses one row per gauge watershed with the
    # flat MLP encoder. ``'grouped'`` uses gauge-watershed static attributes,
    # adds gauge-level derived routing/area features when requested, and feeds
    # them through semantic static groups.
    static_attribute_mode: str = "flat"  # "flat" or "grouped"
    categorical_static_features: List[str] = field(default_factory=list)
    categorical_static_feature_values: dict[str, List[int]] = field(default_factory=dict)

    # ----- MoE-tau architecture (model_type="moe") -----
    moe_n_experts: int = 2
    moe_expert_hidden_size: int = 128
    moe_gate_hidden_size: int = 64
    moe_tau_init: float = 0.5
    moe_tau_min: float = 1e-4
    # Load-balancing: weight on -H(mean_pi) penalty; pushes gate toward uniform routing.
    # 0.0 = off (default). 0.05 is a reasonable starting value.
    moe_balance_weight: float = 0.0

    # ----- Extreme-flow loss weighting -----
    # Per-basin quantile-based: weight ramps from 1 at quantile
    # extreme_start_quantile up to extreme_peak_boost at extreme_top_quantile.
    # Defaults (p99 -> p99.9) treat 1-in-100-day events as the start of
    # "extreme" and 1-in-1000-day events as the full-weight peak, for every
    # basin regardless of flow regime.  Thresholds are computed per basin
    # on the normalised target (flow / precip_mean) at fold-init time.
    extreme_start_quantile: float = 0.99  # lower cutoff (ramp start)
    extreme_top_quantile: float = 0.999   # upper cutoff (ramp end / full boost)
    extreme_peak_boost: float = 8.0       # max multiplier at the top quantile

    # ----- Per-basin loss weighting (gradient balancing) -----
    # Weight per basin = 1 / max(var_b, basin_loss_min_var)^basin_loss_weight_exponent
    # where var_b = var(flow / precip_mean) for basin b.
    #   exponent = 0.0  -> no weighting (uniform; high-var basins dominate)
    #   exponent = 0.5  -> sqrt-compressed (~10x spread)
    #   exponent = 1.0  -> full inverse-variance (~100x spread, balanced per-basin)
    basin_loss_weight_exponent: float = 0.5
    basin_loss_min_var: float = 0.1
    # Optional record-length balancing.  DataLoader sampling is per day, so without
    # this a basin's total exposure is still roughly proportional to n_obs.  Setting
    # 0.5 partially compresses long/short-record imbalance; 1.0 gives each basin
    # roughly equal total loss mass per epoch.  Default 0 preserves legacy behavior.
    basin_record_weight_exponent: float = 0.0

    # ----- Probabilistic output -----
    output_type: str = "deterministic"   # "deterministic" or "cmal"
    cmal_n_components: int = 3           # K mixture components for CMAL
    cmal_hidden_size: int = 32           # CMALHead intermediate layer width
    cmal_loss: str = "crps"               # "nll" or "crps"
    cmal_crps_n_samples: int = 50        # samples per component for CRPS spread term
    cmal_entropy_weight: float = 0.1     # weight on mixture-weight entropy reg (0 = off)
    cmal_scale_reg_weight: float = 0.0   # weight on scale-collapse penalty (0 = off)
    cmal_beta_crps: float = 0.0          # beta-CRPS spread penalty (0 = off; 0.5 typical)
    # ----- CMAL point readout (single LSTM, CMAL only) -----
    # How q_total (the deployed point prediction) is read off the mixture:
    #   "mean"   -> sum_k pi_k (mu_k + b_r_k - b_l_k)              (default)
    #   "median" -> sum_k pi_k * Q_ALD,k(0.5)  (pi-weighted component median;
    #               lower than the mean for right-skewed mixtures, so it pairs
    #               better with a log-shaped low tail / FLV).
    cmal_point_estimate: str = "mean"    # "mean" or "median"
    # ----- CMAL loss shaping (tune the distribution fit directly) -----
    # FLV lever: blend a log-space (chained) energy score into CRPS.  Scale-
    # invariant, so it pulls the low-flow fit down proportionally.  0 = off.
    cmal_log_crps_lambda: float = 0.0
    # FHV lever: up-weight extreme high-flow samples in CRPS using the per-basin
    # extreme_ramp_weight (extreme_start_quantile -> extreme_top_quantile,
    # extreme_peak_boost).  Makes the mixture fit peaks instead of averaging them
    # away.  False = off.
    cmal_extreme_weight: bool = False

    # ----- Grouped static encoder -----
    # Ordered dict of group_name -> list[feature_name].  When set, features
    # are concatenated **in group order** and a GroupedStaticEncoder is used
    # instead of the flat MLP.  Every feature in effective_static_features
    # must appear in exactly one group.  Per-group output dims are auto-sized
    # to max(ceil(2*sqrt(n)), 4), and each model branch gets its own encoder.
    static_feature_groups: dict[str, List[str]] | None = None
    static_group_dropout: float = 0.0  # probability of dropping a whole static group

    # ----- Hardware / throughput tuning -----
    # Each flag is a no-op on devices that don't support the feature, so
    # defaults are safe on macOS MPS and CPU.
    use_amp: bool = True                # bf16 autocast (CUDA only; ignored on MPS/CPU)
    cudnn_benchmark: bool = True        # torch.backends.cudnn.benchmark (CUDA only)
    tf32: bool = True                   # TF32 matmul on Ampere+ (CUDA only)

    def __post_init__(self) -> None:
        for f in _PATH_FIELDS:
            val = getattr(self, f)
            if isinstance(val, str):
                setattr(self, f, Path(val))
        self.training_manifest = str(self.training_manifest).lower()
        if self.training_manifest not in {"gages"}:
            raise ValueError(
                "training_manifest must be 'gages'; "
                f"got {self.training_manifest!r}"
            )
        self.static_attribute_mode = str(self.static_attribute_mode).lower()
        if self.static_attribute_mode not in {"flat", "grouped"}:
            raise ValueError(
                "static_attribute_mode must be 'flat' or 'grouped'; "
                f"got {self.static_attribute_mode!r}"
            )
        if not (0.0 <= self.static_group_dropout < 1.0):
            raise ValueError("static_group_dropout must be in [0, 1)")
        categorical = set(self.categorical_static_features)
        for feat in categorical:
            if feat not in self.categorical_static_feature_values:
                raise ValueError(
                    f"categorical_static_feature_values must define categories for {feat!r}"
                )
            if not self.categorical_static_feature_values[feat]:
                raise ValueError(f"categorical feature {feat!r} must define at least one category")
        for feat in self.effective_static_features:
            if feat in categorical and feat not in self.categorical_static_feature_values:
                raise ValueError(
                    f"categorical static feature {feat!r} is missing category values"
                )
        if self.validation_selection_metric not in ("loss", "nse", "kge"):
            raise ValueError(
                "validation_selection_metric must be 'loss', 'nse', or 'kge'; "
                f"got {self.validation_selection_metric!r}"
            )
        self.cmal_point_estimate = str(self.cmal_point_estimate).lower()
        if self.cmal_point_estimate not in {"mean", "median"}:
            raise ValueError(
                "cmal_point_estimate must be 'mean' or 'median'; "
                f"got {self.cmal_point_estimate!r}"
            )
        if not (0.0 <= self.cmal_log_crps_lambda < 1.0):
            raise ValueError(
                "cmal_log_crps_lambda must be in [0, 1); "
                f"got {self.cmal_log_crps_lambda}"
            )
        if not (0.0 < self.moe_tau_init < 1.0):
            raise ValueError(f"moe_tau_init must be in (0, 1), got {self.moe_tau_init}")
        if not (0.0 < self.moe_tau_min < 1.0):
            raise ValueError(f"moe_tau_min must be in (0, 1), got {self.moe_tau_min}")

    @property
    def effective_static_features(self) -> List[str]:
        """Static features after optional exclusion/addition of climate-derived ones.

        When ``static_feature_groups`` is defined, features are returned in
        **group order** (group-1 features, then group-2, ...) so that
        ``GroupedStaticEncoder`` can split the flat vector by group sizes.
        """
        feats = list(self.static_features)
        if self.exclude_climate_statics and self.climate_static_features:
            exclude = set(self.climate_static_features)
            feats = [f for f in feats if f not in exclude]
        if self.use_window_snow_fraction:
            if "snow_fraction" not in feats:
                feats.append("snow_fraction")

        if self.static_feature_groups is not None:
            # Re-order features to match group order
            ordered: list[str] = []
            for group_feats in self.static_feature_groups.values():
                ordered.extend(group_feats)
            # Validate: every effective feature must be in a group
            feat_set = set(feats)
            ordered_set = set(ordered)
            missing = feat_set - ordered_set
            extra = ordered_set - feat_set
            if missing:
                raise ValueError(
                    f"Features missing from static_feature_groups: {missing}"
                )
            if extra:
                raise ValueError(
                    f"Features in static_feature_groups but not in "
                    f"effective_static_features: {extra}"
                )
            return ordered
        return feats

    @property
    def grouped_static_output_dim(self) -> int | None:
        """Auto-computed output dim of GroupedStaticEncoder, or None for flat mode."""
        sizes = self.static_group_sizes
        if sizes is None:
            return None
        import math
        return sum(max(math.ceil(2.0 * math.sqrt(n)), 4) for n in sizes)

    @property
    def static_group_sizes(self) -> list[int] | None:
        """Number of encoded features per group (in group order), or None."""
        if self.static_feature_groups is None:
            return None
        return [sum(self._encoded_size_for_feature(f) for f in v)
                for v in self.static_feature_groups.values()]

    @property
    def static_group_names(self) -> list[str] | None:
        """Semantic static group names in input-vector order, or None."""
        if self.static_feature_groups is None:
            return None
        return list(self.static_feature_groups.keys())

    @property
    def encoded_static_feature_names(self) -> List[str]:
        """Feature names after expanding categorical columns to one-hot slots."""
        names: list[str] = []
        categorical = set(self.categorical_static_features)
        for feat in self.effective_static_features:
            if feat not in categorical:
                names.append(feat)
                continue
            for value in self.categorical_static_feature_values[feat]:
                names.append(f"{feat}={int(value)}")
        return names

    @property
    def encoded_static_is_categorical(self) -> List[bool]:
        """Mask over encoded static slots that should stay on their native scale."""
        flags: list[bool] = []
        categorical = set(self.categorical_static_features)
        for feat in self.effective_static_features:
            width = self._encoded_size_for_feature(feat)
            flags.extend([feat in categorical] * width)
        return flags

    def _encoded_size_for_feature(self, feature: str) -> int:
        if feature in set(self.categorical_static_features):
            return len(self.categorical_static_feature_values[feature])
        return 1


def load_config(path: str | Path) -> Config:
    """Load a TOML config file and return a Config instance.

        output_dir is derived from the config filename when not explicitly set:
            cfg_<name>.toml      -> data/training/output/<name>/
            <other>.toml         -> data/training/output/<stem>/
    """
    path = Path(path).resolve()
    config_dir = path.parent
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    # Flatten TOML sections into a single dict of field -> value.
    # Sections whose name matches a Config field that expects a nested
    # dict (e.g. static_feature_groups) are preserved as-is.
    _NESTED_FIELDS = {"static_feature_groups", "categorical_static_feature_values"}
    flat: dict = {}
    for key, val in raw.items():
        if key in _NESTED_FIELDS:
            flat[key] = val
        elif isinstance(val, dict):
            flat.update(val)

    # Derive output_dir from filename when the TOML doesn't specify it.
    if "output_dir" not in flat:
        stem = path.stem
        if stem.startswith("cfg_"):
            flat["output_dir"] = str(TRAINING_OUTPUT_DIR / stem[len("cfg_"):])
        else:
            flat["output_dir"] = str(TRAINING_OUTPUT_DIR / stem)

    cfg = Config(**flat)

    # Resolve relative paths against the config file's directory so the
    # scripts work from any cwd.
    for f in _PATH_FIELDS:
        val = getattr(cfg, f)
        if not val.is_absolute():
            setattr(cfg, f, (config_dir / val).resolve())

    return cfg
