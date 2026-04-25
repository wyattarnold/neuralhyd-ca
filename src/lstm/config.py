"""Typed configuration container for experiment hyperparameters.

All tuneable values live in a TOML file (default ``scripts/config.toml``).
The ``Config`` dataclass is the single source of truth consumed by every
other module — no magic numbers should appear elsewhere.

Key exports
-----------
Config
    Dataclass holding every hyperparameter; path fields are resolved to
    absolute ``Path`` objects automatically on load.
load_config(path)
    Parse a TOML file and return a validated ``Config`` instance.  Pass
    an alternate path to run a named experiment.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.paths import DEFAULT_CONFIG, TRAINING_OUTPUT_DIR

_PATH_FIELDS = frozenset(
    ["data_dir", "climate_zarr", "flow_zarr", "static_basin_atlas", "static_climate", "output_dir"]
)
_OPTIONAL_PATH_FIELDS = frozenset(
    ["subbasin_intersect_csv", "subbasin_climate_zarr",
     "subbasin_static_attrs", "subbasin_climate_stats"]
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
    static_embedding_dim: int
    static_hidden_size: int
    dropout: float
    static_dropout: float

    # ----- Blended loss -----
    log_loss_lambda: float
    log_loss_epsilon: float

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

    # ----- Dual-pathway defaults (not needed for single/moe configs) -----
    single_hidden_size: int = 128
    fast_window: int = 28
    info_gap: bool = False
    fast_hidden_size: int = 64
    slow_hidden_size: int = 128
    aux_loss_weight: float = 0.4
    baseflow_alpha: float = 0.925

    # ----- Flow normalisation -----
    # The model uses a jointly-learned per-basin scale head (ScaleHead on the
    # static embedding, zero-init so scale = 1.0 at init).  The dataset-side
    # target normalisation remains y / precip_mean for stable initialisation
    # and universal applicability to any basin with climate data; the scale
    # head then absorbs per-basin amplitude end-to-end under the main loss.

    # ----- Optional climate-static handling -----
    exclude_climate_statics: bool = False
    climate_static_features: List[str] = field(default_factory=lambda: [
        "precip_mean", "pet_mean", "aridity_index",
        "snow_fraction", "low_precip_dur",
    ])
    use_window_snow_fraction: bool = False

    # ----- MoE-τ architecture (model_type="moe") -----
    moe_n_experts: int = 2
    moe_expert_hidden_size: int = 128
    moe_gate_hidden_size: int = 64
    moe_attention_dim: int = 32
    moe_tau_init: float = 0.5

    # ----- Extreme-flow loss weighting -----
    # Per-basin quantile-based: weight ramps from 1 at quantile
    # extreme_start_quantile up to extreme_peak_boost at extreme_top_quantile.
    # Defaults (p99 → p99.9) treat 1-in-100-day events as the start of
    # "extreme" and 1-in-1000-day events as the full-weight peak, for every
    # basin regardless of flow regime.  Thresholds are computed per basin
    # on the normalised target (flow / precip_mean) at fold-init time.
    extreme_start_quantile: float = 0.99  # lower cutoff (ramp start)
    extreme_top_quantile: float = 0.999   # upper cutoff (ramp end / full boost)
    extreme_peak_boost: float = 8.0       # max multiplier at the top quantile

    # ----- Per-basin loss weighting (gradient balancing) -----
    # Weight per basin = 1 / max(var_b, basin_loss_min_var)^basin_loss_weight_exponent
    # where var_b = var(flow / precip_mean) for basin b.
    #   exponent = 0.0  → no weighting (uniform; high-var basins dominate)
    #   exponent = 0.5  → sqrt-compressed (~10× spread)
    #   exponent = 1.0  → full inverse-variance (~100× spread, balanced per-basin)
    basin_loss_weight_exponent: float = 0.5
    basin_loss_min_var: float = 0.1

    # ----- Subbasin-mode (HUC10 / HUC12 aggregation) -----
    # When ``spatial_mode="subbasin"`` the model is trained on HUC sub-basins
    # and predictions are area-weight-aggregated to the gauge before the loss
    # is computed.  Loss is evaluated in **mm/day** (not runoff-ratio) space
    # in this mode.  See ``src/data/subbasin_gauge_intersect.py`` and the
    # subbasin branch in ``src/lstm/train.py``.
    spatial_mode: str = "gauge"              # "gauge" (default) | "subbasin"
    subbasin_level: str = "huc12"            # "huc10" or "huc12"
    # Pipeline artefact paths (used only in subbasin mode)
    subbasin_intersect_csv: Path | None = None  # gauge × subbasin overlap table
    subbasin_climate_zarr: Path | None = None   # per-subbasin climate zarr cube
    subbasin_static_attrs: Path | None = None   # Physical_Attributes_<LEVEL>.csv
    subbasin_climate_stats: Path | None = None  # Climate_Statistics_<LEVEL>.csv
    # Drop a gauge when it sits inside a single subbasin (only one
    # overlap) and covers less than this fraction of that subbasin's
    # area — applied at pipeline-build time by
    # ``subbasin_gauge_intersect.py`` and recorded here for traceability only.
    gauge_min_fraction_of_subbasin: float = 0.70

    # ----- Probabilistic output -----
    output_type: str = "deterministic"   # "deterministic" or "cmal"
    cmal_n_components: int = 3           # K mixture components for CMAL
    cmal_hidden_size: int = 32           # CMALHead intermediate layer width
    cmal_loss: str = "crps"               # "nll" or "crps"
    cmal_crps_n_samples: int = 50        # samples per component for CRPS spread term
    cmal_entropy_weight: float = 0.1     # weight on mixture-weight entropy reg (0 = off)
    cmal_scale_reg_weight: float = 0.0   # weight on scale-collapse penalty (0 = off)
    cmal_beta_crps: float = 0.0          # β-CRPS spread penalty (0 = off; 0.5 typical)

    # ----- Grouped static encoder -----
    # Ordered dict of group_name → list[feature_name].  When set, features
    # are concatenated **in group order** and a GroupedStaticEncoder is used
    # instead of the flat MLP.  Every feature in effective_static_features
    # must appear in exactly one group.
    static_feature_groups: dict[str, List[str]] | None = None
    static_group_hidden: int = 0   # per-group encoder dim (0 = auto)

    # ----- Hardware / throughput tuning -----
    # Each flag is a no-op on devices that don't support the feature, so
    # defaults are safe on macOS MPS and CPU.
    use_amp: bool = True                # bf16 autocast (CUDA only; ignored on MPS/CPU)
    pin_dataset_to_device: bool = True  # pre-move HydroDataset tensors to the
                                        # training device to eliminate per-batch
                                        # host→device copies. Forces num_workers=0
                                        # and pin_memory=False on non-CPU devices.
    cudnn_benchmark: bool = True        # torch.backends.cudnn.benchmark (CUDA only)
    tf32: bool = True                   # TF32 matmul on Ampere+ (CUDA only)

    def __post_init__(self) -> None:
        for f in _PATH_FIELDS:
            val = getattr(self, f)
            if isinstance(val, str):
                setattr(self, f, Path(val))
        for f in _OPTIONAL_PATH_FIELDS:
            val = getattr(self, f)
            if isinstance(val, str):
                setattr(self, f, Path(val))
        if self.spatial_mode not in ("gauge", "subbasin"):
            raise ValueError(
                f"spatial_mode must be 'gauge' or 'subbasin', got {self.spatial_mode!r}"
            )
        if self.spatial_mode == "subbasin":
            if self.output_type == "cmal":
                raise ValueError(
                    "spatial_mode='subbasin' is not yet supported with output_type='cmal'."
                )
            if self.subbasin_level not in ("huc10", "huc12"):
                raise ValueError(
                    f"subbasin_level must be 'huc10' or 'huc12', got {self.subbasin_level!r}"
                )
            missing = [
                f for f in ("subbasin_intersect_csv", "subbasin_climate_zarr",
                            "subbasin_static_attrs", "subbasin_climate_stats")
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"spatial_mode='subbasin' requires these paths in the config: {missing}"
                )

    @property
    def effective_static_features(self) -> List[str]:
        """Static features after optional exclusion/addition of climate-derived ones.

        When ``static_feature_groups`` is defined, features are returned in
        **group order** (group-1 features, then group-2, …) so that
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
    def static_group_sizes(self) -> list[int] | None:
        """Number of features per group (in group order), or None."""
        if self.static_feature_groups is None:
            return None
        return [len(v) for v in self.static_feature_groups.values()]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Load a TOML config file and return a Config instance.

    output_dir is derived from the config filename when not explicitly set:
      config.toml          → data/training/output/
      config_<name>.toml   → data/training/output/<name>/
      <other>.toml         → data/training/output/<stem>/
    """
    path = Path(path).resolve()
    config_dir = path.parent
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    # Flatten TOML sections into a single dict of field → value.
    # Sections whose name matches a Config field that expects a nested
    # dict (e.g. static_feature_groups) are preserved as-is.
    _NESTED_FIELDS = {"static_feature_groups"}
    flat: dict = {}
    for key, val in raw.items():
        if key in _NESTED_FIELDS:
            flat[key] = val
        elif isinstance(val, dict):
            flat.update(val)

    # Derive output_dir from filename when the TOML doesn't specify it.
    if "output_dir" not in flat:
        stem = path.stem  # e.g. "config", "config_single", "my_exp"
        if stem == "config":
            flat["output_dir"] = str(TRAINING_OUTPUT_DIR)
        elif stem.startswith("config_"):
            flat["output_dir"] = str(TRAINING_OUTPUT_DIR / stem[len("config_"):])
        else:
            flat["output_dir"] = str(TRAINING_OUTPUT_DIR / stem)

    cfg = Config(**flat)

    # Resolve relative paths against the config file's directory so the
    # scripts work from any cwd.
    for f in _PATH_FIELDS:
        val = getattr(cfg, f)
        if not val.is_absolute():
            setattr(cfg, f, (config_dir / val).resolve())
    for f in _OPTIONAL_PATH_FIELDS:
        val = getattr(cfg, f)
        if val is not None and not val.is_absolute():
            setattr(cfg, f, (config_dir / val).resolve())

    return cfg
