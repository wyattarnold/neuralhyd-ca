from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.paths import DEFAULT_CONFIG, TRAINING_OUTPUT_DIR

_PATH_FIELDS = frozenset(
    ["data_dir", "climate_dir", "flow_dir", "static_basin_atlas", "static_climate", "output_dir"]
)


@dataclass
class Config:
    # ----- Paths -----
    data_dir: Path
    climate_dir: Path
    flow_dir: Path
    static_basin_atlas: Path
    static_climate: Path
    output_dir: Path

    # ----- Sequence / windows -----
    seq_len: int
    fast_window: int
    info_gap: bool

    # ----- Model architecture -----
    model_type: str
    fast_hidden_size: int
    slow_hidden_size: int
    single_hidden_size: int
    static_embedding_dim: int
    static_hidden_size: int
    dropout: float
    static_dropout: float

    # ----- Pathway auxiliary loss (dual model only) -----
    aux_loss_weight: float
    baseflow_alpha: float
    aux_peak_asymmetry: float

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
    holdout_fraction: float
    seed: int

    # ----- Feature lists -----
    dynamic_features: List[str]
    static_features: List[str]
    log_transform_static: List[str]

    # ----- Optional climate-static handling -----
    exclude_climate_statics: bool = False
    climate_static_features: List[str] | None = None
    use_window_snow_fraction: bool = False

    def __post_init__(self) -> None:
        for f in _PATH_FIELDS:
            val = getattr(self, f)
            if isinstance(val, str):
                setattr(self, f, Path(val))

    @property
    def effective_static_features(self) -> List[str]:
        """Static features after optional exclusion/addition of climate-derived ones."""
        feats = list(self.static_features)
        if self.exclude_climate_statics and self.climate_static_features:
            exclude = set(self.climate_static_features)
            feats = [f for f in feats if f not in exclude]
        if self.use_window_snow_fraction:
            # Append window-derived snow_fraction (computed in dataset)
            if "snow_fraction" not in feats:
                feats.append("snow_fraction")
        return feats


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
    flat: dict = {}
    for val in raw.values():
        if isinstance(val, dict):
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

    # Apply defaults for optional fields not present in the TOML.
    if not hasattr(cfg, "climate_static_features") or cfg.climate_static_features is None:
        cfg.climate_static_features = [
            "precip_mean", "pet_mean", "aridity_index",
            "snow_fraction", "low_precip_dur",
        ]

    return cfg
