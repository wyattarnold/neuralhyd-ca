"""Timeseries data endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import TimeseriesResponse
from app.state import state

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])

_SERIES_ATTRS = ("vic", "vic_baseflow", "vic_surface", "obs", "obs_baseflow",
                 "lstm_pred", "lstm_fast", "lstm_slow", "lstm_single_pred")


def _read_series(source, polygon_id: str) -> pd.Series | None:
    """Read a single column or return None."""
    if source is None or polygon_id not in source.columns:
        return None
    return source.read_column(polygon_id)


def _to_json_list(series: pd.Series, index: pd.DatetimeIndex) -> list[float | None]:
    """Reindex series to common dates and convert to JSON-friendly list."""
    aligned = series.reindex(index)
    return [None if pd.isna(v) else round(float(v), 1) for v in aligned]


@router.get("/{layer_key}/{polygon_id}", response_model=TimeseriesResponse)
def get_timeseries(layer_key: str, polygon_id: str) -> TimeseriesResponse:
    """Return all available timeseries for a single polygon."""
    layer = state.layers.get(layer_key)
    if layer is None:
        raise HTTPException(404, f"Unknown layer: {layer_key}")

    # Read raw series (each has its own DatetimeIndex)
    raw: dict[str, pd.Series] = {}
    for attr in _SERIES_ATTRS:
        s = _read_series(getattr(layer, attr, None), polygon_id)
        if s is not None:
            raw[attr] = s

    if not raw:
        return TimeseriesResponse(id=polygon_id, layer=layer_key, dates=[])

    # Build union date index, sorted
    all_idx = raw[next(iter(raw))].index
    for s in raw.values():
        all_idx = all_idx.union(s.index)
    all_idx = all_idx.sort_values()

    dates = [d.strftime("%Y-%m-%d") for d in all_idx]

    result = dict(id=polygon_id, layer=layer_key, dates=dates)
    for attr in _SERIES_ATTRS:
        result[attr] = _to_json_list(raw[attr], all_idx) if attr in raw else None

    return TimeseriesResponse(**result)
