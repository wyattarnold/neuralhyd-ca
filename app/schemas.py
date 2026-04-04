"""Pydantic response models."""

from __future__ import annotations

from pydantic import BaseModel


class LayerInfo(BaseModel):
    name: str
    id_col: str
    polygon_count: int
    available_series: list[str]


class LayerListResponse(BaseModel):
    layers: list[LayerInfo]


class TimeseriesResponse(BaseModel):
    id: str
    layer: str
    dates: list[str]
    vic: list[float | None] | None = None
    obs: list[float | None] | None = None
    lstm_pred: list[float | None] | None = None
    lstm_fast: list[float | None] | None = None
    lstm_slow: list[float | None] | None = None
