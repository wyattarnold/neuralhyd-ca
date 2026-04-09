"""Watershed geometry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import LayerInfo, LayerListResponse
from app.state import state

router = APIRouter(prefix="/api/layers", tags=["layers"])


@router.get("", response_model=LayerListResponse)
def list_layers() -> LayerListResponse:
    """Return metadata for every loaded boundary layer."""
    infos = []
    for key, layer in state.layers.items():
        infos.append(LayerInfo(
            name=layer.name,
            id_col=layer.id_col,
            polygon_count=layer.polygon_count,
            available_series=layer.available_series,
        ))
    return LayerListResponse(layers=infos)


@router.get("/ca_outline/geojson")
def get_ca_outline() -> JSONResponse:
    """Return CA boundary outline GeoJSON."""
    if not state.ca_outline:
        raise HTTPException(404, "CA outline not built")
    return JSONResponse(state.ca_outline)


@router.get("/static_attrs/{basin_id}")
def get_static_attrs(basin_id: str) -> JSONResponse:
    """Return static attributes for a single basin."""
    attrs = state.static_attrs.get(basin_id)
    if attrs is None:
        raise HTTPException(404, f"No attributes for basin: {basin_id}")
    return JSONResponse(attrs)


@router.get("/{layer_key}/geojson")
def get_layer_geojson(layer_key: str) -> JSONResponse:
    """Return simplified GeoJSON FeatureCollection for a layer."""
    layer = state.layers.get(layer_key)
    if layer is None:
        raise HTTPException(404, f"Unknown layer: {layer_key}")
    return JSONResponse(layer.geojson)
