"""FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import layers, timeseries
from app.state import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data at startup, release on shutdown."""
    state.load()
    # Eagerly load one small layer GeoJSON to verify paths work
    _ = state.layers["huc8"].geojson
    print(f"Loaded {len(state.layers)} layers from pre-built app/data/")
    yield


def create_app(data_dir: str | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    if data_dir is None:
        # Auto-detect: assume we're inside the lstmhyd-ca repo
        data_dir = str(Path(__file__).resolve().parents[1])

    app = FastAPI(
        title="Streamflow Explorer",
        lifespan=lifespan,
    )

    app.include_router(layers.router)
    app.include_router(timeseries.router)

    # Serve built Sphinx docs if they exist
    docs_dir = Path(__file__).resolve().parents[1] / "docs" / "_build" / "html"
    if docs_dir.is_dir():
        app.mount("/docs", StaticFiles(directory=str(docs_dir), html=True), name="docs")

    # Serve built frontend if it exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
