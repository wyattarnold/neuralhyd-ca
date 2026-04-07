"""Backward-compatible re-export of all path constants.

The canonical definitions now live in ``src.paths``.  Existing imports
from ``src.data.paths`` continue to work unchanged.
"""
from __future__ import annotations

# Re-export everything so  ``from src.data.paths import X``  still works.
from src.paths import *  # noqa: F401,F403
