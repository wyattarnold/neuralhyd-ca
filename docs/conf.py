# Sphinx configuration for NeuralHydrology Explorer docs
# Rendered inside an iframe in the app — keep the theme compact.

from __future__ import annotations

project = "NeuralHydrology Explorer"
copyright = "2026, California Department of Water Resources"
author = "DWR"

extensions = []

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -- use Furo for a clean, sidebar-free look in the iframe -----
html_theme = "furo"
html_title = "NeuralHydrology Explorer"
html_theme_options = {
    "sidebar_hide_name": True,
}
# Hide the "Built with Sphinx" footer
html_show_sourcelink = False
html_show_sphinx = False
