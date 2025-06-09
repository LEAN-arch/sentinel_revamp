# sentinel_project_root/visualization/__init__.py
#
# PLATINUM STANDARD - Visualization Package API
# This file initializes the visualization package and defines its public API,
# offering a clean interface for creating themed plots and UI elements.

"""
Initializes the visualization package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.
"""

# --- Charting Functions ---
# High-level, semantic functions for creating standardized, theme-aware Plotly charts.
from .plots import (
    create_empty_figure,
    plot_kpi_trend,
    plot_categorical_distribution,
    plot_choropleth_map,
    plot_supply_forecast,
    plot_age_distribution,
    plot_risk_pyramid
)

# --- UI Element Rendering ---
# Functions for displaying consistent, themed UI components in Streamlit.
from .ui_elements import (
    render_metric_card,
    render_main_header,
    render_dashboard_filter_form
)

# --- Theming ---
# Exposes the global Plotly theme template defined for the application.
from .themes import sentinel_theme_template


# --- Define the Public API for the visualization package ---
# This list controls what is imported when a user does `from visualization import *`
# and is considered the canonical list of public-facing components.
__all__ = [
    # charts
    "create_empty_figure",
    "plot_kpi_trend",
    "plot_categorical_distribution",
    "plot_choropleth_map",
    "plot_supply_forecast",
    "plot_age_distribution",
    "plot_risk_pyramid",

    # ui_elements
    "render_metric_card",
    "render_main_header",
    "render_dashboard_filter_form",

    # themes
    "sentinel_theme_template"
]
