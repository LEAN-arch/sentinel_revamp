# sentinel_project_root/analytics/__init__.py
#
# PLATINUM STANDARD - Analytics Package API
# This file initializes the analytics package and defines its public API.
# It provides a clean, high-level interface to all predictive models,
# forecasting engines, and statistical aggregation functions.

"""
Initializes the analytics package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.
"""

# --- Prediction Models ---
# Functions for making predictions on new data, powered by scikit-learn.
from .prediction import predict_patient_risk


# --- Forecasting Models ---
# Functions for time-series forecasting, powered by Prophet.
from .forecasting import (
    forecast_supply_demand,
    forecast_epi_trend
)

# --- Statistical Aggregation & KPI Calculation ---
# Functions that perform robust statistical aggregations to generate
# decision-grade Key Performance Indicators (KPIs).
from .aggregation import (
    calculate_kpi_statistics,
    aggregate_population_health_stats,
    aggregate_zonal_stats
)


# --- Define the public API for the analytics package ---
# This list controls what is imported when a user does `from analytics import *`
# and is considered the canonical list of public-facing components.
__all__ = [
    # Prediction
    "predict_patient_risk",

    # Forecasting
    "forecast_supply_demand",
    "forecast_epi_trend",

    # Aggregation & KPIs
    "calculate_kpi_statistics",
    "aggregate_population_health_stats",
    "aggregate_zonal_stats",
]
