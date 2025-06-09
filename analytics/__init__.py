# sentinel_project_root/analytics/__init__.py
#
# PLATINUM STANDARD - Analytics Package API (V2.1 - Re-validated)
# This file initializes the analytics package and defines its public API,
# confirming full compatibility with the corrected system architecture.

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
    aggregate_zonal_stats,
    aggregate_district_stats,
    aggregate_program_kpis
)


# --- Define the Public API for the analytics package ---
# This list controls 'from analytics import *' behavior and is the
# canonical list of public-facing components.
__all__ = [
    # Prediction
    "predict_patient_risk",

    # Forecasting
    "forecast_supply_demand",
    "forecast_epi_trend",

    # Aggregation & KPIs
    "calculate_kpi_statistics",
    "aggregate_zonal_stats",
    "aggregate_district_stats",
    "aggregate_program_kpis",
]
