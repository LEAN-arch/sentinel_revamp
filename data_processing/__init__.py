# sentinel_project_root/data_processing/__init__.py
#
# PLATINUM STANDARD - Data Processing Package API
# This file initializes the data_processing package and defines its public API.
# It provides a clean, high-level interface for loading, preparing, and
# enriching all data sources for the Sentinel ecosystem.

"""
Initializes the data_processing package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.
"""

# --- Primary Data Loading Functions ---
# These functions abstract away the underlying file types and provide
# consistently cleaned and typed DataFrames.
from .loaders import (
    load_health_records,
    load_iot_records,
    load_lab_results,
    load_supply_utilization,
    load_zone_data,
    load_ml_model,
    load_json_asset
)

# --- Data Preparation & Cleaning ---
# The DataPipeline provides a modern, fluent (chainable) interface for
# applying a sequence of cleaning and transformation steps.
from .pipeline import DataPipeline

# --- Data Enrichment ---
# Enrichment functions add new, calculated columns (features) to DataFrames
# to make them ready for analytics and machine learning.
from .enrichment import (
    enrich_health_records_with_features
)


# --- Define the Public API for the data_processing package ---
# This list controls what is imported when a user does `from data_processing import *`
# and is considered the canonical list of public-facing components.
__all__ = [
    # --- Loading ---
    "load_health_records",
    "load_iot_records",
    "load_lab_results",
    "load_supply_utilization",
    "load_zone_data",
    "load_ml_model",
    "load_json_asset",

    # --- Preparation ---
    "DataPipeline",

    # --- Enrichment ---
    "enrich_health_records_with_features",
]
