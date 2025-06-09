# sentinel_project_root/data_processing/__init__.py
#
# PLATINUM STANDARD - Data Processing Package API (V2.1 - Re-validated)
# This file initializes the data_processing package and defines its public API,
# confirming full compatibility with the corrected system architecture.

"""
Initializes the data_processing package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.
"""

# --- Primary Data Loading Functions ---
# These functions abstract away file formats and provide clean DataFrames.
from .loaders import (
    load_health_records,
    load_lab_results,
    load_zone_data,
    load_supply_utilization,
    load_program_outcomes,
    load_contact_tracing,
    load_ntd_mass_drug_admin,
    load_ml_model
)

# --- Data Preparation & Cleaning ---
# Provides a fluent (chainable) interface for data preparation.
from .pipeline import DataPipeline

# --- Data Enrichment (Feature Engineering) ---
# Functions that add calculated features to DataFrames for analytics.
from .enrichment import (
    enrich_health_records_with_features,
    enrich_lab_results_with_features
)


# --- Define the Public API for the data_processing package ---
# This list controls 'from data_processing import *' behavior and is the
# canonical list of public components.
__all__ = [
    # --- Loading ---
    "load_health_records",
    "load_lab_results",
    "load_zone_data",
    "load_supply_utilization",
    "load_program_outcomes",
    "load_contact_tracing",
    "load_ntd_mass_drug_admin",
    "load_ml_model",

    # --- Preparation ---
    "DataPipeline",

    # --- Enrichment ---
    "enrich_health_records_with_features",
    "enrich_lab_results_with_features",
]
