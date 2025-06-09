# sentinel_project_root/data_processing/loaders.py
#
# PLATINUM STANDARD - Unified Data Loading Engine (V2 - Public Health Mission Upgrade)
# This module is upgraded to ingest a comprehensive suite of public health datasets,
# including programmatic, adherence, and contact tracing data.

import logging
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Any, Dict, List

# --- Core Application Imports ---
try:
    from config.settings import settings
    from .pipeline import DataPipeline
    from .helpers import robust_json_load
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in loaders.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A declarative, configuration-driven engine for loading and preparing
    data sources. It ensures all data entering the system is validated,
    cleaned, and correctly typed, fulfilling a consistent data contract.
    """
    def __init__(self, data_source_dir: Path):
        self.base_dir = data_source_dir
        if not self.base_dir.exists():
            logger.warning(f"Data source directory not found: {self.base_dir}. Creating it.")
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, file_path: Path) -> Path:
        """Resolves a file path relative to the base data directory."""
        return self.base_dir / file_path if not file_path.is_absolute() else file_path

    def load_csv(self, file_path: Path, date_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Loads a CSV file and applies a standardized cleaning pipeline.
        This method is designed to be resilient, returning an empty DataFrame
        if the source file is missing or critically malformed.
        """
        full_path = self._get_path(file_path)
        log_ctx = f"CSV({full_path.name})"
        logger.debug(f"[{log_ctx}] Attempting to load data from {full_path}")

        if not full_path.exists():
            # This is a common, non-critical scenario (e.g., an optional dataset is not present).
            logger.warning(f"[{log_ctx}] Source file not found. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            df = pd.read_csv(full_path, low_memory=False)
            if df.empty:
                logger.warning(f"[{log_ctx}] File is empty.")
                return pd.DataFrame()

            # Automatically apply cleaning pipeline on load.
            pipeline = DataPipeline(df).clean_column_names()
            if date_cols:
                pipeline.convert_date_columns(date_cols)

            df_processed = pipeline.get_df()
            logger.info(f"[{log_ctx}] Successfully loaded and cleaned {len(df_processed)} records.")
            return df_processed

        except Exception as e:
            logger.critical(f"[{log_ctx}] CRITICAL ERROR loading or processing file: {e}", exc_info=True)
            return pd.DataFrame()

    def load_ml_model(self, model_path: Path) -> Optional[Any]:
        """Loads a serialized machine learning model."""
        log_ctx = f"ML_Model({model_path.name})"
        logger.debug(f"[{log_ctx}] Attempting to load model from {model_path}")
        if not model_path.exists():
            logger.error(f"[{log_ctx}] Model file not found.")
            return None
        try:
            model = joblib.load(model_path)
            logger.info(f"[{log_ctx}] Successfully loaded model.")
            return model
        except Exception as e:
            logger.critical(f"[{log_ctx}] CRITICAL ERROR loading model: {e}", exc_info=True)
            return None


# --- Singleton Instances for Different Directories ---
_data_loader = DataLoader(settings.directories.data_sources)
_model_loader = DataLoader(settings.directories.ml_models)


# --- Public API Functions for Data Loading ---

def load_health_records() -> pd.DataFrame:
    """Loads and cleans the primary health records dataset."""
    return _data_loader.load_csv(
        settings.health_records_path, date_cols=['encounter_date']
    )

def load_lab_results() -> pd.DataFrame:
    """Loads and cleans laboratory results data."""
    # Settings now points to the single, comprehensive lab results file
    return _data_loader.load_csv(
        settings.lab_results_path,
        date_cols=['sample_collection_date', 'result_date']
    )

def load_zone_data() -> pd.DataFrame:
    """Loads and merges zone attribute and geometry data."""
    attributes_df = _data_loader.load_csv(settings.zone_attributes_path)
    if 'zone_id' in attributes_df.columns:
        attributes_df['zone_id'] = attributes_df['zone_id'].astype(str)

    geo_data = robust_json_load(settings.zone_geometries_path, "GeoJSON")
    geometries_df = pd.DataFrame()

    if geo_data and isinstance(geo_data.get('features'), list):
        try:
            geometries_list = [
                {"zone_id": str(feat["properties"]["zone_id"]), "geometry": feat["geometry"]}
                for feat in geo_data["features"]
                if feat.get("properties") and feat.get("geometry") and feat["properties"].get("zone_id") is not None
            ]
            geometries_df = pd.DataFrame(geometries_list)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing GeoJSON features. Check properties format. Error: {e}")

    if attributes_df.empty: return geometries_df
    if geometries_df.empty: return attributes_df
    return pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

def load_supply_utilization() -> pd.DataFrame:
    """Loads and cleans supply utilization/consumption data."""
    return _data_loader.load_csv(
        settings.supply_utilization_path, date_cols=['report_date']
    )

def load_program_outcomes() -> pd.DataFrame:
    """Loads programmatic outcomes data (e.g., linkage to care)."""
    # Assuming path is defined in an extended settings file, or fallback
    path = getattr(settings, "program_outcomes_path", Path("program_outcomes_synthetic.csv"))
    return _data_loader.load_csv(path, date_cols=['diagnosis_date', 'treatment_start_date', 'outcome_date'])

def load_contact_tracing() -> pd.DataFrame:
    """Loads contact tracing data for infectious diseases like TB."""
    path = getattr(settings, "contact_tracing_path", Path("contact_tracing_synthetic.csv"))
    return _data_loader.load_csv(path, date_cols=['index_patient_diagnosis_date', 'contact_date', 'evaluation_date'])

def load_ntd_mass_drug_admin() -> pd.DataFrame:
    """Loads NTD Mass Drug Administration (MDA) program data."""
    path = getattr(settings, "ntd_mda_path", Path("ntd_mda_synthetic.csv"))
    return _data_loader.load_csv(path, date_cols=['campaign_date'])

def load_ml_model(model_path: Path) -> Optional[Any]:
    """Loads a serialized machine learning model."""
    return _model_loader.load_ml_model(model_path)
