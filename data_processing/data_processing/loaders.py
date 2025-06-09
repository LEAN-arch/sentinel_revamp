# sentinel_project_root/data_processing/loaders.py
#
# PLATINUM STANDARD - Unified Data Loading Engine
# This module provides a single, robust, and configuration-driven class
# for loading all data sources (CSVs, GeoJSON, ML models, etc.) into the
# Sentinel application. It ensures data is clean and typed on load.

import logging
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Any, Dict

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

        Args:
            file_path: The path to the CSV file, relative to the data source directory.
            date_cols: A list of columns to be parsed as dates.

        Returns:
            A cleaned and standardized pandas DataFrame.
        """
        full_path = self._get_path(file_path)
        log_ctx = f"CSV({full_path.name})"
        logger.debug(f"[{log_ctx}] Attempting to load data from {full_path}")

        if not full_path.exists():
            logger.error(f"[{log_ctx}] File not found. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            df = pd.read_csv(full_path, low_memory=False)
            if df.empty:
                logger.warning(f"[{log_ctx}] File is empty.")
                return pd.DataFrame()

            # PLATINUM STANDARD: Automatically apply cleaning pipeline on load.
            pipeline = DataPipeline(df).clean_column_names()
            if date_cols:
                pipeline.convert_date_columns(date_cols)

            df_processed = pipeline.get_df()
            logger.info(f"[{log_ctx}] Successfully loaded and cleaned {len(df_processed)} records.")
            return df_processed

        except Exception as e:
            logger.critical(f"[{log_ctx}] CRITICAL ERROR loading or processing file: {e}", exc_info=True)
            return pd.DataFrame()

    def load_geojson(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Loads a GeoJSON file with robust error handling.

        Args:
            file_path: The path to the GeoJSON file.

        Returns:
            A dictionary representing the GeoJSON data, or None on failure.
        """
        full_path = self._get_path(file_path)
        return robust_json_load(full_path, "GeoJSON")

    def load_ml_model(self, model_path: Path) -> Optional[Any]:
        """
        Loads a serialized machine learning model (e.g., scikit-learn pipeline)
        from a .joblib or .pkl file.

        Args:
            model_path: The path to the serialized model file.

        Returns:
            The deserialized Python object (e.g., a scikit-learn model), or None.
        """
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

# --- Singleton Instance and Public API Functions ---
# This pattern provides a single, globally available instance of the loader,
# while the functions below offer a clean, semantic API to the rest of the app.
_data_loader = DataLoader(settings.directories.data_sources)
_asset_loader = DataLoader(settings.directories.assets)
_model_loader = DataLoader(settings.directories.ml_models)


def load_health_records() -> pd.DataFrame:
    """Loads and cleans the primary health records dataset."""
    return _data_loader.load_csv(
        settings.health_records_path,
        date_cols=['encounter_date']
    )

def load_iot_records() -> pd.DataFrame:
    """Loads and cleans IoT environmental records."""
    return _data_loader.load_csv(
        settings.iot_records_path,
        date_cols=['timestamp']
    )

def load_lab_results() -> pd.DataFrame:
    """Loads and cleans laboratory results data."""
    return _data_loader.load_csv(
        settings.lab_results_path,
        date_cols=['sample_collection_date', 'result_date']
    )

def load_supply_utilization() -> pd.DataFrame:
    """Loads and cleans supply utilization/consumption data."""
    return _data_loader.load_csv(
        settings.supply_utilization_path,
        date_cols=['report_date']
    )

def load_zone_data() -> pd.DataFrame:
    """Loads and merges zone attribute and geometry data."""
    attributes_df = _data_loader.load_csv(settings.zone_attributes_path)
    if 'zone_id' in attributes_df.columns:
        attributes_df['zone_id'] = attributes_df['zone_id'].astype(str)

    geo_data = _data_loader.load_geojson(settings.zone_geometries_path)
    geometries_df = pd.DataFrame()

    if geo_data and isinstance(geo_data.get('features'), list):
        try:
            geometries_list = [
                {
                    "zone_id": str(feat["properties"]["zone_id"]),
                    "geometry": feat["geometry"]
                }
                for feat in geo_data["features"]
                if "properties" in feat and "zone_id" in feat["properties"] and "geometry" in feat
            ]
            geometries_df = pd.DataFrame(geometries_list)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing GeoJSON features. Check properties format. Error: {e}")

    if attributes_df.empty: return geometries_df
    if geometries_df.empty: return attributes_df

    return pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

def load_ml_model(model_path: Path) -> Optional[Any]:
    """Loads a serialized machine learning model."""
    return _model_loader.load_ml_model(model_path)

def load_json_asset(asset_path: Path) -> Optional[Dict[str, Any]]:
    """Loads a generic JSON asset file."""
    return robust_json_load(_asset_loader.base_dir / asset_path, "JSON Asset")
