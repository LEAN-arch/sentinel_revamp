# sentinel_project_root/config/settings.py
#
# PLATINUM STANDARD - Centralized Application Configuration
# This file defines the entire application's configuration using Pydantic for
# robust validation, type safety, and clear structure. It loads settings
# from environment variables or a .env file, making it secure and deployable.

import logging
from pathlib import Path
from typing import List, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Define Project Root ---
# This is the single source of truth for all file paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Logger for Settings Module ---
settings_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. NESTED CONFIGURATION MODELS
# Breaking the configuration into logical, smaller models improves readability
# and maintainability.
# -----------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Core application metadata and operational settings."""
    name: str = "Sentinel Health Co-Pilot"
    version: str = "5.0.0 Platinum"
    organization_name: str = "Global Health Intelligence"
    support_contact: str = "support@global-health-intelligence.org"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    random_seed: int = 42

class DirectoryConfig(BaseModel):
    """Manages all key directory paths, ensuring they exist."""
    root: Path = PROJECT_ROOT
    assets: Path = root / "assets"
    data_sources: Path = root / "data_sources"
    ml_models: Path = root / "ml_models"
    logs: Path = root / "logs"

    @model_validator(mode='after')
    def create_directories(self) -> 'DirectoryConfig':
        """Ensure all configured directories exist on initialization."""
        for dir_path in [self.assets, self.data_sources, self.ml_models, self.logs]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

class ThresholdConfig(BaseModel):
    """Defines critical operational and clinical thresholds."""
    spo2_critical_low_pct: int = 90
    spo2_warning_low_pct: int = 94
    body_temp_high_fever_c: float = 39.0
    risk_score_low: int = 40
    risk_score_moderate: int = 60
    risk_score_high: int = 75
    supply_critical_days: int = 7
    supply_warning_days: int = 14
    test_tat_overdue_buffer_days: int = 2
    lab_rejection_rate_target_pct: float = 3.0
    district_risk_score_threshold: float = 65.0 # High risk threshold for a zone

class TestTypeConfig(BaseModel):
    """Configuration for a specific laboratory test type."""
    disease_group: str
    target_tat_days: float
    display_name: str
    is_critical: bool

class ThemeConfig(BaseModel):
    """Centralizes all color and theme information for UI and plots."""
    primary: str = "#0D47A1"
    background: str = "#F0F2F6"
    secondary_background: str = "#FFFFFF"
    text: str = "#263238"

    risk_high: str = "#D32F2F"
    risk_moderate: str = "#FFA000"
    risk_low: str = "#388E3C"

    positive_delta: str = "#2E7D32"
    negative_delta: str = "#C62828"
    neutral_delta: str = "#616161"

    @computed_field
    @property
    def plotly_colorway(self) -> List[str]:
        """Defines the default categorical color sequence for Plotly charts."""
        return ["#0D47A1", "#4CAF50", "#FFC107", "#F44336", "#2196F3", "#9C27B0"]

class MLModelConfig(BaseModel):
    """Configuration for machine learning models and associated files."""
    risk_model_path: Path = DirectoryConfig().ml_models / "sentinel_risk_model_v1.joblib"
    risk_model_features: List[str] = [
        "age", "bmi", "is_smoker", "has_chronic_condition",
        "days_since_last_visit", "abnormal_vital_count"
    ]

# -----------------------------------------------------------------------------
# 2. MAIN SETTINGS CLASS
# This class aggregates all the smaller models and loads settings from the
# environment (e.g., .env file or system variables).
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Main settings class for the Sentinel application.
    Aggregates all configuration models and loads from environment variables.
    """
    # Configure Pydantic to load from a .env file and use a prefix.
    model_config = SettingsConfigDict(
        env_prefix='SENTINEL_',
        case_sensitive=False,
        env_nested_delimiter='__',
        env_file=f"{PROJECT_ROOT}/.env",
        extra='ignore'
    )

    # --- Nested Configuration Models ---
    app: AppConfig = Field(default_factory=AppConfig)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    ml_models: MLModelConfig = Field(default_factory=MLModelConfig)

    # --- API and Secret Keys ---
    # Will automatically load SENTINEL_MAPBOX_TOKEN from environment/.env
    mapbox_token: Optional[str] = None

    # --- Data Source Paths (relative to directories.data_sources) ---
    health_records_path: Path = directories.data_sources / "health_records_synthetic.csv"
    iot_records_path: Path = directories.data_sources / "iot_clinic_env_synthetic.csv"
    zone_attributes_path: Path = directories.data_sources / "zone_attributes.csv"
    zone_geometries_path: Path = directories.data_sources / "zone_geometries.geojson"
    supply_utilization_path: Path = directories.data_sources / "supply_utilization_synthetic.csv"
    lab_results_path: Path = directories.data_sources / "lab_results_synthetic.csv"


    # --- Asset Paths (relative to directories.assets) ---
    app_logo_small_path: Path = directories.assets / "sentinel_logo_small.png"
    app_logo_large_path: Path = directories.assets / "sentinel_logo_large.png"
    style_css_path: Path = directories.assets / "style.css"
    escalation_protocols_path: Path = directories.assets / "escalation_protocols.json"

    # --- Semantic & Operational Definitions ---
    key_diagnoses_for_action: List[str] = [
        'Malaria', 'Pneumonia', 'Acute Diarrheal Disease', 'Tuberculosis (TB)', 'Severe Malnutrition'
    ]
    key_supply_items_for_forecast: List[str] = [
        'Paracetamol', 'Amoxicillin', 'ORS Packet', 'Metformin', 'Gloves', 'Syringes', 'Malaria RDT'
    ]
    key_test_types: Dict[str, TestTypeConfig] = {
        "Malaria_RDT": TestTypeConfig(disease_group="Infectious", target_tat_days=0.5, display_name="Malaria RDT", is_critical=True),
        "CBC": TestTypeConfig(disease_group="General", target_tat_days=1.0, display_name="CBC", is_critical=True),
        "Sputum_AFB": TestTypeConfig(disease_group="Infectious", target_tat_days=2.0, display_name="TB Sputum Test", is_critical=True),
        "COVID-19_Ag": TestTypeConfig(disease_group="Respiratory", target_tat_days=0.25, display_name="COVID-19 Ag", is_critical=True),
    }

    # --- Web Dashboard & Visualization Settings ---
    cache_ttl_seconds: int = 3600
    map_default_zoom: int = 10
    map_default_center_lat: float = -1.286389 # Nairobi
    map_default_center_lon: float = 36.817223 # Nairobi
    map_style: str = "carto-positron"

    @computed_field
    @property
    def app_footer_text(self) -> str:
        """Generates the application footer text dynamically."""
        from datetime import datetime
        return f"© {datetime.now().year} {self.app.organization_name}. All Rights Reserved."

# -----------------------------------------------------------------------------
# 3. SINGLETON INSTANCE
# Create a single, globally-accessible instance of the settings.
# -----------------------------------------------------------------------------

try:
    settings = Settings()
    # Log successful loading and critical details.
    settings_logger.info(
        f"Settings loaded for '{settings.app.name}' v{settings.app.version}. "
        f"LOG_LEVEL={settings.app.log_level}. PROJECT_ROOT='{PROJECT_ROOT}'"
    )
    if not settings.mapbox_token:
        settings_logger.warning(
            "Mapbox token not found. Set 'SENTINEL_MAPBOX_TOKEN' in your environment or .env file. "
            "Maps will use a basic, open-source style."
        )
except Exception as e:
    settings_logger.critical(f"FATAL: Could not initialize application settings. Error: {e}", exc_info=True)
    # Raising the error is critical because the app cannot run without settings.
    raise
