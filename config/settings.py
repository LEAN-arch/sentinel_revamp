# sentinel_project_root/config/settings.py
#
# PLATINUM STANDARD - Centralized Application Configuration (V2 - Public Health Mission Upgrade)
# This file is enhanced to support a wide range of infectious diseases and public
# health programs, turning the system into a true epidemiological tool.

import logging
from pathlib import Path
from typing import List, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
settings_logger = logging.getLogger(__name__)

# --- Nested Configuration Models ---

class AppConfig(BaseModel):
    name: str = "Sentinel Public Health Co-Pilot"
    version: str = "6.0.0 Platinum"
    organization_name: str = "Global Health Diagnostics Initiative"
    support_contact: str = "support@ghdi.org"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s"
    random_seed: int = 42

class DirectoryConfig(BaseModel):
    root: Path = PROJECT_ROOT
    assets: Path = root / "assets"
    data_sources: Path = root / "data_sources"
    ml_models: Path = root / "ml_models"
    logs: Path = root / "logs"

    @model_validator(mode='after')
    def create_directories(self) -> 'DirectoryConfig':
        for dir_path in [self.assets, self.data_sources, self.ml_models, self.logs]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

class AnemiaThresholdConfig(BaseModel):
    """WHO-based anemia thresholds by Hemoglobin (g/dL) for non-pregnant adults."""
    severe: float = 8.0
    moderate: float = 11.0
    mild: float = 12.0

class ProgramTargetConfig(BaseModel):
    """Defines strategic public health program targets."""
    tb_case_detection_rate_pct: float = 85.0
    hiv_linkage_to_care_pct: float = 95.0
    malaria_rdt_confirmation_rate_pct: float = 98.0
    anemia_screening_coverage_pct_at_risk: float = 75.0

class ThresholdConfig(BaseModel):
    spo2_critical_low_pct: int = 90
    risk_score_high: int = 75
    supply_critical_days: int = 10
    lab_rejection_rate_target_pct: float = 3.0
    district_risk_score_threshold: float = 65.0
    anemia: AnemiaThresholdConfig = Field(default_factory=AnemiaThresholdConfig)

class TestTypeConfig(BaseModel):
    disease_group: str
    target_tat_days: float
    display_name: str
    is_critical: bool

class ThemeConfig(BaseModel):
    primary: str = "#0D47A1"
    background: str = "#F0F2F6"
    secondary_background: str = "#FFFFFF"
    text: str = "#263238"
    risk_high: str = "#D32F2F"
    risk_moderate: str = "#FFA000"
    risk_low: str = "#388E3C"

    @computed_field
    @property
    def plotly_colorway(self) -> List[str]:
        return ["#0D47A1", "#4CAF50", "#FFC107", "#F44336", "#9C27B0", "#009688"]

class MLModelConfig(BaseModel):
    risk_model_path: Path = DirectoryConfig().ml_models / "sentinel_risk_model_v1.joblib"
    risk_model_features: List[str] = ["age", "bmi", "is_smoker", "has_chronic_condition", "days_since_last_visit", "abnormal_vital_count"]

# --- Main Settings Class ---

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='SENTINEL_',
        case_sensitive=False,
        env_nested_delimiter='__',
        env_file=f"{PROJECT_ROOT}/.env",
        extra='ignore'
    )

    app: AppConfig = Field(default_factory=AppConfig)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    targets: ProgramTargetConfig = Field(default_factory=ProgramTargetConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    ml_models: MLModelConfig = Field(default_factory=MLModelConfig)

    mapbox_token: Optional[str] = None

    health_records_path: Path = directories.data_sources / "health_records_synthetic.csv"
    lab_results_path: Path = directories.data_sources / "lab_results_synthetic.csv"
    zone_attributes_path: Path = directories.data_sources / "zone_attributes.csv"
    zone_geometries_path: Path = directories.data_sources / "zone_geometries.geojson"
    supply_utilization_path: Path = directories.data_sources / "supply_utilization_synthetic.csv"

    app_logo_large_path: Path = directories.assets / "sentinel_logo_large.png"
    style_css_path: Path = directories.assets / "style.css"

    key_diagnoses_for_action: List[str] = [
        'Tuberculosis', 'Malaria', 'HIV', 'Pneumonia', 'Anemia', 'Syphilis', 'Chlamydia'
    ]
    key_supply_items_for_forecast: List[str] = [
        'Amoxicillin', 'ORS Packet', 'Gloves', 'Syringes', 'GeneXpert Cartridge',
        'Malaria RDT Kit', 'HIV 1/2 Ag/Ab Combo Test', 'Hemoglobin Cuvette'
    ]
    key_test_types: Dict[str, TestTypeConfig] = {
        "GeneXpert": TestTypeConfig(disease_group="Tuberculosis", target_tat_days=1.0, display_name="TB GeneXpert", is_critical=True),
        "Sputum_AFB": TestTypeConfig(disease_group="Tuberculosis", target_tat_days=2.0, display_name="TB Smear Microscopy", is_critical=True),
        "HIV_Ag/Ab": TestTypeConfig(disease_group="HIV/STIs", target_tat_days=0.5, display_name="HIV 4th Gen Test", is_critical=True),
        "Malaria_RDT": TestTypeConfig(disease_group="Malaria", target_tat_days=0.5, display_name="Malaria RDT", is_critical=True),
        "Hemoglobin": TestTypeConfig(disease_group="NCDs/Nutrition", target_tat_days=0.2, display_name="Anemia (Hb)", is_critical=False),
        "Syphilis_RDT": TestTypeConfig(disease_group="HIV/STIs", target_tat_days=0.5, display_name="Syphilis RDT", is_critical=True),
    }

    cache_ttl_seconds: int = 3600
    map_default_zoom: int = 5
    map_default_center_lat: float = 0.0236  # Central Africa
    map_default_center_lon: float = 15.8277 # Central Africa
    map_style: str = "carto-positron"

    @computed_field
    @property
    def app_footer_text(self) -> str:
        from datetime import datetime
        return f"© {datetime.now().year} {self.app.organization_name}. Advancing Diagnostics for All."

# --- Singleton Instance ---
try:
    settings = Settings()
    settings_logger.info(f"Settings loaded for Public Health mission: '{settings.app.name}' v{settings.app.version}.")
except Exception as e:
    settings_logger.critical(f"FATAL: Could not initialize settings for Public Health mission. Error: {e}", exc_info=True)
    raise
