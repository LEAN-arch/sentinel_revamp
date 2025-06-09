# sentinel_project_root/config/settings.py
# Corrected version.

import logging
from pathlib import Path
from typing import List, Dict, Literal, Optional, Any

from pydantic import BaseModel, Field, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
settings_logger = logging.getLogger(__name__)

class AppConfig(BaseModel):
    name: str = "Sentinel Public Health Co-Pilot"
    version: str = "8.0.2 Final"
    organization_name: str = "Global Health Diagnostics Initiative"
    support_contact: str = "support@ghdi.org"

class DirectoryConfig(BaseModel):
    root: Path = PROJECT_ROOT
    assets: Path = root / "assets"
    data_sources: Path = root / "data_sources"
    ml_models: Path = root / "ml_models"

    @model_validator(mode='after')
    def create_directories(self) -> 'DirectoryConfig':
        for dir_path in [self.assets, self.data_sources, self.ml_models]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

class AnemiaThresholdConfig(BaseModel):
    severe: float = 8.0
    moderate: float = 11.0
    mild: float = 12.0

class ProgramTargetConfig(BaseModel):
    tb_case_detection_rate_pct: float = 85.0
    hiv_linkage_to_care_pct: float = 95.0

class ThresholdConfig(BaseModel):
    spo2_critical_low_pct: int = 90
    body_temp_high_fever_c: float = 39.0
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
    def plotly_colorway(self) -> List[str]:
        return ["#0D47A1", "#4CAF50", "#FFC107", "#F44336", "#9C27B0", "#009688"]

class MLModelConfig(BaseModel):
    risk_model_filename: str = "sentinel_risk_model_v1.joblib"
    risk_model_path: Optional[Path] = None
    risk_model_features: List[str] = ["age", "bmi", "is_smoker", "has_chronic_condition", "days_since_last_visit", "abnormal_vital_count"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='SENTINEL_', case_sensitive=False, env_nested_delimiter='__',
        env_file=f"{PROJECT_ROOT}/.env", extra='ignore'
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    random_seed: int = 42
    app: AppConfig = Field(default_factory=AppConfig)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    targets: ProgramTargetConfig = Field(default_factory=ProgramTargetConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    ml_models: MLModelConfig = Field(default_factory=MLModelConfig)
    mapbox_token: Optional[str] = None
    cache_ttl_seconds: int = 3600
    map_style: str = "carto-positron"
    health_records_path: Optional[Path] = None
    lab_results_path: Optional[Path] = None
    zone_attributes_path: Optional[Path] = None
    zone_geometries_path: Optional[Path] = None
    supply_utilization_path: Optional[Path] = None
    program_outcomes_path: Optional[Path] = None
    contact_tracing_path: Optional[Path] = None
    ntd_mda_path: Optional[Path] = None
    app_logo_large_path: Optional[Path] = None
    style_css_path: Optional[Path] = None
    
    key_diagnoses_for_action: List[str] = ['Tuberculosis', 'Malaria', 'HIV', 'Pneumonia', 'Anemia', 'Syphilis', 'Chlamydia']
    
    # --- THE FIX: This now correctly uses the TestTypeConfig Pydantic model ---
    key_test_types: Dict[str, TestTypeConfig] = {
        "GeneXpert": TestTypeConfig(disease_group="Tuberculosis", target_tat_days=1.0, display_name="TB GeneXpert", is_critical=True),
        "Sputum_AFB": TestTypeConfig(disease_group="Tuberculosis", target_tat_days=2.0, display_name="TB Smear Microscopy", is_critical=True),
        "HIV_Ag/Ab": TestTypeConfig(disease_group="HIV/STIs", target_tat_days=0.5, display_name="HIV 4th Gen Test", is_critical=True),
        "Malaria_RDT": TestTypeConfig(disease_group="Malaria", target_tat_days=0.5, display_name="Malaria RDT", is_critical=True),
        "Hemoglobin": TestTypeConfig(disease_group="NCDs/Nutrition", target_tat_days=0.2, display_name="Anemia (Hb)", is_critical=False),
        "Syphilis_RDT": TestTypeConfig(disease_group="HIV/STIs", target_tat_days=0.5, display_name="Syphilis RDT", is_critical=True),
    }

    @computed_field
    def app_footer_text(self) -> str:
        from datetime import datetime
        return f"© {datetime.now().year} {self.app.organization_name}. Advancing Diagnostics for All."
        
    @model_validator(mode='after')
    def assemble_paths(self) -> 'Settings':
        data_dir, asset_dir, ml_dir = self.directories.data_sources, self.directories.assets, self.directories.ml_models
        self.health_records_path = data_dir / "health_records_synthetic.csv"
        self.lab_results_path = data_dir / "lab_results_synthetic.csv"
        self.zone_attributes_path = data_dir / "zone_attributes.csv"
        self.zone_geometries_path = data_dir / "zone_geometries.geojson"
        self.supply_utilization_path = data_dir / "supply_utilization_synthetic.csv"
        self.program_outcomes_path = data_dir / "program_outcomes_synthetic.csv"
        self.contact_tracing_path = data_dir / "contact_tracing_synthetic.csv"
        self.ntd_mda_path = data_dir / "ntd_mda_synthetic.csv"
        self.app_logo_large_path = asset_dir / "sentinel_logo_large.png"
        self.style_css_path = asset_dir / "style.css"
        self.ml_models.risk_model_path = ml_dir / self.ml_models.risk_model_filename
        return self

try:
    settings = Settings()
    settings_logger.info(f"Settings loaded for '{settings.app.name}' v{settings.app.version}.")
except Exception as e:
    settings_logger.critical(f"FATAL: Could not initialize settings. Error: {e}", exc_info=True)
    raise
