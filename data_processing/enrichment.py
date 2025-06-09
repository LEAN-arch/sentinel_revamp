# sentinel_project_root/data_processing/enrichment.py
#
# PLATINUM STANDARD - Health Data Feature Engineering
# This module enriches the core health records DataFrame with new, calculated
# features, preparing it for advanced analytics and machine learning.

import logging
import pandas as pd
import numpy as np
from typing import Optional

# --- Core Application Imports ---
try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in enrichment.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


def enrich_health_records_with_features(df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the primary health records DataFrame with analytics-ready features.

    This function adds columns for:
    - Vital sign flags (e.g., high fever, low SpO2).
    - Composite scores (e.g., count of abnormal vitals).
    - Temporal features (e.g., days since last visit).
    - Features required for machine learning models.

    Args:
        df_health: The cleaned health records DataFrame from the loader.

    Returns:
        The enriched DataFrame with new feature columns.
    """
    if not isinstance(df_health, pd.DataFrame) or df_health.empty:
        logger.warning("Input DataFrame is empty. Skipping feature enrichment.")
        return pd.DataFrame()

    df = df_health.copy()
    num_records = len(df)
    logger.debug(f"Starting feature enrichment for {num_records} health records.")

    # --- 1. Vital Sign Flags & Categorization (Vectorized) ---
    # Create boolean flags for each abnormal vital sign.
    df['is_high_fever'] = df.get('body_temperature_celsius', 0) > settings.thresholds.body_temp_high_fever_c
    df['is_spo2_critical'] = df.get('spo2_percentage', 100) < settings.thresholds.spo2_critical_low_pct
    df['is_spo2_warning'] = (df.get('spo2_percentage', 100) >= settings.thresholds.spo2_critical_low_pct) & \
                             (df.get('spo2_percentage', 100) < settings.thresholds.spo2_warning_low_pct)

    # Composite feature: Count of abnormal vital signs for each encounter.
    # This is a powerful feature for ML models.
    vital_flags = ['is_high_fever', 'is_spo2_critical'] # Add more flags here as needed
    df['abnormal_vital_count'] = df[vital_flags].sum(axis=1)

    # --- 2. Temporal Features ---
    # Calculate days since the last encounter for each patient. This requires sorting.
    if 'encounter_date' in df.columns and 'patient_id' in df.columns:
        df.sort_values(by=['patient_id', 'encounter_date'], inplace=True)
        # Group by patient and shift the encounter date to get the previous date.
        df['days_since_last_visit'] = df.groupby('patient_id')['encounter_date'].diff().dt.days.fillna(0).astype(int)

    # --- 3. Demographic Features ---
    # Create age bands for stratified analysis.
    if 'age' in df.columns:
        age_bins = [0, 5, 17, 45, 65, np.inf]
        age_labels = ['Infant/Toddler (0-5)', 'Child/Adolescent (6-17)', 'Adult (18-45)', 'Middle-Aged (46-65)', 'Senior (65+)']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # --- 4. Mock/Placeholder Features for ML Model Compatibility ---
    # In a real-world scenario, these would come from the source data.
    # Here, we create them to ensure the ML model can be run for demonstration.
    # NOTE: These are synthetic and should be replaced with real data when available.
    if 'bmi' not in df.columns:
        # Generate synthetic BMI data centered around a realistic mean.
        df['bmi'] = np.random.normal(loc=24, scale=4, size=num_records).round(1)
        df['bmi'] = df['bmi'].clip(lower=15, upper=40)

    if 'is_smoker' not in df.columns:
        df['is_smoker'] = np.random.choice([True, False], size=num_records, p=[0.2, 0.8])

    if 'has_chronic_condition' not in df.columns:
        df['has_chronic_condition'] = np.random.choice([True, False], size=num_records, p=[0.3, 0.7])


    logger.info(f"Feature enrichment complete. Added {len(df.columns) - len(df_health.columns)} new columns.")
    return df
