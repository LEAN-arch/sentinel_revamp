# sentinel_project_root/data_processing/enrichment.py
#
# PLATINUM STANDARD - Health Data Feature Engineering (V2 - Public Health Mission Upgrade)
# This module is upgraded to create sophisticated clinical severity and programmatic
# features, crucial for detailed public health surveillance and monitoring.

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


def enrich_lab_results_with_features(df_labs: pd.DataFrame, df_program: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the primary lab results DataFrame with analytics-ready features.

    This function adds columns for:
    - Test Turnaround Time (TAT).
    - Clinical severity levels (e.g., Anemia severity).
    - Programmatic flags (e.g., requires confirmatory test, not linked to care).

    Args:
        df_labs: The cleaned lab results DataFrame from the loader.
        df_program: The program outcomes DataFrame for linkage checks.

    Returns:
        The enriched lab results DataFrame with new feature columns.
    """
    if not isinstance(df_labs, pd.DataFrame) or df_labs.empty:
        logger.warning("Lab results DataFrame is empty. Skipping feature enrichment.")
        return pd.DataFrame()

    df = df_labs.copy()

    # --- 1. Calculate Test Turnaround Time (TAT) ---
    if 'result_date' in df.columns and 'sample_collection_date' in df.columns:
        df['turn_around_time_days'] = (df['result_date'] - df['sample_collection_date']).dt.total_seconds() / (24 * 3600)
        df['turn_around_time_days'] = df['turn_around_time_days'].clip(lower=0) # No negative TAT

    # --- 2. Classify Clinical Severity ---
    if 'test_name' in df.columns and 'result_value' in df.columns:
        # Vectorized Anemia Severity Classification based on WHO thresholds
        anemia_mask = df['test_name'].str.contains("Hemoglobin", case=False, na=False)
        hb_values = pd.to_numeric(df.loc[anemia_mask, 'result_value'], errors='coerce')

        conditions = [
            hb_values < settings.thresholds.anemia.severe,
            hb_values < settings.thresholds.anemia.moderate,
            hb_values < settings.thresholds.anemia.mild
        ]
        choices = ['Severe', 'Moderate', 'Mild']
        df.loc[anemia_mask, 'anemia_severity'] = np.select(conditions, choices, default='Normal')

    # --- 3. Generate Programmatic Flags ---
    # Example: Identify positive HIV tests that are not yet linked to care
    if 'test_name' in df.columns and 'test_result' in df.columns and not df_program.empty:
        positive_hiv_mask = (df['test_name'].str.contains("HIV", na=False)) & (df['test_result'] == 'Positive')
        linked_patients = df_program[df_program['program_name'] == 'HIV Care']['patient_id'].unique()
        df['is_hiv_positive_unlinked'] = positive_hiv_mask & (~df['patient_id'].isin(linked_patients))

    # Example: Identify TB cases from microscopy that need a confirmatory test
    sputum_positive_mask = (df['test_name'].str.contains("Sputum|Smear", case=False, na=False)) & (df['test_result'] == 'Positive')
    confirmed_patients = df[df['test_name'].str.contains("GeneXpert", case=False, na=False)]['patient_id'].unique()
    df['is_tb_positive_unconfirmed'] = sputum_positive_mask & (~df['patient_id'].isin(confirmed_patients))

    logger.info(f"Lab results enriched. Added {len(df.columns) - len(df_labs.columns)} new feature columns.")
    return df


def enrich_health_records_with_features(df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the primary health records DataFrame with ML-ready features.
    (Retained for patient-level risk modeling).
    """
    if not isinstance(df_health, pd.DataFrame) or df_health.empty:
        logger.warning("Health records DataFrame is empty. Skipping enrichment.")
        return pd.DataFrame()

    df = df_health.copy()
    num_records = len(df)
    logger.debug(f"Starting feature enrichment for {num_records} health records.")

    # --- Vectorized Vital Sign Flags ---
    df['is_high_fever'] = df.get('body_temperature_celsius', 0) > settings.thresholds.body_temp_high_fever_c
    df['is_spo2_critical'] = df.get('spo2_percentage', 100) < settings.thresholds.spo2_critical_low_pct
    df['abnormal_vital_count'] = df[['is_high_fever', 'is_spo2_critical']].sum(axis=1)

    # --- Temporal Features ---
    if 'encounter_date' in df.columns and 'patient_id' in df.columns:
        df.sort_values(by=['patient_id', 'encounter_date'], inplace=True)
        df['days_since_last_visit'] = df.groupby('patient_id')['encounter_date'].diff().dt.days.fillna(0).astype(int)

    # --- Demographic Features ---
    if 'age' in df.columns:
        age_bins = [0, 5, 17, 45, 65, np.inf]
        age_labels = ['Infant/Toddler (0-5)', 'Child/Adolescent (6-17)', 'Adult (18-45)', 'Middle-Aged (46-65)', 'Senior (65+)']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # --- Mock/Placeholder Features for ML Model ---
    # These ensure compatibility with the pre-trained ML model.
    if 'bmi' not in df.columns:
        df['bmi'] = np.random.normal(loc=22, scale=4, size=num_records).round(1).clip(15, 40)
    if 'is_smoker' not in df.columns:
        df['is_smoker'] = np.random.choice([True, False], size=num_records, p=[0.15, 0.85])
    if 'has_chronic_condition' not in df.columns:
        # More realistic for LMIC setting; links to specific priority diseases
        chronic_conditions_list = ['HIV', 'Tuberculosis', 'Diabetes'] #subset for demonstration
        df['has_chronic_condition'] = df['primary_diagnosis'].isin(chronic_conditions_list)

    logger.info(f"Health records enriched. Added {len(df.columns) - len(df_health.columns)} new columns.")
    return df
