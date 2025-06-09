# sentinel_project_root/data_processing/enrichment.py
# Final corrected version.

import logging
import pandas as pd
import numpy as np

try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in enrichment.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


def enrich_lab_results_with_features(df_labs: pd.DataFrame) -> pd.DataFrame:
    """Enriches lab results with TAT, clinical severity, and programmatic flags."""
    if not isinstance(df_labs, pd.DataFrame) or df_labs.empty:
        return pd.DataFrame()

    df = df_labs.copy()

    df['result_date'] = pd.to_datetime(df['result_date'], errors='coerce')
    df['sample_collection_date'] = pd.to_datetime(df['sample_collection_date'], errors='coerce')

    df['turn_around_time_days'] = (df['result_date'] - df['sample_collection_date']).dt.total_seconds() / (24 * 3600)
    df['turn_around_time_days'] = df['turn_around_time_days'].clip(lower=0)

    df['is_rejected'] = df['is_rejected'].fillna(False).astype(bool)
    
    conditions_status = [df['is_rejected'] == True, df['result_date'].notna()]
    choices_status = ['Rejected', 'Completed']
    df['test_status'] = np.select(conditions_status, choices_status, default='Pending')
    
    critical_test_map = {test_name: props.is_critical for test_name, props in settings.key_test_types.items()}
    df['is_critical'] = df['test_name'].map(critical_test_map).fillna(False)
    
    if 'test_name' in df.columns and 'result_value' in df.columns:
        anemia_mask = df['test_name'].str.contains("Hemoglobin", case=False, na=False)
        if anemia_mask.any():
            hb_values = pd.to_numeric(df.loc[anemia_mask, 'result_value'], errors='coerce')
            severity_conditions = [
                hb_values < settings.thresholds.anemia.severe,
                hb_values < settings.thresholds.anemia.moderate,
                hb_values < settings.thresholds.anemia.mild
            ]
            severity_choices = ['Severe', 'Moderate', 'Mild']
            df.loc[anemia_mask, 'anemia_severity'] = np.select(severity_conditions, severity_choices, default='Normal')

    return df


def enrich_program_outcomes_with_features(df_program: pd.DataFrame) -> pd.DataFrame:
    """Enriches program outcomes data with boolean flags for aggregation."""
    if not isinstance(df_program, pd.DataFrame) or df_program.empty:
        return pd.DataFrame()
    
    df = df_program.copy()

    # --- THE FIX: Create the boolean flags here ---
    df['is_linked_to_care_hiv'] = (df['program_name'] == 'HIV Care') & (pd.to_datetime(df['treatment_start_date'], errors='coerce').notna())
    df['is_treatment_success_tb'] = (df['program_name'] == 'Tuberculosis Care') & (df['outcome'] == 'Treatment Completed')
    
    return df


def enrich_health_records_with_features(df_health: pd.DataFrame) -> pd.DataFrame:
    """Enriches the primary health records DataFrame with ML-ready features."""
    if not isinstance(df_health, pd.DataFrame) or df_health.empty:
        return pd.DataFrame()

    df = df_health.copy()
    num_records = len(df)

    df['is_high_fever'] = df.get('body_temperature_celsius', 0) > settings.thresholds.body_temp_high_fever_c
    df['is_spo2_critical'] = df.get('spo2_percentage', 100) < settings.thresholds.spo2_critical_low_pct
    df['abnormal_vital_count'] = df[['is_high_fever', 'is_spo2_critical']].sum(axis=1)

    if 'encounter_date' in df.columns and 'patient_id' in df.columns:
        df = df.sort_values(by=['patient_id', 'encounter_date'])
        df['days_since_last_visit'] = df.groupby('patient_id')['encounter_date'].diff().dt.days.fillna(0).astype(int)

    if 'age' in df.columns:
        age_bins = [0, 5, 17, 45, 65, np.inf]
        age_labels = ['Infant/Toddler (0-5)', 'Child/Adolescent (6-17)', 'Adult (18-45)', 'Middle-Aged (46-65)', 'Senior (65+)']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    if 'bmi' not in df.columns:
        df['bmi'] = np.random.normal(loc=22, scale=4, size=num_records).round(1).clip(15, 40)
    if 'is_smoker' not in df.columns:
        df['is_smoker'] = np.random.choice([True, False], size=num_records, p=[0.15, 0.85])
    if 'has_chronic_condition' not in df.columns:
        chronic_conditions_list = ['HIV', 'Tuberculosis', 'Diabetes']
        df['has_chronic_condition'] = df['primary_diagnosis'].astype(str).isin(chronic_conditions_list)

    return df
