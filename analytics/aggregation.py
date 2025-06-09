# sentinel_project_root/analytics/aggregation.py
# Final corrected version.

import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, Any

try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in aggregation.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

def calculate_kpi_statistics(
    current_period_series: pd.Series, previous_period_series: Optional[pd.Series] = None, higher_is_better: bool = False
) -> Dict[str, Any]:
    stats_result = {'current_mean': np.nan, 'current_ci': (np.nan, np.nan), 'delta_abs': np.nan, 'delta_pct': np.nan, 'p_value': np.nan, 'is_significant': False, 'is_positive_change': None}
    current_period_series = current_period_series.dropna()
    if current_period_series.empty: return stats_result
    stats_result['current_mean'] = current_period_series.mean()
    if len(current_period_series) > 1:
        ci = stats.sem(current_period_series) * stats.t.ppf((1 + 0.95) / 2., len(current_period_series)-1)
        stats_result['current_ci'] = (stats_result['current_mean'] - ci, stats_result['current_mean'] + ci)
    if previous_period_series is not None:
        previous_period_series_clean = previous_period_series.dropna()
        if not previous_period_series_clean.empty:
            prev_mean = previous_period_series_clean.mean()
            stats_result['delta_abs'] = stats_result['current_mean'] - prev_mean
            if prev_mean != 0:
                stats_result['delta_pct'] = (stats_result['delta_abs'] / prev_mean)
            if len(current_period_series) > 1 and len(previous_period_series_clean) > 1:
                _, p_value = stats.ttest_ind(current_period_series, previous_period_series_clean, equal_var=False)
                stats_result['p_value'] = p_value
                stats_result['is_significant'] = p_value < 0.05
            stats_result['is_positive_change'] = (stats_result['delta_abs'] > 0) if higher_is_better else (stats_result['delta_abs'] < 0)
    return stats_result

def aggregate_zonal_stats(
    df_health: pd.DataFrame, df_labs: pd.DataFrame, df_program: pd.DataFrame, df_zones: pd.DataFrame
) -> pd.DataFrame:
    if df_zones.empty: return pd.DataFrame()

    df_merged = df_zones.copy()

    # Create a health context dataframe with only necessary columns for joining
    health_context = pd.DataFrame()
    if 'encounter_id' in df_health.columns and 'zone_id' in df_health.columns:
        health_context = df_health[['encounter_id', 'zone_id']].drop_duplicates()

    if not df_health.empty:
        health_agg = df_health.groupby('zone_id').agg(
            avg_risk_score=('ai_risk_score', 'mean'),
            total_encounters=('encounter_id', 'nunique')).reset_index()
        df_merged = pd.merge(df_merged, health_agg, on='zone_id', how='left')

    if not df_labs.empty and not health_context.empty:
        df_labs_with_zone = pd.merge(df_labs, health_context, on='encounter_id', how='left').dropna(subset=['zone_id'])
        
        if not df_labs_with_zone.empty:
            lab_agg = df_labs_with_zone.groupby('zone_id').agg(
                avg_tat_days=('turn_around_time_days', 'mean'),
                total_tests_processed=('test_id', 'nunique'),
                rejection_rate=('is_rejected', 'mean')).reset_index()
            lab_agg['rejection_rate'] = (lab_agg['rejection_rate'] * 100).fillna(0)

            # --- THE FIX: Create is_positive flag before aggregation ---
            df_labs_with_zone['is_positive'] = (df_labs_with_zone['test_result'] == 'Positive')

            for test_key in settings.key_test_types.keys():
                positivity_df = df_labs_with_zone[df_labs_with_zone['test_name'] == test_key]
                if not positivity_df.empty:
                    positivity_agg = positivity_df.groupby('zone_id')['is_positive'].mean().reset_index()
                    positivity_agg.rename(columns={'is_positive': f'positivity_{test_key.lower()}'}, inplace=True)
                    lab_agg = pd.merge(lab_agg, positivity_agg, on='zone_id', how='left')
            
            df_merged = pd.merge(df_merged, lab_agg, on='zone_id', how='left')

    if not df_program.empty:
        # Assuming program data is joined similarly if it doesn't have zone_id
        df_program_with_zone = pd.merge(df_program, df_health[['patient_id', 'zone_id']].drop_duplicates(subset=['patient_id']), on='patient_id', how='left').dropna(subset=['zone_id'])
        if not df_program_with_zone.empty:
            program_agg = df_program_with_zone.groupby('zone_id').agg(
                hiv_linkage_rate=('is_linked_to_care_hiv', 'mean'),
                tb_treatment_success_rate=('is_treatment_success_tb', 'mean')).reset_index()
            program_agg['hiv_linkage_rate'] = (program_agg['hiv_linkage_rate'] * 100).fillna(0)
            program_agg['tb_treatment_success_rate'] = (program_agg['tb_treatment_success_rate'] * 100).fillna(0)
            df_merged = pd.merge(df_merged, program_agg, on='zone_id', how='left')

    numeric_cols = df_merged.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col not in df_zones.columns:
            df_merged[col] = df_merged[col].fillna(0)
            
    return df_merged

def aggregate_district_stats(df_zonal_agg: pd.DataFrame) -> Dict[str, Any]:
    if df_zonal_agg.empty: return {}
    total_population = df_zonal_agg['population'].sum()
    district_summary = {'total_population': total_population, 'total_zones': df_zonal_agg['zone_id'].nunique()}
    weighted_metrics = {'avg_risk_score': 'population', 'hiv_linkage_rate': 'population', 'rejection_rate': 'total_tests_processed'}
    for metric, weight_col in weighted_metrics.items():
        if metric in df_zonal_agg.columns and weight_col in df_zonal_agg.columns:
            weights = df_zonal_agg[weight_col].clip(lower=1)
            district_summary[f'population_weighted_{metric}'] = np.average(df_zonal_agg[metric].fillna(0), weights=weights)
    count_cols = ['total_encounters', 'total_tests_processed']
    for col in count_cols:
        if col in df_zonal_agg.columns:
            district_summary[col] = df_zonal_agg[col].sum()
    return district_summary

def aggregate_program_kpis(df_zonal_agg: pd.DataFrame) -> Dict[str, Any]:
    if df_zonal_agg.empty: return {}
    kpis = {}
    if 'hiv_linkage_rate' in df_zonal_agg.columns and 'population' in df_zonal_agg.columns:
        weights = df_zonal_agg['population'].clip(lower=1)
        kpis['hiv_linkage_to_care'] = np.average(df_zonal_agg['hiv_linkage_rate'].fillna(0), weights=weights)
    return kpis
