# sentinel_project_root/analytics/aggregation.py
#
# PLATINUM STANDARD - Public Health Statistics Engine (V2.2 - Re-validated)
# This module is re-validated to ensure full compatibility with the corrected
# settings, enabling robust calculation of all programmatic and epi KPIs.

import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, Any

# --- Core Application Imports ---
try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in aggregation.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


def calculate_kpi_statistics(
    current_period_series: pd.Series,
    previous_period_series: Optional[pd.Series] = None,
    higher_is_better: bool = False
) -> Dict[str, Any]:
    """Calculates key statistics for a KPI, including confidence intervals and t-test for significance."""
    stats_result = {'current_mean': np.nan, 'current_ci': (np.nan, np.nan), 'delta_abs': np.nan,
                    'delta_pct': np.nan, 'p_value': np.nan, 'is_significant': False,
                    'is_positive_change': None}
    current_period_series = current_period_series.dropna()
    if current_period_series.empty:
        return stats_result
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
    """Aggregates comprehensive health, lab, and program data to a zonal level."""
    if df_zones.empty:
        return pd.DataFrame()
    logger.debug("Aggregating comprehensive health data to zonal level.")

    # Start with the base zone data
    df_merged = df_zones.copy()

    # 1. Aggregate health record stats
    if not df_health.empty:
        health_agg = df_health.groupby('zone_id').agg(
            avg_risk_score=('ai_risk_score', 'mean'),
            total_encounters=('encounter_id', 'nunique')
        ).reset_index()
        df_merged = pd.merge(df_merged, health_agg, on='zone_id', how='left')

    # 2. Aggregate lab stats
    if not df_labs.empty:
        lab_agg = df_labs.groupby('zone_id').agg(
            avg_tat_days=('turn_around_time_days', 'mean'),
            total_tests_processed=('test_id', 'nunique'),
            rejection_rate=('is_rejected', 'mean')
        ).reset_index()
        lab_agg['rejection_rate'] = (lab_agg['rejection_rate'] * 100).fillna(0)

        # Disease-specific positivity rates
        for test_key, test_config in settings.key_test_types.items():
            positivity_df = df_labs[df_labs['test_name'] == test_key]
            if not positivity_df.empty:
                positivity_agg = positivity_df.groupby('zone_id')['is_positive'].mean().reset_index()
                positivity_agg.rename(columns={'is_positive': f'positivity_{test_key.lower()}'}, inplace=True)
                lab_agg = pd.merge(lab_agg, positivity_agg, on='zone_id', how='left')
        df_merged = pd.merge(df_merged, lab_agg, on='zone_id', how='left')
    
    # 3. Aggregate program stats
    if not df_program.empty:
        program_agg = df_program.groupby('zone_id').agg(
            hiv_linkage_rate=('is_linked_to_care_hiv', 'mean'),
            tb_treatment_success_rate=('is_treatment_success_tb', 'mean')
        ).reset_index()
        program_agg['hiv_linkage_rate'] = (program_agg['hiv_linkage_rate'] * 100).fillna(0)
        program_agg['tb_treatment_success_rate'] = (program_agg['tb_treatment_success_rate'] * 100).fillna(0)
        df_merged = pd.merge(df_merged, program_agg, on='zone_id', how='left')

    # Final cleanup: fill NaNs created by merges with 0
    numeric_cols = df_merged.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col not in df_zones.columns: # Don't fill original data like population
            df_merged[col] = df_merged[col].fillna(0)

    return df_merged


def aggregate_district_stats(df_zonal_agg: pd.DataFrame) -> Dict[str, Any]:
    """Rolls up zonal statistics to a single district-wide summary."""
    if df_zonal_agg.empty: return {}
    logger.debug("Rolling up zonal statistics to district level.")
    
    total_population = df_zonal_agg['population'].sum()
    district_summary = {'total_population': total_population, 'total_zones': df_zonal_agg['zone_id'].nunique()}
    
    # Define metrics and their weights for weighted averaging
    weighted_metrics = {
        'avg_risk_score': 'population',
        'hiv_linkage_rate': 'population',
        'rejection_rate': 'total_tests_processed'
    }

    for metric, weight_col in weighted_metrics.items():
        if metric in df_zonal_agg.columns and weight_col in df_zonal_agg.columns:
            weights = df_zonal_agg[weight_col].clip(lower=1) # Avoid division by zero
            district_summary[f'population_weighted_{metric}'] = np.average(
                df_zonal_agg[metric].fillna(0), weights=weights
            )
    
    # Sums for counts
    count_cols = ['total_encounters', 'total_tests_processed']
    for col in count_cols:
        if col in df_zonal_agg.columns:
            district_summary[col] = df_zonal_agg[col].sum()
            
    return district_summary


def aggregate_program_kpis(df_zonal_agg: pd.DataFrame) -> Dict[str, Any]:
    """Calculates district-wide programmatic KPIs against strategic targets."""
    if df_zonal_agg.empty: return {}
    kpis = {}
    
    # TB Case Detection Rate (Example Logic)
    if 'positivity_genexpert' in df_zonal_agg.columns:
        # Simplified logic: Real logic would involve population incidence estimates.
        total_tb_tests = df_zonal_agg[df_zonal_agg['positivity_genexpert'] > 0]['total_tests_processed'].sum()
        if total_tb_tests > 0:
            # Placeholder until better estimation is available.
            kpis['tb_case_detection_rate'] = df_zonal_agg['positivity_genexpert'].mean() * 100

    # HIV Linkage to Care
    if 'hiv_linkage_rate' in df_zonal_agg.columns and 'population' in df_zonal_agg.columns:
        weights = df_zonal_agg['population'].clip(lower=1)
        kpis['hiv_linkage_to_care'] = np.average(df_zonal_agg['hiv_linkage_rate'].fillna(0), weights=weights)

    return kpis
