# sentinel_project_root/analytics/aggregation.py
#
# PLATINUM STANDARD - Public Health Statistics Engine (V2 - Mission Upgrade)
# This module is upgraded to calculate complex epidemiological and programmatic
# KPIs, forming the analytical core of the surveillance system.

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
    if previous_period_series is not None and not previous_period_series.dropna().empty:
        prev_mean = previous_period_series.dropna().mean()
        stats_result['delta_abs'] = stats_result['current_mean'] - prev_mean
        if prev_mean != 0:
            stats_result['delta_pct'] = (stats_result['delta_abs'] / prev_mean)
        if len(current_period_series) > 1 and len(previous_period_series.dropna()) > 1:
            _, p_value = stats.ttest_ind(current_period_series, previous_period_series.dropna(), equal_var=False)
            stats_result['p_value'] = p_value
            stats_result['is_significant'] = p_value < 0.05
        stats_result['is_positive_change'] = (stats_result['delta_abs'] > 0) if higher_is_better else (stats_result['delta_abs'] < 0)
    return stats_result


def aggregate_zonal_stats(
    df_health: pd.DataFrame, df_labs: pd.DataFrame, df_program: pd.DataFrame, df_zones: pd.DataFrame
) -> pd.DataFrame:
    """Aggregates comprehensive health, lab, and program data to a zonal (regional) level."""
    if df_zones.empty:
        return pd.DataFrame()
    logger.debug("Aggregating comprehensive health data to zonal level.")

    # 1. Aggregate health record stats
    health_agg = pd.DataFrame()
    if not df_health.empty:
        health_agg = df_health.groupby('zone_id').agg(
            avg_risk_score=('ai_risk_score', 'mean'),
            total_encounters=('encounter_id', 'nunique')
        ).reset_index()

    # 2. Aggregate lab stats (positivity rates, TAT)
    lab_agg = pd.DataFrame()
    if not df_labs.empty:
        lab_agg = df_labs.groupby('zone_id').agg(
            avg_tat_days=('turn_around_time_days', 'mean'),
            total_tests_processed=('test_id', 'nunique'),
            rejection_rate=('is_rejected', 'mean')
        ).reset_index()
        lab_agg['rejection_rate'] *= 100

        # Disease-specific positivity rates
        for test_name in df_labs['test_name'].unique():
            positivity_df = df_labs[df_labs['test_name'] == test_name]
            positivity_agg = positivity_df.groupby('zone_id')['is_positive'].mean().reset_index()
            positivity_agg.rename(columns={'is_positive': f'positivity_{test_name.lower()}'}, inplace=True)
            lab_agg = pd.merge(lab_agg, positivity_agg, on='zone_id', how='left')

    # 3. Aggregate program stats (e.g., linkage to care)
    program_agg = pd.DataFrame()
    if not df_program.empty:
        program_agg = df_program.groupby('zone_id').agg(
            hiv_linkage_rate=('is_linked_to_care_hiv', 'mean'),
            tb_treatment_success_rate=('is_treatment_success_tb', 'mean')
        ).reset_index()
        program_agg['hiv_linkage_rate'] *= 100
        program_agg['tb_treatment_success_rate'] *= 100

    # 4. Merge all aggregates into the base zone dataframe
    df_merged = df_zones.copy()
    for agg_df in [health_agg, lab_agg, program_agg]:
        if not agg_df.empty:
            df_merged = pd.merge(df_merged, agg_df, on='zone_id', how='left')
    
    # Fill NA for aggregated columns with 0 after merging
    for col in df_merged.columns:
        if col not in df_zones.columns and pd.api.types.is_numeric_dtype(df_merged[col]):
            df_merged[col].fillna(0, inplace=True)

    return df_merged


def aggregate_district_stats(df_zonal_agg: pd.DataFrame) -> Dict[str, Any]:
    """Rolls up zonal statistics to a single district-wide summary."""
    if df_zonal_agg.empty:
        return {}

    logger.debug("Rolling up zonal statistics to district level.")
    total_population = df_zonal_agg['population'].sum()
    
    # Use population as weights for averaging, where appropriate
    district_summary = {
        'total_population': total_population,
        'total_zones': df_zonal_agg['zone_id'].nunique(),
        'population_weighted_risk_score': np.average(df_zonal_agg['avg_risk_score'].fillna(0), weights=df_zonal_agg['population']),
        'total_encounters': df_zonal_agg['total_encounters'].sum(),
        'overall_hiv_linkage_rate': np.average(df_zonal_agg['hiv_linkage_rate'].fillna(0), weights=df_zonal_agg['population']),
        'overall_rejection_rate': np.average(df_zonal_agg['rejection_rate'].fillna(0), weights=df_zonal_agg['total_tests_processed'].clip(lower=1)),
    }
    
    # Find top prevalence condition
    positivity_cols = {col: col.replace('positivity_', '') for col in df_zonal_agg.columns if col.startswith('positivity_')}
    if positivity_cols:
        top_condition = df_zonal_agg[list(positivity_cols.keys())].mean().idxmax()
        district_summary['top_condition_by_positivity'] = positivity_cols.get(top_condition, 'N/A').replace('_', ' ').title()

    return district_summary


def aggregate_program_kpis(df_zonal_agg: pd.DataFrame) -> Dict[str, Any]:
    """Calculates district-wide programmatic KPIs against strategic targets."""
    if df_zonal_agg.empty:
        return {}

    kpis = {}
    
    # TB Case Detection Rate (Example Logic)
    # Assumes 'population' and some TB case count from lab results are in the aggregated frame
    if 'positivity_genexpert' in df_zonal_agg.columns:
        estimated_tb_cases = df_zonal_agg['population'].sum() * 0.001  # Example: 100 cases per 100,000 population incidence rate
        detected_tb_cases = df_zonal_agg[df_zonal_agg['positivity_genexpert'] > 0]['total_encounters'].sum() # Simplified proxy
        if estimated_tb_cases > 0:
             kpis['tb_case_detection_rate'] = (detected_tb_cases / estimated_tb_cases) * 100

    # HIV Linkage to Care
    if 'hiv_linkage_rate' in df_zonal_agg.columns and 'population' in df_zonal_agg.columns:
        weights = df_zonal_agg['population']
        kpis['hiv_linkage_to_care'] = np.average(df_zonal_agg['hiv_linkage_rate'].fillna(0), weights=weights)

    # Add other programmatic KPIs here...
    return kpis
