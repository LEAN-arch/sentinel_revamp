# sentinel_project_root/analytics/aggregation.py
#
# PLATINUM STANDARD - Statistical Aggregation Engine
# This module provides robust functions for aggregating data and calculating
# decision-grade Key Performance Indicators (KPIs) with statistical rigor.

import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, Any, List

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
    """
    Calculates key statistics for a KPI, including confidence intervals and
    period-over-period significance testing (t-test).

    Args:
        current_period_series: A pandas Series of raw values for the current period.
        previous_period_series: An optional Series of values for the previous period.
        higher_is_better: A boolean indicating the desired direction of change.

    Returns:
        A dictionary containing calculated metrics (mean, confidence interval, p-value, etc.).
    """
    stats_result: Dict[str, Any] = {
        'current_mean': None, 'current_ci': (None, None), 'delta_abs': None,
        'delta_pct': None, 'p_value': None, 'is_significant': False,
        'is_positive_change': None
    }
    
    # --- Current Period Statistics ---
    current_period_series = current_period_series.dropna()
    if current_period_series.empty:
        return stats_result
    
    stats_result['current_mean'] = current_period_series.mean()
    
    # Calculate 95% Confidence Interval for the mean
    if len(current_period_series) > 1:
        se = stats.sem(current_period_series)
        ci = se * stats.t.ppf((1 + 0.95) / 2., len(current_period_series)-1)
        stats_result['current_ci'] = (
            stats_result['current_mean'] - ci,
            stats_result['current_mean'] + ci
        )

    # --- Period-over-Period Comparison ---
    if previous_period_series is not None:
        previous_period_series = previous_period_series.dropna()
        if not previous_period_series.empty:
            prev_mean = previous_period_series.mean()
            stats_result['delta_abs'] = stats_result['current_mean'] - prev_mean
            if prev_mean != 0:
                stats_result['delta_pct'] = (stats_result['delta_abs'] / prev_mean)

            # Two-sample T-test for statistical significance
            if len(current_period_series) > 1 and len(previous_period_series) > 1:
                _, p_value = stats.ttest_ind(current_period_series, previous_period_series, equal_var=False, nan_policy='omit')
                stats_result['p_value'] = p_value
                stats_result['is_significant'] = p_value < 0.05
            
            # Determine if the change is "good" or "bad"
            if higher_is_better:
                stats_result['is_positive_change'] = stats_result['delta_abs'] > 0
            else:
                stats_result['is_positive_change'] = stats_result['delta_abs'] < 0
    
    return stats_result


def aggregate_zonal_stats(df_health: pd.DataFrame, df_zones: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates health data to a zonal (regional) level.

    Args:
        df_health: The enriched health records DataFrame.
        df_zones: The DataFrame containing zone attributes (e.g., population).

    Returns:
        A DataFrame with one row per zone and aggregated health statistics.
    """
    if df_health.empty or df_zones.empty:
        return pd.DataFrame()

    logger.debug("Aggregating health data to zonal level.")

    # Define aggregation rules for clarity and reuse
    aggregations = {
        'avg_risk_score': pd.NamedAgg(column='ai_risk_score', aggfunc='mean'),
        'total_encounters': pd.NamedAgg(column='encounter_id', aggfunc='nunique'),
        'high_risk_patients': pd.NamedAgg(column='ai_risk_score', aggfunc=lambda x: (x >= settings.thresholds.risk_score_high).sum())
    }

    # Perform the primary aggregation
    zone_agg = df_health.groupby('zone_id').agg(**aggregations).reset_index()

    # Merge with zone attributes to get population and other static data
    df_merged = pd.merge(df_zones, zone_agg, on='zone_id', how='left')

    # Calculate derived metrics post-merge
    if 'population' in df_merged.columns and df_merged['population'].notna().any():
        df_merged['prevalence_per_1000'] = (df_merged['total_encounters'] / df_merged['population']) * 1000
        df_merged['high_risk_rate_per_1000'] = (df_merged['high_risk_patients'] / df_merged['population']) * 1000
        
    return df_merged


def aggregate_population_health_stats(df_health: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates high-level statistics for an entire population cohort.

    Args:
        df_health: The health records for the filtered population.

    Returns:
        A dictionary of key population-level statistics.
    """
    if df_health.empty:
        return {}
    
    logger.debug(f"Calculating population health stats for {len(df_health)} records.")
    
    unique_patients = df_health['patient_id'].nunique()
    
    stats_dict = {
        'total_encounters': df_health['encounter_id'].nunique(),
        'unique_patients': unique_patients,
        'avg_risk_score': df_health['ai_risk_score'].mean(),
        'median_risk_score': df_health['ai_risk_score'].median(),
        'avg_age': df_health['age'].mean(),
    }
    
    # Risk stratification
    if unique_patients > 0:
        high_risk_count = df_health[df_health['ai_risk_score'] >= settings.thresholds.risk_score_high]['patient_id'].nunique()
        moderate_risk_count = df_health[
            (df_health['ai_risk_score'] >= settings.thresholds.risk_score_moderate) &
            (df_health['ai_risk_score'] < settings.thresholds.risk_score_high)
        ]['patient_id'].nunique()
        
        stats_dict['high_risk_pct'] = (high_risk_count / unique_patients) * 100
        stats_dict['moderate_risk_pct'] = (moderate_risk_count / unique_patients) * 100

    return stats_dict
