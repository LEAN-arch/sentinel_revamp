# sentinel_project_root/analytics/forecasting.py
#
# PLATINUM STANDARD - Time-Series Forecasting Engine
# This module provides robust, statistically-grounded time-series forecasting
# capabilities for supply chain management and epidemiological trends using
# the Prophet library.

import logging
import pandas as pd
from typing import Optional, Dict

# --- Core Application & External Library Imports ---
try:
    # Use try-except for Prophet as it can be a complex dependency.
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None # To satisfy type hints if not installed

from config.settings import settings

logger = logging.getLogger(__name__)

# Check for Prophet installation on module load.
if not PROPHET_AVAILABLE:
    logger.critical(
        "Prophet library not found. All forecasting functions will be disabled. "
        "Please install it (`pip install prophet`) to enable this feature."
    )

def _prepare_prophet_dataframe(
    daily_aggregated_series: pd.Series,
    floor: int = 0
) -> Optional[pd.DataFrame]:
    """
    Formats a pandas Series into the two-column [ds, y] DataFrame required by Prophet.

    Args:
        daily_aggregated_series: A Series with a DatetimeIndex and numeric values.
        floor: The minimum value for the forecast (e.g., 0 for counts).

    Returns:
        A DataFrame ready for Prophet, or None if input is invalid.
    """
    if not isinstance(daily_aggregated_series, pd.Series) or daily_aggregated_series.empty:
        return None

    df_prophet = daily_aggregated_series.reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet['floor'] = floor
    return df_prophet

def _run_prophet_forecast(
    df_prophet: pd.DataFrame,
    forecast_days: int,
    prophet_params: Optional[Dict] = None
) -> Optional[pd.DataFrame]:
    """
    Internal function to run the core Prophet forecasting model.

    Args:
        df_prophet: The DataFrame formatted for Prophet ([ds, y]).
        forecast_days: Number of days into the future to forecast.
        prophet_params: Optional dictionary of parameters to pass to the Prophet model.

    Returns:
        A DataFrame containing the forecast, trend, and uncertainty intervals.
    """
    if not PROPHET_AVAILABLE or df_prophet.empty or len(df_prophet) < 3:
        # Prophet requires at least 2 data points, but more is better.
        logger.warning(f"Not enough data points ({len(df_prophet)}) to generate a forecast. At least 3 are recommended.")
        return None

    # Use default params if none are provided
    params = prophet_params or {
        'growth': 'linear',
        'daily_seasonality': False,
        'weekly_seasonality': True,
        'yearly_seasonality': False,
        'interval_width': 0.95  # 95% uncertainty interval
    }

    try:
        model = Prophet(**params)
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=forecast_days)
        future['floor'] = df_prophet['floor'].iloc[0] # Carry the floor into the future
        
        forecast_df = model.predict(future)
        return forecast_df

    except Exception as e:
        logger.error(f"Prophet model fitting or prediction failed: {e}", exc_info=True)
        return None

def forecast_supply_demand(
    daily_consumption_series: pd.Series,
    forecast_days: int = 60
) -> Optional[pd.DataFrame]:
    """
    Forecasts future demand for a single supply item.

    Args:
        daily_consumption_series: A Series of daily consumption counts, indexed by date.
        forecast_days: Number of days to forecast into the future.

    Returns:
        A forecast DataFrame with trend and uncertainty intervals.
    """
    logger.info(f"Generating {forecast_days}-day supply demand forecast...")
    df_prophet = _prepare_prophet_dataframe(daily_consumption_series, floor=0)
    if df_prophet is None:
        return None
        
    return _run_prophet_forecast(df_prophet, forecast_days)

def forecast_epi_trend(
    daily_case_series: pd.Series,
    forecast_days: int = 30
) -> Optional[pd.DataFrame]:
    """
    Forecasts epidemiological trends for disease incidence.

    Args:
        daily_case_series: A Series of daily new case counts, indexed by date.
        forecast_days: Number of days to forecast into the future.

    Returns:
        A forecast DataFrame with trend and uncertainty intervals.
    """
    logger.info(f"Generating {forecast_days}-day epidemiological forecast...")
    df_prophet = _prepare_prophet_dataframe(daily_case_series, floor=0)
    if df_prophet is None:
        return None
        
    # Epidemiology might have stronger weekly patterns.
    params = {
        'growth': 'linear',
        'daily_seasonality': False,
        'weekly_seasonality': True,
        'yearly_seasonality': True, # Assume some yearly pattern
        'interval_width': 0.95
    }

    return _run_prophet_forecast(df_prophet, forecast_days, prophet_params=params)
