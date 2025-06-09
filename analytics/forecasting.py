# sentinel_project_root/analytics/forecasting.py
#
# PLATINUM STANDARD - Time-Series Forecasting Engine (V2.2 - Re-validated)
# This module provides robust, context-aware time-series forecasting for
# supply chain and epidemiological trends using the Prophet library.

import logging
import pandas as pd
from typing import Optional, Dict

# --- Core Application & External Library Imports ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in forecasting.py: Settings could not be imported. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

if not PROPHET_AVAILABLE:
    logger.critical(
        "Prophet library not found. Forecasting functions will be disabled. "
        "Install with `pip install prophet`."
    )

def _prepare_prophet_dataframe(
    daily_aggregated_series: pd.Series,
    floor: int = 0,
    min_points: int = 14
) -> Optional[pd.DataFrame]:
    """
    Formats a pandas Series into the two-column [ds, y] DataFrame required by Prophet,
    ensuring sufficient data points exist.
    """
    if not isinstance(daily_aggregated_series, pd.Series) or daily_aggregated_series.empty:
        logger.warning("Forecasting input series is empty.")
        return None

    # Drop NaNs and ensure we have enough data for a reliable forecast
    series_clean = daily_aggregated_series.dropna()
    if len(series_clean) < min_points:
        logger.warning(f"Insufficient data for forecast ({len(series_clean)} points). "
                       f"A minimum of {min_points} is recommended.")
        return None

    df_prophet = series_clean.reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet['floor'] = floor
    return df_prophet

def _run_prophet_forecast(
    df_prophet: pd.DataFrame,
    forecast_days: int,
    prophet_params: Optional[Dict] = None
) -> Optional[pd.DataFrame]:
    """Internal function to run the core Prophet forecasting model with dynamic parameters."""
    if not PROPHET_AVAILABLE or df_prophet.empty:
        return None

    # --- Dynamic Seasonality based on data length ---
    data_duration_days = (df_prophet['ds'].max() - df_prophet['ds'].min()).days
    
    # Default parameters
    params = {
        'growth': 'linear',
        'changepoint_prior_scale': 0.05,  # Allows more flexibility for trend changes
        'daily_seasonality': False,
        'weekly_seasonality': 'auto', # Let Prophet decide if there's enough data
        'yearly_seasonality': False, # Default to off unless there's enough data
        'interval_width': 0.95
    }

    # Only enable yearly seasonality if we have more than a year of data
    if data_duration_days > 365:
        params['yearly_seasonality'] = True
        logger.debug("Enabling yearly seasonality for forecast.")

    # Override defaults with any user-provided parameters
    if prophet_params:
        params.update(prophet_params)

    try:
        model = Prophet(**params)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_days)
        future['floor'] = df_prophet['floor'].iloc[0]
        
        forecast_df = model.predict(future)
        # Ensure forecasted values don't go below the floor (e.g., negative consumption)
        forecast_df[['yhat', 'yhat_lower', 'yhat_upper']] = forecast_df[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=df_prophet['floor'].iloc[0])
        return forecast_df
    except Exception as e:
        logger.error(f"Prophet model fitting or prediction failed: {e}", exc_info=True)
        return None

def forecast_supply_demand(
    daily_consumption_series: pd.Series,
    forecast_days: int = 90
) -> Optional[pd.DataFrame]:
    """
    Forecasts future demand for a single supply item using intelligent defaults.
    """
    logger.info(f"Generating {forecast_days}-day supply demand forecast...")
    df_prophet = _prepare_prophet_dataframe(daily_consumption_series, floor=0)
    if df_prophet is None:
        return None
        
    # Supply chain might have less yearly seasonality but strong weekly patterns.
    # We can let the dynamic logic in _run_prophet_forecast handle this.
    return _run_prophet_forecast(df_prophet, forecast_days)

def forecast_epi_trend(
    daily_case_series: pd.Series,
    forecast_days: int = 45
) -> Optional[pd.DataFrame]:
    """
    Forecasts epidemiological trends for disease incidence with robust seasonality.
    """
    logger.info(f"Generating {forecast_days}-day epidemiological forecast...")
    df_prophet = _prepare_propHnet_dataframe(daily_case_series, floor=0)
    if df_prophet is None:
        return None
        
    # For epi data, explicitly enabling yearly seasonality if data allows is often key.
    # The dynamic logic in _run_prophet_forecast will handle this automatically.
    # No custom params needed unless specific holidays or regressors are added.
    return _run_prophet_forecast(df_prophet, forecast_days)
