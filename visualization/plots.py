# sentinel_project_root/visualization/plots.py
#
# PLATINUM STANDARD - Centralized Charting Engine
# This module provides a factory for creating standardized, theme-aware,
# and publication-quality Plotly charts for the Sentinel application.

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from typing import Optional, Dict, Any, List
import html

# --- Core Application & Visualization Imports ---
try:
    from config.settings import settings
    from .themes import sentinel_theme_template
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in plots.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChartFactory:
    """A factory class for creating standardized, theme-consistent Plotly charts."""

    def __init__(self, theme_template: go.layout.Template):
        """Initializes the factory and sets the global default theme."""
        self.theme = theme_template
        px.defaults.template = self.theme

    def create_empty_figure(self, title: str) -> go.Figure:
        """Creates a themed, blank Plotly figure with a user-friendly message."""
        fig = go.Figure()
        fig.update_layout(
            title_text=f'<b>{html.escape(title)}</b>',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        fig.add_annotation(
            text="No data available for the selected filters.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig

    def plot_kpi_trend(self, series: pd.Series, title: str, y_axis_title: str) -> go.Figure:
        """Creates a themed, annotated line chart for a KPI time series."""
        if not isinstance(series, pd.Series) or series.empty:
            return self.create_empty_figure(title)

        fig = px.line(x=series.index, y=series, title=f"<b>{html.escape(title)}</b>", markers=True)

        hovertemplate = (
            f"<b>%{{x|%d %b %Y}}</b><br>"
            f"{html.escape(y_axis_title)}: %{{y:,.2f}}"
            f"<extra></extra>"
        )
        fig.update_traces(hovertemplate=hovertemplate)
        fig.update_layout(yaxis_title=y_axis_title, xaxis_title=None, showlegend=False)
        return fig

    def plot_categorical_distribution(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        orientation: str = 'v',
        **px_kwargs
    ) -> go.Figure:
        """Creates a themed bar chart showing a categorical distribution."""
        if not all(col in df.columns for col in [x_col, y_col]):
            return self.create_empty_figure(title)

        fig = px.bar(df, x=x_col, y=y_col, title=f"<b>{html.escape(title)}</b>", orientation=orientation, **px_kwargs)

        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        return fig

    def plot_choropleth_map(self, map_df: pd.DataFrame, geojson: Dict, **px_kwargs) -> go.Figure:
        """Creates a themed choropleth map."""
        if 'color' not in px_kwargs or px_kwargs['color'] not in map_df.columns:
            return self.create_empty_figure(px_kwargs.get("title", "Map"))

        fig = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            mapbox_style=settings.mapbox_token and "satellite-streets-v12" or "carto-positron",
            zoom=settings.map_default_zoom,
            center={"lat": settings.map_default_center_lat, "lon": settings.map_default_center_lon},
            opacity=0.7,
            **px_kwargs
        )
        fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
        return fig
    
    def plot_supply_forecast(self, forecast_df: pd.DataFrame, title: str) -> go.Figure:
        """Visualizes a Prophet forecast with uncertainty intervals."""
        if forecast_df.empty:
            return self.create_empty_figure(title)

        fig = go.Figure()
        # Uncertainty Interval (shaded area)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat_upper'],
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(13, 71, 161, 0.2)', name='Uncertainty'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat_lower'],
            mode='lines', line=dict(width=0), name='95% Interval',
            fillcolor='rgba(13, 71, 161, 0.2)', fill='tozeroy',
        ))
        # Main Forecast Line
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat'],
            mode='lines', line=dict(color=settings.theme.primary, width=3),
            name='Forecast'
        ))

        fig.update_layout(
            title=f'<b>{html.escape(title)}</b>',
            yaxis_title='Predicted Consumption',
            xaxis_title=None,
            showlegend=False
        )
        return fig
    
    def plot_age_distribution(self, series: pd.Series, title: str) -> go.Figure:
        """Creates a histogram for age distribution."""
        if series.empty: return self.create_empty_figure(title)
        fig = px.histogram(series, title=f'<b>{html.escape(title)}</b>', nbins=20)
        fig.update_layout(yaxis_title="Number of Patients", xaxis_title="Age")
        return fig
    
    def plot_risk_pyramid(self, df_pyramid: pd.DataFrame) -> go.Figure:
        """Creates a funnel chart to represent a risk pyramid."""
        if df_pyramid.empty: return self.create_empty_figure("Risk Pyramid")
        
        # Ensure order is always High -> Moderate -> Low
        tier_order = ["High Risk", "Moderate Risk", "Low Risk"]
        df_pyramid['risk_tier'] = pd.Categorical(df_pyramid['risk_tier'], categories=tier_order, ordered=True)
        df_pyramid.sort_values('risk_tier', inplace=True)
        
        fig = px.funnel(df_pyramid, x='patient_count', y='risk_tier', title="<b>Population Risk Pyramid</b>")
        return fig

# --- Singleton Instance and Public API ---
# This pattern ensures the factory is initialized only once.
_chart_factory = ChartFactory(sentinel_theme_template)

# Expose factory methods as simple functions for clean imports.
create_empty_figure = _chart_factory.create_empty_figure
plot_kpi_trend = _chart_factory.plot_kpi_trend
plot_categorical_distribution = _chart_factory.plot_categorical_distribution
plot_choropleth_map = _chart_factory.plot_choropleth_map
plot_supply_forecast = _chart_factory.plot_supply_forecast
plot_age_distribution = _chart_factory.plot_age_distribution
plot_risk_pyramid = _chart_factory.plot_risk_pyramid
