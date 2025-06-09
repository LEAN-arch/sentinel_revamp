# sentinel_project_root/visualization/plots.py
#
# PLATINUM STANDARD - Comprehensive Public Health Visualization Engine (V2 Final)
# This module provides a single, authoritative factory for creating a full suite of
# specialized and general-purpose charts for public health surveillance and analysis.

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


class PublicHealthChartFactory:
    """A factory class for creating standardized, public health-specific Plotly charts."""

    def __init__(self, theme_template: go.layout.Template):
        self.theme = theme_template
        px.defaults.template = self.theme

    def create_empty_figure(self, title: str) -> go.Figure:
        """Creates a themed, blank figure with a user-friendly message."""
        fig = go.Figure()
        fig.update_layout(title_text=f'<b>{title}</b>', xaxis={'visible': False}, yaxis={'visible': False})
        fig.add_annotation(text="No data available for the selected filters.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=14)
        return fig

    def plot_kpi_trend(self, series: pd.Series, title: str, y_axis_title: str) -> go.Figure:
        """Creates a themed, annotated line chart for a KPI time series."""
        if not isinstance(series, pd.Series) or series.empty:
            return self.create_empty_figure(title)
        fig = px.line(x=series.index, y=series, title=f"<b>{title}</b>", markers=True)
        hovertemplate = (f"<b>%{{x|%d %b %Y}}</b><br>"
                         f"{html.escape(y_axis_title)}: %{{y:,.2f}}<extra></extra>")
        fig.update_traces(hovertemplate=hovertemplate)
        fig.update_layout(yaxis_title=y_axis_title, xaxis_title=None, showlegend=False)
        return fig

    def plot_categorical_distribution(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, **px_kwargs) -> go.Figure:
        """Creates a themed bar chart showing a categorical distribution."""
        if not all(col in df.columns for col in [x_col, y_col]):
            return self.create_empty_figure(title)
        fig = px.bar(df, x=x_col, y=y_col, title=f'<b>{title}</b>', text_auto=True, **px_kwargs)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis_title=None, xaxis_title=x_col.replace("_", " ").title())
        fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
        return fig
    
    def plot_age_distribution(self, series: pd.Series, title: str) -> go.Figure:
        """Creates a histogram for age distribution."""
        if series.empty: return self.create_empty_figure(title)
        fig = px.histogram(series, title=f'<b>{title}</b>', nbins=20, marginal="box")
        fig.update_layout(yaxis_title="Number of Patients", xaxis_title="Age")
        return fig

    def plot_risk_pyramid(self, df_pyramid: pd.DataFrame) -> go.Figure:
        """Creates a funnel chart to represent a population risk pyramid."""
        if df_pyramid.empty: return self.create_empty_figure("Risk Pyramid")
        tier_order = ["High Risk", "Moderate Risk", "Low Risk"]
        df_pyramid['risk_tier'] = pd.Categorical(df_pyramid['risk_tier'], categories=tier_order, ordered=True)
        df_pyramid.sort_values('risk_tier', inplace=True)
        fig = px.funnel(df_pyramid, x='patient_count', y='risk_tier', title="<b>Population Risk Pyramid</b>",
                        color_discrete_map={"High Risk": settings.theme.risk_high, "Moderate Risk": settings.theme.risk_moderate, "Low Risk": settings.theme.risk_low},
                        color="risk_tier", labels={'patient_count': 'Number of Patients'})
        fig.update_layout(legend_title_text='Risk Tier')
        return fig
        
    def plot_epidemiological_curve(self, case_series: pd.Series, title: str, moving_avg_window: int = 7) -> go.Figure:
        """Creates a classic epidemiological curve (Epi Curve) with a moving average."""
        if case_series.empty:
            return self.create_empty_figure(title)
        moving_avg = case_series.rolling(window=moving_avg_window).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=case_series.index, y=case_series, name='Cases', marker_color=settings.theme.primary))
        fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg, mode='lines', name=f'{moving_avg_window}-Day MA', line=dict(color=settings.theme.risk_high, width=3)))
        fig.update_layout(title=f'<b>{title}</b>', yaxis_title='Case Count', xaxis_title=None, showlegend=True, legend=dict(x=0.01, y=0.99), barmode='overlay')
        return fig

    def plot_programmatic_kpi(self, value: float, target: float, title: str) -> go.Figure:
        """Creates a bullet gauge chart to visualize performance against a target."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value, title={'text': f"<b>{title}</b>"},
            gauge={'axis': {'range': [0, max(value, target) * 1.2]},
                   'steps': [{'range': [0, target * 0.75], 'color': settings.theme.risk_high},
                             {'range': [target * 0.75, target], 'color': settings.theme.risk_moderate},
                             {'range': [target, max(value, target) * 1.2], 'color': settings.theme.risk_low}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.9, 'value': target},
                   'bar': {'color': settings.theme.primary, 'thickness': 0.4}}))
        fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=30))
        return fig
    
    def plot_comorbidity_heatmap(self, corr_matrix: pd.DataFrame, title: str) -> go.Figure:
        """Visualizes a comorbidity correlation matrix as a heatmap."""
        if corr_matrix.empty: return self.create_empty_figure(title)
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale='Reds', title=f'<b>{title}</b>')
        fig.update_xaxes(side="top")
        return fig

    def plot_forecast_with_uncertainty(self, forecast_df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
        """Visualizes a Prophet forecast with uncertainty intervals."""
        if forecast_df.empty:
            return self.create_empty_figure(title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(13, 71, 161, 0.2)', name='Uncertainty'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(width=0), name='95% Interval', fillcolor='rgba(13, 71, 161, 0.2)', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', line=dict(color=settings.theme.primary, width=3), name='Forecast'))
        fig.update_layout(title=f'<b>{title}</b>', yaxis_title=y_title, xaxis_title=None, showlegend=False)
        return fig
    
    def plot_choropleth_map(self, map_df: pd.DataFrame, geojson: Dict, **px_kwargs) -> go.Figure:
        """Creates a themed choropleth map."""
        if 'color' not in px_kwargs or px_kwargs['color'] not in map_df.columns:
            return self.create_empty_figure(px_kwargs.get("title", "Map"))
        fig = px.choropleth_mapbox(
            map_df, geojson=geojson, mapbox_style=settings.mapbox_token and "satellite-streets-v12" or "carto-positron",
            zoom=settings.map_default_zoom, center={"lat": settings.map_default_center_lat, "lon": settings.map_default_center_lon},
            opacity=0.75, **px_kwargs)
        fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
        return fig

# --- Singleton Instance and Public API ---
_chart_factory = PublicHealthChartFactory(sentinel_theme_template)

create_empty_figure = _chart_factory.create_empty_figure
plot_kpi_trend = _chart_factory.plot_kpi_trend
plot_categorical_distribution = _chart_factory.plot_categorical_distribution
plot_age_distribution = _chart_factory.plot_age_distribution
plot_risk_pyramid = _chart_factory.plot_risk_pyramid
plot_epidemiological_curve = _chart_factory.plot_epidemiological_curve
plot_programmatic_kpi = _chart_factory.plot_programmatic_kpi
plot_choropleth_map = _chart_factory.plot_choropleth_map
plot_comorbidity_heatmap = _chart_factory.plot_comorbidity_heatmap
plot_forecast_with_uncertainty = _chart_factory.plot_forecast_with_uncertainty
# Retain an alias for backwards compatibility or general use if needed
plot_supply_forecast = _chart_factory.plot_forecast_with_uncertainty
