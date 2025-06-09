# sentinel_project_root/visualization/themes.py
#
# PLATINUM STANDARD - Centralized Plotting Theme
# This module defines a single, consistent, and professional Plotly theme
# for the entire Sentinel application, ensuring a cohesive visual identity.

import plotly.graph_objects as go
import logging

# --- Core Application Imports ---
try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in themes.py: A core dependency is missing. {e}", exc_info=True)
    # Define a fallback theme so the app doesn't crash if settings fail.
    sentinel_theme_template = go.layout.Template()
else:
    # --- Sentinel Plotly Theme Template Definition ---
    # This template is registered with Plotly and can be used globally.
    sentinel_theme_template = go.layout.Template(
        layout=go.Layout(
            # --- Fonts ---
            font=dict(
                family="sans-serif",
                size=12,
                color=settings.theme.text
            ),
            # --- Title ---
            title=dict(
                font=dict(size=18, family="sans-serif"),
                x=0.05,  # Align title to the left
                xanchor='left'
            ),
            # --- Background Colors ---
            paper_bgcolor=settings.theme.secondary_background,
            plot_bgcolor=settings.theme.secondary_background,

            # --- Colorways ---
            # Default color sequence for categorical data in plots.
            colorway=settings.theme.plotly_colorway,

            # --- Axes Configuration ---
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor=settings.theme.text,
                zeroline=False,
                ticks='outside',
                title_standoff=10
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#EAEAEA',
                showline=False,
                zeroline=False,
                ticks='outside',
                title_standoff=10
            ),
            # --- Legend ---
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor=settings.theme.secondary_background,
                bordercolor=settings.theme.secondary_background,
                traceorder='normal'
            ),
            # --- Margins ---
            margin=dict(l=70, r=30, t=80, b=70),

            # --- Hover Labels ---
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                font_size=12,
                font_family="sans-serif"
            ),
            hovermode='x unified',

            # --- Geospatial ---
            geo=dict(
                bgcolor=settings.theme.secondary_background,
                showland=True,
                landcolor="#F9F9F9",
                subunitcolor="#CCCCCC"
            )
        )
    )

    logging.debug("Sentinel Plotly theme template created successfully.")
