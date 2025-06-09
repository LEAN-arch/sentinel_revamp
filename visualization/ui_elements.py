# sentinel_project_root/visualization/ui_elements.py
# Corrected version.

import streamlit as st
import logging
import pandas as pd  # <<< THE FIX: Import pandas to resolve the NameError.
from typing import Dict, Any

try:
    from config.settings import settings
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in ui_elements.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


def render_main_header(title: str, subtitle: str) -> None:
    """Renders a standardized main page header."""
    header_cols = st.columns([0.1, 0.9])
    with header_cols[0]:
        if settings.app_logo_large_path and settings.app_logo_large_path.exists():
            st.image(str(settings.app_logo_large_path), width=90)
        else:
            st.markdown("### 🏥")

    with header_cols[1]:
        st.title(title)
        st.subheader(subtitle)
    st.divider()


def render_metric_card(
    title: str,
    stats: Dict[str, Any],
    kpi_format: str = "{:.1f}",
    unit_prefix: str = "",
    unit_suffix: str = ""
) -> None:
    """Renders a Streamlit metric card with detailed statistical information."""
    main_value = stats.get('current_mean')
    value_str = f"{unit_prefix}{kpi_format.format(main_value)}{unit_suffix}" if main_value is not None else "N/A"

    delta_str = "No comparison period"
    delta_val = stats.get('delta_pct')
    delta_color = "off"
    if delta_val is not None:
        delta_str = f"{delta_val:+.1%}"
        if stats.get('is_significant'):
            is_positive_change = stats.get('is_positive_change')
            if is_positive_change is True:
                delta_color = "normal"
            elif is_positive_change is False:
                delta_color = "inverse"

    ci_lower, ci_upper = stats.get('current_ci', (None, None))
    p_value = stats.get('p_value')
    help_parts = []
    if ci_lower is not None and ci_upper is not None:
        help_parts.append(f"95% CI: [{kpi_format.format(ci_lower)}, {kpi_format.format(ci_upper)}]")
    if p_value is not None:
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        help_parts.append(f"Significance vs. last period: {p_str}")
    help_text = " | ".join(help_parts) if help_parts else "Current period statistics."

    st.metric(
        label=title,
        value=value_str,
        delta=delta_str,
        delta_color=delta_color,
        help=help_text
    )


def render_dashboard_filter_form(
    df: pd.DataFrame,
    date_col: str,
    default_days: int = 28
) -> Dict[str, Any]:
    """Renders a standardized filter form in the sidebar."""
    st.sidebar.header("Dashboard Filters")
    filters = {}

    if df.empty or date_col not in df.columns:
        st.sidebar.warning("No data to filter.")
        return filters

    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=default_days-1))

    # Use a unique key for the widget to prevent state conflicts between pages
    page_key = st.session_state.get('current_page', 'default_page_key')

    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"filter_date_range_{page_key}"
    )

    if len(date_range) == 2:
        filters['start_date'] = date_range[0]
        filters['end_date'] = date_range[1]
    else:
        filters['start_date'] = default_start
        filters['end_date'] = max_date
        
    return filters
