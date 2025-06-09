# sentinel_project_root/pages/01_field_operations.py
#
# PLATINUM STANDARD - Field Operations & Zonal Command Dashboard
# This dashboard provides supervisors with a high-level overview of field
# team activities, patient risk stratification, and emerging epi signals.

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

# --- Core Application Imports ---
try:
    from config.settings import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from data_processing.enrichment import enrich_health_records_with_features
    from analytics.prediction import predict_patient_risk
    from analytics.aggregation import calculate_kpi_statistics
    from visualization.ui_elements import render_main_header, render_metric_card
    from visualization.plots import plot_kpi_trend
except ImportError as e:
    st.error(f"A required application module could not be loaded. Please check your project structure. Error: {e}")
    st.stop()

logger = logging.getLogger(__name__)


@st.cache_data(ttl=settings.cache_ttl_seconds)
def get_processed_data() -> pd.DataFrame:
    """Loads, enriches, and caches the primary health records with AI risk scores."""
    df = load_health_records()
    if df.empty:
        return pd.DataFrame()
    df_enriched = enrich_health_records_with_features(df)
    df_with_risk = predict_patient_risk(df_enriched)
    return df_with_risk


class FieldOperationsDashboard:
    """An encapsulated class to manage the state and rendering of the dashboard."""

    def __init__(self):
        st.set_page_config(
            page_title="Field Operations",
            page_icon="🧑‍⚕️",
            layout="wide"
        )
        # Load all necessary data during initialization.
        self.health_df = get_processed_data()
        self.zone_df = load_zone_data()
        self.state: Dict[str, Any] = self._initialize_state()

    def _initialize_state(self) -> Dict[str, Any]:
        """Sets up the sidebar filters and returns their current state."""
        st.sidebar.header("📋 Dashboard Filters")
        
        if self.health_df.empty:
            st.sidebar.warning("No health data available.")
            return {}

        zone_options = ["All Zones"] + sorted(self.zone_df['name'].dropna().unique())
        selected_zone = st.sidebar.selectbox("Filter by Zone", zone_options)

        min_date = self.health_df['encounter_date'].min().date()
        max_date = self.health_df['encounter_date'].max().date()
        default_start = max(min_date, max_date - pd.Timedelta(days=29))

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        state = {
            'selected_zone': selected_zone,
            'start_date': date_range[0] if len(date_range) == 2 else default_start,
            'end_date': date_range[1] if len(date_range) == 2 else max_date,
        }
        return state
        
    def _get_filtered_data(self) -> Dict[str, pd.DataFrame]:
        """Filters the main DataFrame based on the current state."""
        if self.health_df.empty or not self.state:
            return {'current': pd.DataFrame(), 'previous': pd.DataFrame(), 'all': pd.DataFrame()}
            
        is_filtered = self.state['selected_zone'] != "All Zones"
        
        df_filtered = self.health_df
        if is_filtered:
            zone_id = self.zone_df.loc[self.zone_df['name'] == self.state['selected_zone'], 'zone_id'].iloc[0]
            df_filtered = self.health_df[self.health_df['zone_id'] == zone_id]
        
        current_period_df = df_filtered[
            df_filtered['encounter_date'].dt.date.between(self.state['start_date'], self.state['end_date'])
        ].copy()

        # Define previous period for comparison
        period_duration = (self.state['end_date'] - self.state['start_date']).days
        prev_start_date = self.state['start_date'] - pd.Timedelta(days=period_duration + 1)
        prev_end_date = self.state['start_date'] - pd.Timedelta(days=1)
        previous_period_df = df_filtered[
            df_filtered['encounter_date'].dt.date.between(prev_start_date, prev_end_date)
        ].copy()

        return {
            'current': current_period_df,
            'previous': previous_period_df,
            'all': df_filtered # The full history for the selected zone
        }

    def _render_kpis(self, data: Dict[str, pd.DataFrame]):
        """Calculates and renders the key performance indicators."""
        st.subheader("Key Performance Indicators")
        cols = st.columns(4)
        
        with cols[0]:
            stats = calculate_kpi_statistics(
                current_period_series=data['current'].drop_duplicates(subset=['patient_id'])['patient_id'].value_counts(),
                previous_period_series=data['previous'].drop_duplicates(subset=['patient_id'])['patient_id'].value_counts(),
                higher_is_better=True
            )
            render_metric_card(
                "Unique Patients Seen",
                {'current_mean': data['current']['patient_id'].nunique()}, # Simplified for this specific metric
                kpi_format="{:,.0f}"
            )
            
        with cols[1]:
            stats = calculate_kpi_statistics(
                current_period_series=data['current']['encounter_id'].value_counts(),
                previous_period_series=data['previous']['encounter_id'].value_counts(),
                higher_is_better=True
            )
            render_metric_card(
                "Total Encounters",
                {'current_mean': len(data['current'])},
                kpi_format="{:,.0f}"
            )

        with cols[2]:
            stats = calculate_kpi_statistics(
                current_period_series=data['current']['ai_risk_score'],
                previous_period_series=data['previous']['ai_risk_score'],
                higher_is_better=False
            )
            render_metric_card("Avg. Patient Risk Score", stats, kpi_format="{:.1f}")

        with cols[3]:
            # This is a rate, not an average, so we handle it differently.
            current_high_risk_pct = 0
            if not data['current'].empty:
                high_risk_count = data['current']['ai_risk_score'][data['current']['ai_risk_score'] >= settings.thresholds.risk_score_high].count()
                total_count = len(data['current'])
                if total_count > 0:
                    current_high_risk_pct = (high_risk_count / total_count) * 100
            
            render_metric_card("High-Risk Encounters", {'current_mean': current_high_risk_pct}, unit_suffix="%")
        st.divider()

    def _render_trends(self, all_zone_data: pd.DataFrame):
        """Renders time-series trend charts."""
        st.subheader("Performance Trends")
        
        col1, col2 = st.columns(2)
        with col1:
            encounters_trend = all_zone_data.set_index('encounter_date')['encounter_id'].resample('W').nunique()
            st.plotly_chart(plot_kpi_trend(encounters_trend, "Weekly Encounters", "Total Encounters"), use_container_width=True)
            
        with col2:
            risk_trend = all_zone_data.set_index('encounter_date')['ai_risk_score'].resample('W').mean()
            st.plotly_chart(plot_kpi_trend(risk_trend, "Weekly Average Patient Risk", "Avg. AI Risk Score"), use_container_width=True)
        st.divider()

    def _render_patient_watch_list(self, current_data: pd.DataFrame):
        """Displays a list of high-priority patients for review."""
        st.subheader("Patient Watch List")
        
        watch_list_df = (
            current_data[current_data['ai_risk_score'] >= settings.thresholds.risk_score_high]
            .sort_values(by='ai_risk_score', ascending=False)
            .drop_duplicates(subset=['patient_id'], keep='first')
        )
        
        if watch_list_df.empty:
            st.success("✅ No patients are currently in the high-risk category for this period.")
            return

        # Prepare for display
        display_cols = {
            'patient_id': 'Patient ID',
            'ai_risk_score': 'Risk Score',
            'age': 'Age',
            'gender': 'Gender',
            'days_since_last_visit': 'Days Since Last Visit',
            'abnormal_vital_count': 'Abnormal Vital Signs'
        }
        
        display_df = watch_list_df[list(display_cols.keys())].rename(columns=display_cols)
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score", format="%d", min_value=0, max_value=100
                )
            }
        )

    def run(self):
        """Main method to render the entire dashboard."""
        render_main_header(
            title="Field Operations Dashboard",
            subtitle=f"Displaying data for: **{self.state.get('selected_zone', 'All Zones')}**"
        )
        
        if self.health_df.empty:
            st.error("Health data could not be loaded. Dashboard functionality is limited.")
            return

        filtered_data = self._get_filtered_data()
        
        if filtered_data['current'].empty:
            st.info("No encounter data available for the selected zone and date range.")
            return

        self._render_kpis(filtered_data)
        self._render_trends(filtered_data['all'])
        self._render_patient_watch_list(filtered_data['current'])
        

if __name__ == "__main__":
    dashboard = FieldOperationsDashboard()
    dashboard.run()
