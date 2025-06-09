# sentinel_project_root/pages/03_district_dashboard.py
#
# PLATINUM STANDARD - District Health Strategic Command Center
# This dashboard provides District Health Officers with tools for operational
# oversight, resource allocation, and public health program monitoring.

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

# --- Core Application Imports ---
try:
    from config.settings import settings
    from data_processing.loaders import (
        load_health_records, load_lab_results, load_zone_data,
        load_program_outcomes, load_contact_tracing
    )
    from data_processing.enrichment import enrich_lab_results_with_features
    from analytics.aggregation import aggregate_zonal_stats, aggregate_district_stats, aggregate_program_kpis
    from visualization.ui_elements import render_main_header
    from visualization.plots import (
        plot_choropleth_map, plot_categorical_distribution,
        plot_programmatic_kpi, create_empty_figure
    )
except ImportError as e:
    st.error(f"A required application module could not be loaded. Error: {e}")
    st.stop()

logger = logging.getLogger(__name__)


@st.cache_data(ttl=settings.cache_ttl_seconds)
def get_processed_district_data() -> Dict[str, pd.DataFrame]:
    """
    Loads, enriches, and aggregates all data needed for the district dashboard.
    This function centralizes data prep, making the dashboard code cleaner.
    """
    zones = load_zone_data()
    health = load_health_records()
    labs = load_lab_results()
    program = load_program_outcomes()
    
    # Enrich the raw data with calculated features
    labs_enriched = enrich_lab_results_with_features(labs, program)
    
    # Aggregate data to the zonal level
    zonal_stats = aggregate_zonal_stats(health, labs_enriched, program, zones)
    
    # Aggregate zonal data to the district level
    district_stats = aggregate_district_stats(zonal_stats)
    
    # Calculate performance against programmatic targets
    program_kpis = aggregate_program_kpis(zonal_stats)

    return {
        "zonal_stats": zonal_stats,
        "district_stats": district_stats,
        "program_kpis": program_kpis,
    }

class DistrictDashboard:
    """Encapsulates the state and rendering logic for the DHO dashboard."""

    def __init__(self):
        st.set_page_config(
            page_title="District Command Center",
            page_icon="🗺️",
            layout="wide"
        )
        self.data = get_processed_district_data()
        self.zonal_df = self.data["zonal_stats"]
        self.district_stats = self.data["district_stats"]
        self.program_kpis = self.data["program_kpis"]

    def _render_program_kpis(self):
        st.subheader("Performance Against Public Health Targets")
        if not self.program_kpis:
            st.info("Programmatic KPI data is not available.")
            return

        cols = st.columns(len(self.program_kpis))
        kpi_map = {
            'tb_case_detection_rate': ("TB Case Detection Rate", settings.targets.tb_case_detection_rate_pct),
            'hiv_linkage_to_care': ("HIV Linkage to Care", settings.targets.hiv_linkage_to_care_pct)
        }
        
        for i, (key, (title, target)) in enumerate(kpi_map.items()):
            if i < len(cols) and key in self.program_kpis:
                with cols[i]:
                    value = self.program_kpis.get(key, 0)
                    st.plotly_chart(plot_programmatic_kpi(value, target, title), use_container_width=True)
        st.divider()

    def _render_geospatial_overview(self):
        st.subheader("Geospatial Command Center")
        if 'geometry' not in self.zonal_df.columns:
            st.warning("Map unavailable: Zone geometry data is missing.")
            return

        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": row['geometry'], "id": row['zone_id']}
                for _, row in self.zonal_df.iterrows() if pd.notna(row.get('geometry'))
            ]
        }

        metric_options = {
            "Avg. Patient Risk Score": "avg_risk_score",
            "Avg. Lab TAT (Days)": "avg_tat_days",
            "HIV Linkage to Care (%)": "hiv_linkage_rate",
            "TB GeneXpert Positivity (%)": "positivity_genexpert",
            "Lab Rejection Rate (%)": "rejection_rate"
        }

        # Filter to only available metrics
        available_metrics = {name: col for name, col in metric_options.items() if col in self.zonal_df.columns}

        if not available_metrics:
            st.warning("No metrics are available for map display.")
            return
            
        selected_metric = st.selectbox("Select Map Layer:", options=list(available_metrics.keys()))
        color_col = available_metrics[selected_metric]
        
        map_df = self.zonal_df.copy()
        map_df[color_col] = pd.to_numeric(map_df[color_col], errors='coerce').fillna(0)

        st.plotly_chart(
            plot_choropleth_map(
                map_df,
                geojson=geojson_data,
                locations="zone_id",
                color=color_col,
                hover_name="name",
                hover_data={"population": True, "total_encounters": True, color_col: ':.2f'},
                title=f"<b>{selected_metric} by Zone</b>",
                color_continuous_scale="Viridis"
            ),
            use_container_width=True
        )

    def _render_zonal_scorecard(self):
        st.subheader("Zonal Performance Scorecard")
        st.info("Click on column headers to sort and compare zonal performance.")
        
        display_cols = [
            "name", "population", "avg_risk_score", "total_encounters",
            "avg_tat_days", "rejection_rate", "hiv_linkage_rate", "positivity_genexpert"
        ]
        
        # Filter to columns that actually exist in the final aggregated data
        available_display_cols = [col for col in display_cols if col in self.zonal_df.columns]
        
        if not available_display_cols:
            st.warning("No zonal data available to display in the scorecard.")
            return
            
        st.dataframe(
            self.zonal_df[available_display_cols].rename(columns=lambda c: c.replace('_', ' ').title()),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Risk Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "Population": st.column_config.NumberColumn(format="%d")
            }
        )

    def _render_intervention_planner(self):
        st.subheader("Targeted Intervention Planning Assistant")
        st.markdown(
            "Identify zones requiring immediate attention based on a combination of risk factors. "
            "This tool helps prioritize resources for maximum public health impact."
        )

        criteria_options = {
            "High Patient Risk Score": self.zonal_df['avg_risk_score'] > settings.thresholds.district_risk_score_threshold,
            "High Lab Rejection Rate": self.zonal_df['rejection_rate'] > settings.thresholds.lab_rejection_rate_target_pct * 1.5,
            "Poor HIV Linkage to Care": self.zonal_df['hiv_linkage_rate'] < settings.targets.hiv_linkage_to_care_pct * 0.8,
            "High TB Positivity": self.zonal_df['positivity_genexpert'] > 0.15 # Example threshold: 15%
        }
        
        # Filter criteria if the column doesn't exist
        available_criteria = {name: mask for name, mask in criteria_options.items() if mask.name in self.zonal_df.columns}

        selected_criteria = st.multiselect(
            "Select criteria to flag zones:",
            options=list(available_criteria.keys())
        )
        
        if not selected_criteria:
            st.info("Select one or more criteria above to identify priority zones.")
            return

        # Combine selected criteria with an OR condition
        final_mask = pd.Series(False, index=self.zonal_df.index)
        for criterion in selected_criteria:
            final_mask |= available_criteria[criterion]

        priority_zones = self.zonal_df[final_mask]

        st.markdown(f"#### **Identified {len(priority_zones)} Priority Zone(s)**")
        if not priority_zones.empty:
            st.dataframe(
                priority_zones[[c for c in ['name', 'population', 'avg_risk_score', 'rejection_rate', 'hiv_linkage_rate'] if c in priority_zones.columns]],
                use_container_width=True, hide_index=True)
        else:
            st.success("✅ No zones currently meet the selected intervention criteria.")

    def run(self):
        """Main method to render the entire dashboard page."""
        render_main_header("District Strategic Command", "Operational Oversight & Program Monitoring")
        
        if self.zonal_df.empty:
            st.error("Aggregated zonal data could not be loaded or generated. Dashboard cannot be displayed.")
            return

        self._render_program_kpis()

        tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial Overview", " scorecard Zonal Scorecard", "🎯 Intervention Planner"])
        
        with tab1:
            self._render_geospatial_overview()
        
        with tab2:
            self._render_zonal_scorecard()

        with tab3:
            self._render_intervention_planner()


if __name__ == "__main__":
    dashboard = DistrictDashboard()
    dashboard.run()
