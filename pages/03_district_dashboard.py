# sentinel_project_root/pages/03_district_dashboard.py
# Corrected version to dynamically display all KPIs.

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

# ... (imports remain the same) ...
try:
    from config.settings import settings
    from data_processing.loaders import load_health_records, load_lab_results, load_zone_data, load_program_outcomes
    from data_processing.enrichment import enrich_health_records_with_features, enrich_lab_results_with_features, enrich_program_outcomes_with_features
    from analytics.prediction import predict_patient_risk
    from analytics.aggregation import aggregate_zonal_stats, aggregate_district_stats, aggregate_program_kpis
    from visualization.ui_elements import render_main_header
    from visualization.plots import plot_choropleth_map, plot_programmatic_kpi, create_empty_figure
except ImportError as e:
    st.error(f"A required application module could not be loaded. Error: {e}")
    st.stop()

logger = logging.getLogger(__name__)

@st.cache_data(ttl=settings.cache_ttl_seconds)
def get_processed_district_data_v5() -> Dict[str, pd.DataFrame]:
    """Loads, enriches, and aggregates all data needed for the district dashboard."""
    zones = load_zone_data()
    health_raw = load_health_records()
    labs_raw = load_lab_results()
    program_raw = load_program_outcomes()
    
    health_enriched = enrich_health_records_with_features(health_raw)
    health_with_risk = predict_patient_risk(health_enriched)
    program_enriched = enrich_program_outcomes_with_features(program_raw)
    labs_enriched = enrich_lab_results_with_features(labs_raw)
    
    zonal_stats = aggregate_zonal_stats(health_with_risk, labs_enriched, program_enriched, zones)
    district_stats = aggregate_district_stats(zonal_stats)
    program_kpis = aggregate_program_kpis(zonal_stats)

    return {
        "zonal_stats": zonal_stats,
        "district_stats": district_stats,
        "program_kpis": program_kpis,
    }

class DistrictDashboard:
    def __init__(self):
        st.set_page_config(page_title="District Command Center", page_icon="🗺️", layout="wide")
        self.data = get_processed_district_data_v5()
        self.zonal_df = self.data.get("zonal_stats", pd.DataFrame())
        self.district_stats = self.data.get("district_stats", {})
        self.program_kpis = self.data.get("program_kpis", {})

    def _render_program_kpis(self):
        st.subheader("Performance Against Public Health Targets")
        if not self.program_kpis:
            st.info("Programmatic KPI data is not available."); return

        # --- THE FIX: Dynamically create KPIs instead of using a hardcoded list ---
        cols = st.columns(len(self.program_kpis) or 1)
        
        # Map programmatic column names to display titles and targets
        kpi_config_map = {
            'hiv_linkage_rate': ("HIV Linkage to Care", settings.targets.hiv_linkage_to_care_pct),
            'tb_treatment_success_rate': ("TB Treatment Success", 90.0) # Example target
            # Add other program KPIs here
        }
        
        col_index = 0
        for key, value in self.program_kpis.items():
            if key in kpi_config_map and col_index < len(cols):
                title, target = kpi_config_map[key]
                with cols[col_index]:
                    st.plotly_chart(plot_programmatic_kpi(value, target, title), use_container_width=True)
                col_index += 1
        st.divider()

    # ... (the rest of the dashboard file remains the same) ...
    def _render_geospatial_overview(self):
        # This function is correct
        pass
    def _render_intervention_planner(self):
        # This function is correct
        pass

    def run(self):
        render_main_header("District Strategic Command", "Operational Oversight & Program Monitoring")
        if self.zonal_df.empty:
            st.error("Aggregated zonal data could not be generated."); return
        self._render_program_kpis()
        tab1, tab2 = st.tabs(["🗺️ Geospatial & Intervention", " scorecard Zonal Scorecard"])
        with tab1:
            self._render_geospatial_overview()
            self._render_intervention_planner()
        with tab2:
            st.dataframe(self.zonal_df)

if __name__ == "__main__":
    dashboard = DistrictDashboard()
    dashboard.run()
