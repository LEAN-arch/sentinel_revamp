# sentinel_project_root/pages/03_district_dashboard.py
# Final corrected version.

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

try:
    from config.settings import settings
    from data_processing.loaders import (
        load_health_records, load_lab_results, load_zone_data,
        load_program_outcomes
    )
    from data_processing.enrichment import (
        enrich_health_records_with_features, 
        enrich_lab_results_with_features,
        enrich_program_outcomes_with_features # THE FIX: Import the missing function
    )
    from analytics.prediction import predict_patient_risk
    from analytics.aggregation import aggregate_zonal_stats, aggregate_district_stats, aggregate_program_kpis
    from visualization.ui_elements import render_main_header
    from visualization.plots import (
        plot_choropleth_map,
        plot_programmatic_kpi, create_empty_figure
    )
except ImportError as e:
    st.error(f"A required application module could not be loaded. Error: {e}")
    st.stop()

logger = logging.getLogger(__name__)

@st.cache_data(ttl=settings.cache_ttl_seconds)
def get_processed_district_data_v4() -> Dict[str, pd.DataFrame]:
    """Loads, enriches, and aggregates all data needed for the district dashboard."""
    zones = load_zone_data()
    health_raw = load_health_records()
    labs_raw = load_lab_results()
    program_raw = load_program_outcomes()
    
    # Execute the full, correct data processing pipeline
    health_enriched = enrich_health_records_with_features(health_raw)
    health_with_risk = predict_patient_risk(health_enriched)
    
    # --- THE FIX: The program enrichment must be called BEFORE aggregation ---
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
        self.data = get_processed_district_data_v4()
        self.zonal_df = self.data.get("zonal_stats", pd.DataFrame())
        self.district_stats = self.data.get("district_stats", {})
        self.program_kpis = self.data.get("program_kpis", {})

    def _render_program_kpis(self):
        st.subheader("Performance Against Public Health Targets")
        if not self.program_kpis:
            st.info("Programmatic KPI data is not available."); return

        cols = st.columns(len(self.program_kpis) or 1)
        kpi_map = {
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
        if self.zonal_df.empty or 'geometry' not in self.zonal_df.columns:
            st.warning("Map unavailable: Zonal data or geometry is missing."); return

        geojson_data = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": row['geometry'], "id": row['zone_id']}
                for _, row in self.zonal_df.iterrows() if pd.notna(row.get('geometry'))]
        }
        metric_options = {
            "Avg. Patient Risk Score": "avg_risk_score",
            "Avg. Lab TAT (Days)": "avg_tat_days",
            "HIV Linkage to Care (%)": "hiv_linkage_rate",
        }
        available_metrics = {name: col for name, col in metric_options.items() if col in self.zonal_df.columns}
        if not available_metrics:
            st.warning("No metrics are available for map display."); return
            
        selected_metric = st.selectbox("Select Map Layer:", options=list(available_metrics.keys()))
        color_col = available_metrics[selected_metric]
        st.plotly_chart(plot_choropleth_map(
            self.zonal_df, geojson=geojson_data, locations="zone_id", color=color_col,
            hover_name="name", hover_data={"population": True, color_col: ':.2f'},
            title=f"<b>{selected_metric} by Zone</b>", color_continuous_scale="Viridis"), use_container_width=True)

    def _render_intervention_planner(self):
        st.subheader("Targeted Intervention Planning Assistant")
        criteria_options = {
            "High Patient Risk": self.zonal_df['avg_risk_score'] > settings.thresholds.district_risk_score_threshold,
            # This line will now work correctly
            "Poor HIV Linkage to Care": self.zonal_df['hiv_linkage_rate'] < settings.targets.hiv_linkage_to_care_pct * 0.8,
        }
        available_criteria = {name: mask for name, mask in criteria_options.items() if isinstance(mask, pd.Series)}
        selected_criteria = st.multiselect("Select criteria to flag zones:", options=list(available_criteria.keys()))
        
        if not selected_criteria:
            st.info("Select one or more criteria to identify priority zones."); return

        final_mask = pd.Series(False, index=self.zonal_df.index)
        for criterion in selected_criteria:
            final_mask |= available_criteria[criterion]
        priority_zones = self.zonal_df[final_mask]

        st.markdown(f"#### **Identified {len(priority_zones)} Priority Zone(s)**")
        if not priority_zones.empty:
            st.dataframe(priority_zones[['name', 'population', 'avg_risk_score', 'hiv_linkage_rate']], use_container_width=True, hide_index=True)
        else:
            st.success("✅ No zones currently meet the selected intervention criteria.")

    def run(self):
        render_main_header("District Strategic Command", "Operational Oversight & Program Monitoring")
        if self.zonal_df.empty:
            st.error("Aggregated zonal data could not be generated."); return

        self._render_program_kpis()
        tab1, tab2 = st.tabs(["🗺️ Geospatial & Intervention", " scorecard Zonal Scorecard"])
        with tab1:
            self._render_geospatial_overview()
            st.divider()
            self._render_intervention_planner()
        with tab2:
            st.dataframe(self.zonal_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    dashboard = DistrictDashboard()
    dashboard.run()
