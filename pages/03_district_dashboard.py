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
        enrich_program_outcomes_with_features
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
def get_processed_district_data_v6() -> Dict[str, Any]:
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
        self.data = get_processed_district_data_v6()
        self.zonal_df = self.data.get("zonal_stats", pd.DataFrame())
        self.district_stats = self.data.get("district_stats", {})
        self.program_kpis = self.data.get("program_kpis", {})

    def _render_program_kpis(self):
        st.subheader("Performance Against Public Health Targets")
        if not self.program_kpis:
            st.info("Programmatic KPI data is not available."); return

        # --- THE FIX: Dynamically display all calculated KPIs ---
        kpi_config_map = {
            'hiv_linkage_rate': ("HIV Linkage to Care", settings.targets.hiv_linkage_to_care_pct),
            'tb_treatment_success_rate': ("TB Treatment Success", 90.0) # Example target
        }
        
        # Filter to only KPIs we have data for and a config for
        available_kpis = {k: v for k, v in self.program_kpis.items() if k in kpi_config_map}
        
        if not available_kpis:
            st.info("No configured programmatic KPIs available to display."); return
            
        cols = st.columns(len(available_kpis))
        for i, (key, value) in enumerate(available_kpis.items()):
            title, target = kpi_config_map[key]
            with cols[i]:
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
            "Avg. Patient Risk Score": "avg_risk_score", "Avg. Lab TAT (Days)": "avg_tat_days",
            "HIV Linkage to Care (%)": "hiv_linkage_rate", "TB Treatment Success (%)": "tb_treatment_success_rate"
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

    def _render_zonal_scorecard(self):
        st.subheader("Zonal Performance Scorecard")
        st.info("Click on column headers to sort and compare zonal performance.")
        display_cols = ["name", "population", "avg_risk_score", "total_encounters", "avg_tat_days", "hiv_linkage_rate", "tb_treatment_success_rate"]
        available_display_cols = [col for col in display_cols if col in self.zonal_df.columns]
        if not available_display_cols:
            st.warning("No zonal data available for scorecard."); return
            
        st.dataframe(
            self.zonal_df[available_display_cols].rename(columns=lambda c: c.replace('_', ' ').title()),
            use_container_width=True, hide_index=True,
            column_config={"Avg Risk Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100)})

    def _render_intervention_planner(self):
        st.subheader("Targeted Intervention Planning Assistant")
        criteria_options = {
            "High Patient Risk": self.zonal_df['avg_risk_score'] > settings.thresholds.district_risk_score_threshold,
            "Poor HIV Linkage": self.zonal_df['hiv_linkage_rate'] < settings.targets.hiv_linkage_to_care_pct * 0.8,
            "Poor TB Treatment Success": self.zonal_df.get('tb_treatment_success_rate', 0) < 90 * 0.8
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
            display_cols = ['name', 'population', 'avg_risk_score', 'hiv_linkage_rate', 'tb_treatment_success_rate']
            st.dataframe(priority_zones[[c for c in display_cols if c in priority_zones.columns]], use_container_width=True, hide_index=True)
        else:
            st.success("✅ No zones currently meet the selected intervention criteria.")

    def run(self):
        render_main_header("District Strategic Command", "Operational Oversight & Program Monitoring")
        if self.zonal_df.empty:
            st.error("Aggregated zonal data could not be generated."); return

        self._render_program_kpis()
        
        # --- THE FIX: Restore the tabbed layout ---
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
