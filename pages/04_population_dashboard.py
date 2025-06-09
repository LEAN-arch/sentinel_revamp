# sentinel_project_root/pages/04_population_dashboard.py
#
# PLATINUM STANDARD - Population Health Analytics & Research Console
# A deep-dive toolkit for epidemiologists and health analysts to explore demographic,
# clinical, and risk factor data across the population.

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List

# --- Core Application Imports ---
try:
    from config.settings import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from data_processing.enrichment import enrich_health_records_with_features
    from analytics.prediction import predict_patient_risk
    from visualization.ui_elements import render_main_header
    from visualization.plots import (
        plot_age_distribution, plot_risk_pyramid, plot_categorical_distribution,
        plot_comorbidity_heatmap, plot_epidemiological_curve, create_empty_figure
    )
except ImportError as e:
    st.error(f"A required application module could not be loaded. Error: {e}")
    st.stop()

logger = logging.getLogger(__name__)


@st.cache_data(ttl=settings.cache_ttl_seconds)
def get_processed_population_data() -> pd.DataFrame:
    """Loads, enriches, and caches the primary health records for population analysis."""
    df_health = load_health_records()
    if df_health.empty:
        return pd.DataFrame()
    # Add synthetic socioeconomic data for this dashboard's purpose
    if 'socioeconomic_quintile' not in df_health.columns:
        df_health['socioeconomic_quintile'] = pd.cut(
            df_health['patient_id'].str.slice(-2).astype(int), # Create reproducible quintiles
            bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True
        )
    df_enriched = enrich_health_records_with_features(df_health)
    return predict_patient_risk(df_enriched)

@st.cache_data
def get_comorbidity_matrix(df: pd.DataFrame, diseases: List[str]) -> pd.DataFrame:
    """Calculates the comorbidity matrix for a list of diseases."""
    if df.empty:
        return pd.DataFrame()
    
    # Pivot the data to have patients as rows and diseases as columns
    comorbidity_df = df[df['primary_diagnosis'].isin(diseases)].pivot_table(
        index='patient_id',
        columns='primary_diagnosis',
        aggfunc='size',
        fill_value=0
    )
    # Convert counts to boolean flags (1 if present, 0 if not)
    comorbidity_df = (comorbidity_df > 0).astype(int)
    
    # Calculate the correlation matrix
    corr_matrix = comorbidity_df.corr()
    
    # Return only where correlation is meaningful (not self-correlation)
    return corr_matrix.where(lambda x: x != 1)


class PopulationDashboard:
    """Encapsulates the state and rendering logic for the population health dashboard."""

    def __init__(self):
        st.set_page_config(
            page_title="Population Analytics",
            page_icon="📊",
            layout="wide"
        )
        self.health_df = get_processed_population_data()
        self.state = self._initialize_state()
        self.filtered_df = self._get_filtered_data()

    def _initialize_state(self) -> Dict[str, Any]:
        """Sets up the advanced sidebar filters."""
        st.sidebar.header("🔬 Cohort Filters")
        
        state = {}
        if self.health_df.empty:
            st.sidebar.warning("No health data loaded.")
            return state

        # Date Range Filter
        min_date = self.health_df['encounter_date'].min().date()
        max_date = self.health_df['encounter_date'].max().date()
        date_range = st.sidebar.date_input("Filter by Encounter Date:", value=(min_date, max_date))
        state['start_date'] = date_range[0] if len(date_range) == 2 else min_date
        state['end_date'] = date_range[1] if len(date_range) == 2 else max_date

        # Demographic Filters
        state['age_range'] = st.sidebar.slider("Filter by Age:", 
                                               int(self.health_df['age'].min()), 
                                               int(self.health_df['age'].max()), 
                                               (0, 100))
        state['gender_filter'] = st.sidebar.multiselect("Filter by Gender:", 
                                                        options=self.health_df['gender'].unique(),
                                                        default=self.health_df['gender'].unique())
        
        # Socioeconomic Filter
        state['socioeconomic_filter'] = st.sidebar.multiselect("Filter by Socioeconomic Quintile (1=Poorest):", 
                                                               options=sorted(self.health_df['socioeconomic_quintile'].unique()),
                                                               default=sorted(self.health_df['socioeconomic_quintile'].unique()))

        # Comorbidity Filter (for focused analysis)
        state['comorbidity_filter'] = st.sidebar.multiselect("Filter by Presence of Comorbidity:", 
                                                            options=settings.key_diagnoses_for_action)

        return state
        
    def _get_filtered_data(self) -> pd.DataFrame:
        """Applies all active filters to the main dataframe."""
        if self.health_df.empty or not self.state:
            return pd.DataFrame()
            
        df = self.health_df[
            (self.health_df['encounter_date'].dt.date.between(self.state['start_date'], self.state['end_date'])) &
            (self.health_df['age'].between(self.state['age_range'][0], self.state['age_range'][1])) &
            (self.health_df['gender'].isin(self.state['gender_filter'])) &
            (self.health_df['socioeconomic_quintile'].isin(self.state['socioeconomic_filter']))
        ]
        
        if self.state['comorbidity_filter']:
            patients_with_comorbidity = df[df['primary_diagnosis'].isin(self.state['comorbidity_filter'])]['patient_id'].unique()
            df = df[df['patient_id'].isin(patients_with_comorbidity)]

        return df

    def _render_summary_kpis(self):
        st.subheader("Filtered Cohort Summary")
        
        cols = st.columns(4)
        if self.filtered_df.empty:
            for col in cols:
                col.metric("Value", "N/A")
            return
            
        cols[0].metric("Unique Patients in Cohort", f"{self.filtered_df['patient_id'].nunique():,}")
        cols[1].metric("Median Age", f"{self.filtered_df['age'].median():.1f}")
        cols[2].metric("Median AI Risk Score", f"{self.filtered_df['ai_risk_score'].median():.1f}")
        
        # Calculate most prevalent condition in the filtered cohort
        top_diagnosis = self.filtered_df['primary_diagnosis'].mode()
        cols[3].metric("Top Diagnosis", top_diagnosis[0] if not top_diagnosis.empty else "N/A")
        
        st.download_button(
            "⬇️ Download Cohort Data (CSV)",
            self.filtered_df.to_csv(index=False),
            file_name=f"sentinel_cohort_{self.state['start_date']}_to_{self.state['end_date']}.csv",
            mime="text/csv",
        )
        st.divider()

    def _render_disease_surveillance(self):
        st.subheader("Disease Surveillance")
        disease_options = settings.key_diagnoses_for_action
        selected_disease = st.selectbox("Select a Disease for Trend Analysis:", disease_options)
        
        if selected_disease:
            case_series = self.filtered_df[self.filtered_df['primary_diagnosis'] == selected_disease].set_index('encounter_date')['patient_id'].resample('W-MON').nunique()
            st.plotly_chart(plot_epidemiological_curve(case_series, f"Weekly Incidence: {selected_disease}"), use_container_width=True)
            
    def _render_risk_analysis(self):
        st.subheader("Demographic & Risk Factor Analysis")
        df = self.filtered_df.drop_duplicates(subset=['patient_id'], keep='last')

        col1, col2 = st.columns([1, 2])
        with col1:
            if not df.empty:
                pyramid_data = df['risk_tier'].value_counts().rename_axis('risk_tier').reset_index(name='patient_count')
                st.plotly_chart(plot_risk_pyramid(pyramid_data), use_container_width=True)
            else:
                st.plotly_chart(create_empty_figure("Risk Pyramid"), use_container_width=True)
                
        with col2:
            st.plotly_chart(plot_age_distribution(df['age'], "Age Distribution of Cohort"), use_container_width=True)
            
        st.subheader("Risk Score by Demographic Group")
        risk_by_group = df.groupby('socioeconomic_quintile')['ai_risk_score'].mean().reset_index()
        st.plotly_chart(plot_categorical_distribution(risk_by_group, 'socioeconomic_quintile', 'ai_risk_score', 'Avg. AI Risk Score by Socioeconomic Quintile', text_auto=".1f"), use_container_width=True)

    def _render_comorbidity_insights(self):
        st.subheader("Comorbidity Analysis")
        st.info("This heatmap shows the correlation of co-occurrence between priority diseases. Higher values indicate a stronger tendency for patients to have both diagnoses.")
        
        comorbidity_diseases = settings.key_diagnoses_for_action
        
        matrix = get_comorbidity_matrix(self.filtered_df, comorbidity_diseases)
        
        st.plotly_chart(plot_comorbidity_heatmap(matrix, "Comorbidity Correlation Heatmap"), use_container_width=True)

    def run(self):
        """Main method to render the entire dashboard page."""
        render_main_header("Population Health Research Console", "In-depth Cohort Analysis and Epidemiological Surveillance")
        
        if self.health_df.empty:
            st.error("Population health data could not be loaded. Dashboard is disabled.")
            return

        self._render_summary_kpis()
        
        if self.filtered_df.empty:
            st.warning("No data matches the selected filters.")
            return

        tab1, tab2, tab_comorbidity = st.tabs(["Disease Surveillance", "Demographic & Risk Analysis", "Comorbidity Insights"])
        
        with tab1:
            self._render_disease_surveillance()

        with tab2:
            self._render_risk_analysis()
            
        with tab_comorbidity:
            self._render_comorbidity_insights()


if __name__ == "__main__":
    dashboard = PopulationDashboard()
    dashboard.run()
