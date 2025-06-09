# sentinel_project_root/pages/02_clinic_dashboard.py
# Re-validated version.

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

try:
    from config.settings import settings
    from data_processing.loaders import load_lab_results, load_supply_utilization, load_program_outcomes
    from data_processing.enrichment import enrich_lab_results_with_features
    from analytics.aggregation import calculate_kpi_statistics
    from analytics.forecasting import forecast_supply_demand
    from visualization.ui_elements import render_main_header, render_metric_card
    from visualization.plots import (
        plot_kpi_trend, plot_categorical_distribution, plot_supply_forecast,
        create_empty_figure
    )
except ImportError as e:
    st.error(f"A required application module could not be loaded. Please check your project structure. Error: {e}")
    st.stop()

logger = logging.getLogger(__name__)

@st.cache_data(ttl=settings.cache_ttl_seconds)
def get_clinic_data() -> Dict[str, pd.DataFrame]:
    """Loads and caches all data sources required for the clinic dashboard."""
    labs_raw = load_lab_results()
    program_outcomes = load_program_outcomes()
    
    labs_enriched = enrich_lab_results_with_features(labs_raw, program_outcomes)

    return {
        'labs': labs_enriched,
        'supply': load_supply_utilization()
    }

class ClinicDashboard:
    """An encapsulated class to manage state and rendering for the clinic dashboard."""
    def __init__(self):
        st.set_page_config(
            page_title="Clinic Dashboard",
            page_icon="🏥",
            layout="wide"
        )
        self.all_data = get_clinic_data()
        self.state: Dict[str, Any] = self._initialize_state()
        self.filtered_data = self._get_filtered_data()

    def _initialize_state(self) -> Dict[str, Any]:
        """Sets up the sidebar filters and returns their current state."""
        st.sidebar.header("🔬 Dashboard Filters")
        
        df = self.all_data.get('labs')
        if df is None or df.empty:
            st.sidebar.warning("No lab data available.")
            return {}

        min_date = df['sample_collection_date'].min().date()
        max_date = df['sample_collection_date'].max().date()
        default_start = max(min_date, max_date - pd.Timedelta(days=29))

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        return {'start_date': date_range[0], 'end_date': date_range[1]}

    def _get_filtered_data(self) -> Dict[str, pd.DataFrame]:
        """Filters all datasets based on the current date range state."""
        filtered = {}
        if not self.state:
            return {'labs': pd.DataFrame(), 'supply': pd.DataFrame()}

        for name, df in self.all_data.items():
            if df.empty:
                filtered[name] = pd.DataFrame()
                continue
            
            date_col = next((col for col in df.columns if 'date' in col and 'result' not in col), None)
            if date_col:
                filtered[name] = df[
                    df[date_col].dt.date.between(self.state['start_date'], self.state['end_date'])
                ].copy()
            else:
                filtered[name] = df.copy()
        return filtered

    def _render_lab_kpis(self, labs_current: pd.DataFrame, labs_all: pd.DataFrame):
        """Renders KPIs for the Laboratory Performance tab."""
        st.subheader("Laboratory Key Performance Indicators")
        
        period_duration = (self.state['end_date'] - self.state['start_date']).days
        prev_start_date = self.state['start_date'] - pd.Timedelta(days=period_duration + 1)
        prev_end_date = self.state['start_date'] - pd.Timedelta(days=1)
        labs_previous = labs_all[
            labs_all['sample_collection_date'].dt.date.between(prev_start_date, prev_end_date)
        ]

        cols = st.columns(3)
        with cols[0]:
            stats = calculate_kpi_statistics(
                current_period_series=labs_current['turn_around_time_days'],
                previous_period_series=labs_previous['turn_around_time_days'],
                higher_is_better=False)
            render_metric_card("Avg. Test Turnaround", stats, unit_suffix=" days")

        with cols[1]:
            current_rejection_rate = (labs_current['is_rejected'].mean() or 0) * 100
            previous_rejection_rate = (labs_previous['is_rejected'].mean() or 0) * 100
            delta_val = current_rejection_rate - previous_rejection_rate
            render_metric_card("Sample Rejection Rate", 
                               {'current_mean': current_rejection_rate, 
                                'delta_pct': delta_val/previous_rejection_rate if previous_rejection_rate > 0 else None,
                                'is_significant': False},
                               unit_suffix="%")
        with cols[2]:
            # This line will now work correctly with the updated CSV.
            critical_pending_df = labs_all[
                (labs_all['test_status'] == 'Pending') & (labs_all['is_critical'])
            ]
            render_metric_card(
                "Pending Critical Tests",
                {'current_mean': len(critical_pending_df)},
                kpi_format="{:,.0f}")
        st.divider()

        st.subheader("Performance Analysis")
        col1, col2 = st.columns(2)
        with col1:
            tat_trend = labs_all.set_index('result_date')['turn_around_time_days'].resample('W').mean()
            st.plotly_chart(plot_kpi_trend(tat_trend, "Weekly Avg. Test Turnaround Time", "Avg. TAT (Days)"), use_container_width=True)

        with col2:
            reasons_df = labs_current[labs_current['is_rejected']]['rejection_reason'].value_counts().reset_index()
            reasons_df.columns = ['reason', 'count']
            st.plotly_chart(
                plot_categorical_distribution(
                    reasons_df.head(5), y_col='reason', x_col='count', title="Top 5 Rejection Reasons", orientation='h'), 
                use_container_width=True)

    def _render_supply_forecast(self):
        """Renders the Supply Chain tab with predictive forecasting."""
        df_supply = self.all_data.get('supply')
        if df_supply is None or df_supply.empty:
            st.info("Supply utilization data not available.")
            return
            
        st.subheader("Key Item Consumption Forecast")
        all_items = sorted(df_supply['item_name'].unique())
        selected_item = st.selectbox("Select Supply Item to Forecast", all_items)
        
        if selected_item:
            item_series = df_supply[df_supply['item_name'] == selected_item].set_index('report_date')['consumption_count']
            daily_series = item_series.resample('D').sum().fillna(0)
            
            with st.spinner(f"Generating forecast for {selected_item}..."):
                forecast_df = forecast_supply_demand(daily_series, forecast_days=60)
            
            if forecast_df is None:
                st.warning(f"Could not generate forecast for {selected_item}. Insufficient data.")
                st.plotly_chart(create_empty_figure("Consumption Forecast"), use_container_width=True)
                return

            st.plotly_chart(plot_supply_forecast(forecast_df, f"60-Day Consumption Forecast: {selected_item}", "Predicted Consumption"), use_container_width=True)
            
            latest_stock_row = df_supply[df_supply['item_name'] == selected_item].sort_values('report_date').iloc[-1]
            latest_stock = latest_stock_row['stock_on_hand']
            forecast_df['cumulative_consumption'] = forecast_df['yhat'].clip(lower=0).cumsum()
            stockout_df = forecast_df[forecast_df['cumulative_consumption'] >= latest_stock]
            
            st.subheader("Supply Status Analysis")
            sc_col1, sc_col2 = st.columns(2)
            sc_col1.metric("Current Stock on Hand", f"{int(latest_stock):,}")
            if not stockout_df.empty:
                stockout_date = stockout_df.iloc[0]['ds'].strftime('%d %b %Y')
                sc_col2.metric("Predicted Stockout Date", stockout_date, delta="ACTION REQUIRED", delta_color="inverse")
            else:
                sc_col2.metric("Predicted Stockout Date", "Beyond 60 Days", delta_color="off")
    
    def run(self):
        """Main method to render the entire dashboard page."""
        render_main_header("Clinic Operations Console", "Real-time laboratory, supply chain, and patient flow monitoring")
        labs_df = self.filtered_data.get('labs', pd.DataFrame())

        if labs_df.empty:
            st.info("No lab data available for the selected period.")
            return

        tab_labs, tab_supply = st.tabs(["🔬 Laboratory Performance", "💊 Supply Chain"])
        with tab_labs:
            self._render_lab_kpis(labs_df, self.all_data.get('labs', pd.DataFrame()))
        with tab_supply:
            self._render_supply_forecast()

if __name__ == "__main__":
    dashboard = ClinicDashboard()
    dashboard.run()
