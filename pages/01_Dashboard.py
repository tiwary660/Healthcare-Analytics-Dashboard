import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualizations import HealthcareVisualizations
from utils.advanced_visualizations import AdvancedHealthcareVisualizations

st.set_page_config(
    page_title="Healthcare Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize components
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()
if 'advanced_visualizer' not in st.session_state:
    st.session_state.advanced_visualizer = AdvancedHealthcareVisualizations()

st.title("📊 Healthcare Analytics Dashboard")
st.markdown("---")

# Load sample data if not available
if 'healthcare_data' not in st.session_state:
    with st.spinner("Loading sample healthcare data..."):
        st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(1000)

df = st.session_state.healthcare_data

# Calculate metrics
metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(df)

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Patients",
        value=f"{metrics['total_patients']:,}",
        delta=f"+{np.random.randint(50, 200)} this month"
    )

with col2:
    st.metric(
        label="Average Length of Stay",
        value=f"{metrics['avg_length_of_stay']:.1f} days",
        delta=f"{np.random.uniform(-0.5, 0.3):.1f} days"
    )

with col3:
    st.metric(
        label="Readmission Rate",
        value=f"{metrics['readmission_rate']:.1f}%",
        delta=f"{np.random.uniform(-2, 1):.1f}%"
    )

with col4:
    st.metric(
        label="Hospital Efficiency",
        value=f"{metrics['efficiency_score']:.1f}%",
        delta=f"{np.random.uniform(-1, 3):.1f}%"
    )

st.markdown("---")

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    demo_chart = st.session_state.visualizer.create_patient_demographics_chart(df)
    if demo_chart:
        st.plotly_chart(demo_chart, use_container_width=True)
    else:
        st.warning("Unable to create demographics chart. Missing age data.")

with col2:
    st.subheader("Medical Conditions Distribution")
    condition_chart = st.session_state.visualizer.create_condition_distribution_chart(df)
    if condition_chart:
        st.plotly_chart(condition_chart, use_container_width=True)
    else:
        st.warning("Unable to create conditions chart. Missing condition data.")

# Advanced Dashboard Options
dashboard_view = st.selectbox(
    "Select Dashboard View:",
    ["Standard View", "Advanced Gauges", "Correlation Analysis", "Flow Diagram"]
)

if dashboard_view == "Standard View":
    # Hospital efficiency dashboard
    st.subheader("Hospital Efficiency Metrics")
    efficiency_dashboard = st.session_state.visualizer.create_hospital_efficiency_dashboard(metrics)
    if efficiency_dashboard:
        st.plotly_chart(efficiency_dashboard, use_container_width=True)

elif dashboard_view == "Advanced Gauges":
    st.subheader("Performance Gauge Dashboard")
    gauge_dashboard = st.session_state.advanced_visualizer.create_gauge_dashboard(metrics)
    if gauge_dashboard:
        st.plotly_chart(gauge_dashboard, use_container_width=True)

elif dashboard_view == "Correlation Analysis":
    st.subheader("Healthcare Metrics Correlation Matrix")
    correlation_heatmap = st.session_state.advanced_visualizer.create_heatmap_correlation(df)
    if correlation_heatmap:
        st.plotly_chart(correlation_heatmap, use_container_width=True)
    else:
        st.warning("Correlation analysis requires more numeric data columns.")

elif dashboard_view == "Flow Diagram":
    st.subheader("Patient Flow Analysis")
    flow_diagram = st.session_state.advanced_visualizer.create_advanced_patient_flow_diagram(df)
    if flow_diagram:
        st.plotly_chart(flow_diagram, use_container_width=True)
    else:
        st.warning("Flow diagram requires condition and length of stay data.")

# Length of stay analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Length of Stay Analysis")
    los_chart = st.session_state.visualizer.create_length_of_stay_analysis(df)
    if los_chart:
        st.plotly_chart(los_chart, use_container_width=True)
    else:
        st.warning("Unable to create length of stay chart.")

with col2:
    st.subheader("Patient Admissions Timeline")
    timeline_chart = st.session_state.visualizer.create_patient_timeline_chart(df)
    if timeline_chart:
        st.plotly_chart(timeline_chart, use_container_width=True)
    else:
        st.warning("Unable to create timeline chart.")

# Data summary
st.markdown("---")
st.subheader("Data Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.write("**Age Statistics**")
    if 'age' in df.columns:
        st.write(f"• Mean Age: {df['age'].mean():.1f} years")
        st.write(f"• Median Age: {df['age'].median():.1f} years")
        st.write(f"• Age Range: {df['age'].min()}-{df['age'].max()} years")
    else:
        st.write("Age data not available")

with summary_col2:
    st.write("**Gender Distribution**")
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"• {gender}: {count} ({percentage:.1f}%)")
    else:
        st.write("Gender data not available")

with summary_col3:
    st.write("**Medical Conditions**")
    if 'primary_condition' in df.columns:
        condition_counts = df['primary_condition'].value_counts().head(3)
        for condition, count in condition_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"• {condition}: {count} ({percentage:.1f}%)")
    else:
        st.write("Condition data not available")

# Refresh data button
st.markdown("---")
if st.button("🔄 Refresh Dashboard Data", use_container_width=True):
    # Generate new sample data
    st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(
        np.random.randint(800, 1200)
    )
    st.rerun()

# Export functionality
st.markdown("---")
st.subheader("Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Export Current Data as CSV"):
        csv_data = st.session_state.data_processor.export_to_csv(df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"healthcare_dashboard_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📊 Export Summary Report"):
        summary_report = f"""
Healthcare Analytics Dashboard Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Metrics:
- Total Patients: {metrics['total_patients']:,}
- Average Length of Stay: {metrics['avg_length_of_stay']:.1f} days
- Readmission Rate: {metrics['readmission_rate']:.1f}%
- Hospital Efficiency Score: {metrics['efficiency_score']:.1f}%

Patient Demographics:
- Age Range: {df['age'].min()}-{df['age'].max()} years
- Mean Age: {df['age'].mean():.1f} years
- Gender Distribution: {dict(df['gender'].value_counts())}

Top Medical Conditions:
{dict(df['primary_condition'].value_counts().head(5))}
        """
        
        st.download_button(
            label="Download Report",
            data=summary_report,
            file_name=f"healthcare_summary_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
