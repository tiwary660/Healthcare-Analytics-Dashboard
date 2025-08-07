import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import RiskAssessmentModel
from utils.visualizations import HealthcareVisualizations

# Configure page
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = RiskAssessmentModel()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()

# Main page content
st.title("🏥 Healthcare Analytics Dashboard")
st.markdown("---")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Patients",
        value="12,458",
        delta="156 this week"
    )

with col2:
    st.metric(
        label="Hospital Efficiency",
        value="87.3%",
        delta="2.1% improvement"
    )

with col3:
    st.metric(
        label="Average Length of Stay",
        value="4.2 days",
        delta="-0.3 days"
    )

with col4:
    st.metric(
        label="Readmission Rate",
        value="8.7%",
        delta="-1.2% reduction"
    )

st.markdown("---")

# Key Features Section
st.header("Dashboard Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Analytics Views")
    st.write("• **Dashboard**: Real-time healthcare metrics and KPIs")
    st.write("• **Risk Assessment**: ML-powered patient risk stratification")
    st.write("• **Data Upload**: Import your own healthcare datasets")
    st.write("• **Reports**: Generate and export analytical reports")

with col2:
    st.subheader("🤖 Machine Learning Features")
    st.write("• Patient risk scoring using advanced algorithms")
    st.write("• Predictive modeling for hospital efficiency")
    st.write("• Automated risk stratification (Low/Medium/High)")
    st.write("• Real-time model performance monitoring")

st.markdown("---")

# Quick Actions
st.header("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Dashboard", use_container_width=True):
        st.switch_page("pages/01_Dashboard.py")

with col2:
    if st.button("🎯 Risk Assessment", use_container_width=True):
        st.switch_page("pages/02_Risk_Assessment.py")

with col3:
    if st.button("📤 Upload Data", use_container_width=True):
        st.switch_page("pages/03_Data_Upload.py")

# System Status
st.markdown("---")
st.header("System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.success("✅ Data Pipeline: Active")

with status_col2:
    st.success("✅ ML Models: Online")

with status_col3:
    st.success("✅ Real-time Monitoring: Active")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Healthcare Analytics Dashboard v1.0 | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
