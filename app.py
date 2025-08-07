import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import RiskAssessmentModel
from utils.visualizations import HealthcareVisualizations
from utils.advanced_visualizations import AdvancedHealthcareVisualizations

# Configure page
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 3px solid #4CAF50;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #4CAF50;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .action-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = RiskAssessmentModel()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()
if 'advanced_visualizer' not in st.session_state:
    st.session_state.advanced_visualizer = AdvancedHealthcareVisualizations()

# Main page content with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🏥 Healthcare Analytics Dashboard</h1>
    <p>Advanced predictive modeling and interactive visualizations for patient risk assessment</p>
</div>
""", unsafe_allow_html=True)

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

# Usage Guide Section
st.markdown("---")
st.header("🚀 How to Get Started")

usage_tabs = st.tabs(["Quick Start", "Data Requirements", "Best Practices"])

with usage_tabs[0]:
    st.markdown("""
    <div class="feature-card">
    <h4>5-Minute Quick Start Guide:</h4>
    <ol>
    <li><strong>Upload Data:</strong> Go to 'Data Upload' page and upload your patient CSV/Excel file</li>
    <li><strong>View Analytics:</strong> Check the 'Dashboard' for instant insights and visualizations</li>
    <li><strong>Risk Assessment:</strong> Use 'Risk Assessment' to predict patient readmission risks</li>
    <li><strong>Generate Reports:</strong> Create professional reports in the 'Reports' section</li>
    <li><strong>Advanced Analysis:</strong> Explore 3D visualizations in 'Advanced Analytics'</li>
    </ol>
    <p><em>💡 Tip: Start with sample data if you don't have your own dataset yet!</em></p>
    </div>
    """, unsafe_allow_html=True)

with usage_tabs[1]:
    st.markdown("""
    <div class="feature-card">
    <h4>Required Data Columns:</h4>
    <ul>
    <li><strong>Patient Info:</strong> Age, Gender, Insurance_Type</li>
    <li><strong>Medical:</strong> Primary_Diagnosis, Medical_Conditions, Severity_Score</li>
    <li><strong>Administrative:</strong> Admission_Source, Admission_Type, Length_of_Stay</li>
    <li><strong>Outcomes:</strong> Readmission_Risk, Patient_Satisfaction</li>
    </ul>
    <p><em>🔧 The system automatically handles missing data and performs quality checks</em></p>
    </div>
    """, unsafe_allow_html=True)

with usage_tabs[2]:
    st.markdown("""
    <div class="feature-card">
    <h4>Best Practices for Healthcare Analytics:</h4>
    <ul>
    <li><strong>Data Privacy:</strong> Ensure all patient data is properly anonymized</li>
    <li><strong>Regular Updates:</strong> Upload fresh data weekly for accurate predictions</li>
    <li><strong>Model Training:</strong> Retrain ML models monthly with new patient data</li>
    <li><strong>Clinical Review:</strong> Always validate ML predictions with clinical expertise</li>
    <li><strong>Trend Monitoring:</strong> Track key metrics over time for operational insights</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick Actions
st.header("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Dashboard", use_container_width=True):
        st.switch_page("pages/01_Dashboard.py")
    if st.button("📤 Upload Data", use_container_width=True):
        st.switch_page("pages/03_Data_Upload.py")

with col2:
    if st.button("🎯 Risk Assessment", use_container_width=True):
        st.switch_page("pages/02_Risk_Assessment.py")
    if st.button("📋 Reports", use_container_width=True):
        st.switch_page("pages/04_Reports.py")

with col3:
    if st.button("🔬 Advanced Analytics", use_container_width=True):
        st.switch_page("pages/05_Advanced_Analytics.py")
    if st.button("📖 About & Guide", use_container_width=True):
        st.switch_page("pages/06_About.py")

# System Status with enhanced indicators
st.markdown("---")
st.header("System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; padding: 1rem; background: #e8f5e8; border-radius: 8px; margin: 0.5rem 0;">
        <span class="status-indicator status-active"></span>
        <strong>Data Pipeline: Active</strong>
    </div>
    """, unsafe_allow_html=True)

with status_col2:
    st.markdown("""
    <div style="display: flex; align-items: center; padding: 1rem; background: #e8f5e8; border-radius: 8px; margin: 0.5rem 0;">
        <span class="status-indicator status-active"></span>
        <strong>ML Models: Online</strong>
    </div>
    """, unsafe_allow_html=True)

with status_col3:
    st.markdown("""
    <div style="display: flex; align-items: center; padding: 1rem; background: #e8f5e8; border-radius: 8px; margin: 0.5rem 0;">
        <span class="status-indicator status-active"></span>
        <strong>Real-time Monitoring: Active</strong>
    </div>
    """, unsafe_allow_html=True)

# Quick Preview Section
st.markdown("---")
st.header("Quick Preview")

preview_option = st.selectbox(
    "Select a visualization to preview:",
    ["Performance Gauges", "3D Risk Analysis", "Correlation Matrix", "Patient Flow Diagram"]
)

if preview_option == "Performance Gauges":
    # Generate sample data if not available
    if 'healthcare_data' not in st.session_state:
        st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(500)
    
    sample_metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(
        st.session_state.healthcare_data
    )
    gauge_preview = st.session_state.advanced_visualizer.create_gauge_dashboard(sample_metrics)
    if gauge_preview:
        st.plotly_chart(gauge_preview, use_container_width=True)

elif preview_option == "3D Risk Analysis":
    if 'healthcare_data' not in st.session_state:
        st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(300)
    
    # Train model if needed for preview
    if not st.session_state.ml_model.is_trained:
        with st.spinner("Training model for preview..."):
            st.session_state.ml_model.train_models(st.session_state.healthcare_data)
    
    risk_predictions, _ = st.session_state.ml_model.predict_patient_risk(st.session_state.healthcare_data)
    scatter_3d_preview = st.session_state.advanced_visualizer.create_3d_risk_scatter(
        st.session_state.healthcare_data, risk_predictions
    )
    if scatter_3d_preview:
        st.plotly_chart(scatter_3d_preview, use_container_width=True)

elif preview_option == "Correlation Matrix":
    if 'healthcare_data' not in st.session_state:
        st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(500)
    
    correlation_preview = st.session_state.advanced_visualizer.create_heatmap_correlation(
        st.session_state.healthcare_data
    )
    if correlation_preview:
        st.plotly_chart(correlation_preview, use_container_width=True)

elif preview_option == "Patient Flow Diagram":
    if 'healthcare_data' not in st.session_state:
        st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(500)
    
    flow_preview = st.session_state.advanced_visualizer.create_advanced_patient_flow_diagram(
        st.session_state.healthcare_data
    )
    if flow_preview:
        st.plotly_chart(flow_preview, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>Healthcare Analytics Dashboard v2.0</h3>
        <p>Advanced ML-powered analytics with interactive visualizations</p>
        <p>Built with Streamlit | Enhanced with 3D visualizations, animated charts, and predictive modeling</p>
    </div>
    """,
    unsafe_allow_html=True
)
