import streamlit as st
import pandas as pd

# Configure page
st.set_page_config(
    page_title="About - Healthcare Analytics Dashboard",
    page_icon="📖",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .about-header {
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
    
    .usage-step {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .tech-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="about-header">
    <h1>📖 About Healthcare Analytics Dashboard</h1>
    <p>Comprehensive documentation and usage guide</p>
</div>
""", unsafe_allow_html=True)

# Project Overview
st.header("🏥 Project Overview")

st.markdown("""
<div class="feature-card">
<h3>What is this Healthcare Analytics Dashboard?</h3>

The Healthcare Analytics Dashboard is an advanced data analysis platform designed specifically for healthcare professionals, hospital administrators, and data scientists working in the medical field. This comprehensive tool combines machine learning, interactive visualizations, and predictive analytics to help healthcare organizations make informed decisions about patient care, resource allocation, and operational efficiency.

<h4>Key Benefits:</h4>
<ul>
<li><strong>Predictive Patient Care:</strong> Identify high-risk patients before complications occur</li>
<li><strong>Resource Optimization:</strong> Optimize bed allocation and staff scheduling based on data insights</li>
<li><strong>Operational Efficiency:</strong> Monitor hospital performance metrics in real-time</li>
<li><strong>Cost Reduction:</strong> Reduce readmission rates through predictive modeling</li>
<li><strong>Quality Improvement:</strong> Track patient outcomes and identify improvement opportunities</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Who Should Use This
st.header("👥 Who Should Use This Dashboard?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h4>🏥 Healthcare Administrators</h4>
    <ul>
    <li>Monitor hospital efficiency metrics</li>
    <li>Track patient satisfaction scores</li>
    <li>Analyze readmission patterns</li>
    <li>Optimize resource allocation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <h4>👩‍⚕️ Clinical Staff</h4>
    <ul>
    <li>Identify high-risk patients</li>
    <li>Predict length of stay</li>
    <li>Monitor patient outcomes</li>
    <li>Access predictive insights</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="feature-card">
    <h4>📊 Data Scientists</h4>
    <ul>
    <li>Analyze healthcare datasets</li>
    <li>Build predictive models</li>
    <li>Create advanced visualizations</li>
    <li>Generate automated reports</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
    <h4>💼 Decision Makers</h4>
    <ul>
    <li>Access executive dashboards</li>
    <li>Review performance metrics</li>
    <li>Make data-driven decisions</li>
    <li>Monitor strategic initiatives</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Features Overview
st.header("🚀 Key Features")

feature_tabs = st.tabs(["Dashboard Analytics", "Risk Assessment", "Data Management", "Reporting", "Advanced Analytics"])

with feature_tabs[0]:
    st.markdown("""
    <div class="feature-card">
    <h4>📊 Interactive Dashboard</h4>
    <ul>
    <li><strong>Real-time Metrics:</strong> Live patient count, readmission rates, and satisfaction scores</li>
    <li><strong>Visual Analytics:</strong> Interactive charts for patient demographics and outcomes</li>
    <li><strong>Performance Monitoring:</strong> Hospital efficiency and operational metrics</li>
    <li><strong>Trend Analysis:</strong> Historical data visualization and pattern recognition</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_tabs[1]:
    st.markdown("""
    <div class="feature-card">
    <h4>🎯 Machine Learning Risk Assessment</h4>
    <ul>
    <li><strong>Readmission Prediction:</strong> Random Forest model to predict patient readmission risk</li>
    <li><strong>Length of Stay Forecasting:</strong> Gradient Boosting model for hospital stay duration</li>
    <li><strong>Risk Scoring:</strong> Comprehensive patient risk assessment with confidence intervals</li>
    <li><strong>Model Performance:</strong> Real-time accuracy metrics and feature importance analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_tabs[2]:
    st.markdown("""
    <div class="feature-card">
    <h4>📁 Data Upload and Processing</h4>
    <ul>
    <li><strong>File Support:</strong> CSV and Excel file uploads with automatic validation</li>
    <li><strong>Data Quality Checks:</strong> Automated data cleaning and preprocessing</li>
    <li><strong>Sample Data Generation:</strong> Built-in realistic healthcare dataset creation</li>
    <li><strong>Data Integration:</strong> Seamless integration with existing hospital systems</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_tabs[3]:
    st.markdown("""
    <div class="feature-card">
    <h4>📋 Automated Reporting</h4>
    <ul>
    <li><strong>Executive Reports:</strong> High-level summaries for leadership teams</li>
    <li><strong>Clinical Reports:</strong> Detailed patient analysis for medical staff</li>
    <li><strong>Operational Reports:</strong> Efficiency metrics and resource utilization</li>
    <li><strong>Export Options:</strong> PDF, CSV, and image export capabilities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_tabs[4]:
    st.markdown("""
    <div class="feature-card">
    <h4>🔬 Advanced Analytics & Presentations</h4>
    <ul>
    <li><strong>3D Visualizations:</strong> Interactive 3D scatter plots for complex data analysis</li>
    <li><strong>Sankey Diagrams:</strong> Patient flow and care pathway visualization</li>
    <li><strong>Animated Timelines:</strong> Dynamic patient journey tracking</li>
    <li><strong>Gauge Dashboards:</strong> Real-time performance indicators</li>
    <li><strong>Presentation Mode:</strong> Interactive slides with auto-advance functionality</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# How to Use Guide
st.header("📚 How to Use This Dashboard")

st.markdown("""
<div class="usage-step">
<h4>Step 1: Getting Started</h4>
<p><strong>Navigate to the Dashboard:</strong> Start by exploring the main dashboard to get an overview of your healthcare data. The dashboard shows key metrics, patient demographics, and performance indicators.</p>
</div>

<div class="usage-step">
<h4>Step 2: Upload Your Data</h4>
<p><strong>Go to Data Upload page:</strong> Upload your patient data in CSV or Excel format. The system will automatically validate and process your data. If you don't have data, use the sample data generator.</p>
</div>

<div class="usage-step">
<h4>Step 3: Risk Assessment</h4>
<p><strong>Use the Risk Assessment page:</strong> Train machine learning models on your data to predict patient readmission risks and length of stay. View model performance metrics and feature importance.</p>
</div>

<div class="usage-step">
<h4>Step 4: Generate Reports</h4>
<p><strong>Create Reports:</strong> Use the Reports page to generate executive summaries, clinical reports, and operational analyses. Export reports as PDFs or download charts as images.</p>
</div>

<div class="usage-step">
<h4>Step 5: Advanced Analytics</h4>
<p><strong>Explore Advanced Features:</strong> Visit the Advanced Analytics page for 3D visualizations, interactive presentations, and comprehensive data analysis tools.</p>
</div>
""", unsafe_allow_html=True)

# Technical Specifications
st.header("⚙️ Technical Specifications")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h4>🛠️ Technology Stack</h4>
    <div>
    <span class="tech-badge">Python 3.11</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">Pandas</span>
    <span class="tech-badge">NumPy</span>
    <span class="tech-badge">Scikit-learn</span>
    <span class="tech-badge">Plotly</span>
    <span class="tech-badge">Joblib</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <h4>🎯 ML Models</h4>
    <ul>
    <li><strong>Random Forest:</strong> Readmission prediction</li>
    <li><strong>Gradient Boosting:</strong> Length of stay forecasting</li>
    <li><strong>Feature Engineering:</strong> Automated preprocessing</li>
    <li><strong>Model Validation:</strong> Cross-validation and metrics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Data Requirements
st.header("📋 Data Requirements")

st.markdown("""
<div class="feature-card">
<h4>Required Data Fields for Optimal Performance:</h4>

<strong>Patient Demographics:</strong>
<ul>
<li>Age, Gender, Insurance Type</li>
<li>Admission Source and Type</li>
<li>Patient ID (anonymized)</li>
</ul>

<strong>Clinical Information:</strong>
<ul>
<li>Primary Diagnosis, Medical Conditions</li>
<li>Length of Stay, Readmission Status</li>
<li>Severity Score, Risk Factors</li>
</ul>

<strong>Operational Data:</strong>
<ul>
<li>Admission and Discharge Dates</li>
<li>Department/Unit Information</li>
<li>Cost and Billing Data (optional)</li>
</ul>

<strong>Outcome Metrics:</strong>
<ul>
<li>Patient Satisfaction Scores</li>
<li>Readmission within 30 days</li>
<li>Complications or Adverse Events</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Support and Contact
st.header("🤝 Support and Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h4>📖 Best Practices</h4>
    <ul>
    <li><strong>Data Quality:</strong> Ensure data is clean and complete before upload</li>
    <li><strong>Regular Updates:</strong> Refresh data regularly for accurate predictions</li>
    <li><strong>Model Retraining:</strong> Retrain models with new data for better performance</li>
    <li><strong>User Training:</strong> Ensure staff understand how to interpret results</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <h4>⚠️ Important Notes</h4>
    <ul>
    <li><strong>Data Privacy:</strong> Ensure all patient data is properly anonymized</li>
    <li><strong>HIPAA Compliance:</strong> Follow healthcare data protection regulations</li>
    <li><strong>Model Limitations:</strong> ML predictions are aids, not replacements for clinical judgment</li>
    <li><strong>Regular Monitoring:</strong> Monitor model performance and accuracy over time</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Version Information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <h4>Healthcare Analytics Dashboard v2.0</h4>
    <p>Enhanced with Advanced ML Analytics and Interactive Visualizations</p>
    <p><strong>Last Updated:</strong> August 2025 | <strong>Built with:</strong> Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)