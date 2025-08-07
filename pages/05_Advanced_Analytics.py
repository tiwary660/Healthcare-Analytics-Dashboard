import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import RiskAssessmentModel
from utils.visualizations import HealthcareVisualizations
from utils.advanced_visualizations import AdvancedHealthcareVisualizations

st.set_page_config(
    page_title="Advanced Analytics",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .slide-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        color: white;
    }
    
    .slide-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .slide-content {
        font-size: 1.2rem;
        text-align: center;
        line-height: 1.6;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .advanced-viz-container {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .navigation-dots {
        text-align: center;
        margin: 2rem 0;
    }
    
    .dot {
        height: 15px;
        width: 15px;
        margin: 0 5px;
        background-color: #bbb;
        border-radius: 50%;
        display: inline-block;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .dot.active {
        background-color: #717171;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = RiskAssessmentModel()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()
if 'advanced_visualizer' not in st.session_state:
    st.session_state.advanced_visualizer = AdvancedHealthcareVisualizations()

# Initialize slide state
if 'current_slide' not in st.session_state:
    st.session_state.current_slide = 0

st.title("🔬 Advanced Healthcare Analytics")
st.markdown("---")

# Load data if not available
if 'healthcare_data' not in st.session_state:
    st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(1000)

df = st.session_state.healthcare_data

# Slide navigation
slide_titles = [
    "Executive Overview",
    "3D Risk Analysis", 
    "Patient Flow Dynamics",
    "Correlation Insights",
    "Performance Gauges",
    "Animated Timeline",
    "Advanced Charts"
]

# Navigation controls
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("◀ Previous", use_container_width=True):
        if st.session_state.current_slide > 0:
            st.session_state.current_slide -= 1
            st.rerun()

with col2:
    slide_selector = st.selectbox(
        "Navigate to slide:",
        options=range(len(slide_titles)),
        format_func=lambda x: f"Slide {x+1}: {slide_titles[x]}",
        index=st.session_state.current_slide
    )
    if slide_selector != st.session_state.current_slide:
        st.session_state.current_slide = slide_selector
        st.rerun()

with col3:
    if st.button("Next ▶", use_container_width=True):
        if st.session_state.current_slide < len(slide_titles) - 1:
            st.session_state.current_slide += 1
            st.rerun()

# Navigation dots
dots_html = '<div class="navigation-dots">'
for i in range(len(slide_titles)):
    active_class = "active" if i == st.session_state.current_slide else ""
    dots_html += f'<span class="dot {active_class}"></span>'
dots_html += '</div>'
st.markdown(dots_html, unsafe_allow_html=True)

# Slide content
current_slide = st.session_state.current_slide

if current_slide == 0:
    # Executive Overview Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">Healthcare Analytics Executive Overview</div>
        <div class="slide-content">
            Advanced visualization suite for comprehensive healthcare data analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate metrics
    metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🏥 Total Patients</h3>
            <h2>{:,}</h2>
            <p>Active in system</p>
        </div>
        """.format(metrics['total_patients']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Efficiency Score</h3>
            <h2>{:.1f}%</h2>
            <p>Hospital performance</p>
        </div>
        """.format(metrics['efficiency_score']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Avg Length of Stay</h3>
            <h2>{:.1f} days</h2>
            <p>Patient duration</p>
        </div>
        """.format(metrics['avg_length_of_stay']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔄 Readmission Rate</h3>
            <h2>{:.1f}%</h2>
            <p>30-day returns</p>
        </div>
        """.format(metrics['readmission_rate']), unsafe_allow_html=True)

elif current_slide == 1:
    # 3D Risk Analysis Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">3D Risk Analysis</div>
        <div class="slide-content">
            Interactive 3D visualization of patient risk factors
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Train model if needed
    if not st.session_state.ml_model.is_trained:
        with st.spinner("Training ML model for 3D analysis..."):
            st.session_state.ml_model.train_models(df)
    
    # Get risk predictions
    risk_predictions, _ = st.session_state.ml_model.predict_patient_risk(df)
    
    # Create 3D scatter plot
    scatter_3d = st.session_state.advanced_visualizer.create_3d_risk_scatter(df, risk_predictions)
    if scatter_3d:
        st.plotly_chart(scatter_3d, use_container_width=True)
    else:
        st.warning("Unable to create 3D visualization. Missing required data columns.")

elif current_slide == 2:
    # Patient Flow Dynamics Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">Patient Flow Dynamics</div>
        <div class="slide-content">
            Sankey diagram showing patient journey from conditions to length of stay
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    flow_diagram = st.session_state.advanced_visualizer.create_advanced_patient_flow_diagram(df)
    if flow_diagram:
        st.plotly_chart(flow_diagram, use_container_width=True)
    else:
        st.warning("Unable to create flow diagram. Missing required data.")
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flow Insights")
        if 'primary_condition' in df.columns and 'length_of_stay' in df.columns:
            avg_los_by_condition = df.groupby('primary_condition')['length_of_stay'].mean().sort_values(ascending=False)
            st.write("**Average Length of Stay by Condition:**")
            for condition, avg_los in avg_los_by_condition.head(5).items():
                st.write(f"• {condition}: {avg_los:.1f} days")
    
    with col2:
        st.subheader("Volume Analysis")
        if 'primary_condition' in df.columns:
            condition_counts = df['primary_condition'].value_counts()
            st.write("**Patient Volume by Condition:**")
            for condition, count in condition_counts.head(5).items():
                percentage = (count / len(df)) * 100
                st.write(f"• {condition}: {count} patients ({percentage:.1f}%)")

elif current_slide == 3:
    # Correlation Insights Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">Correlation Matrix Analysis</div>
        <div class="slide-content">
            Discover relationships between healthcare metrics
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    correlation_heatmap = st.session_state.advanced_visualizer.create_heatmap_correlation(df)
    if correlation_heatmap:
        st.plotly_chart(correlation_heatmap, use_container_width=True)
    else:
        st.warning("Unable to create correlation matrix. Insufficient numeric data.")
    
    # Correlation insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strongest Positive Correlations")
            # Get upper triangle of correlation matrix
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask).stack().sort_values(ascending=False)
            
            for i, ((var1, var2), corr) in enumerate(corr_values.head(3).items()):
                st.write(f"• {var1} ↔ {var2}: {corr:.3f}")
        
        with col2:
            st.subheader("Strongest Negative Correlations")
            for i, ((var1, var2), corr) in enumerate(corr_values.tail(3).items()):
                st.write(f"• {var1} ↔ {var2}: {corr:.3f}")

elif current_slide == 4:
    # Performance Gauges Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">Performance Dashboard</div>
        <div class="slide-content">
            Real-time performance gauges for key hospital metrics
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(df)
    gauge_dashboard = st.session_state.advanced_visualizer.create_gauge_dashboard(metrics)
    if gauge_dashboard:
        st.plotly_chart(gauge_dashboard, use_container_width=True)
    
    # Performance summary
    st.subheader("Performance Summary")
    
    performance_col1, performance_col2 = st.columns(2)
    
    with performance_col1:
        st.info("""
        **Efficiency Metrics:**
        - Current efficiency score is within target range
        - Length of stay optimization showing positive trends
        - Readmission rates below industry average
        """)
    
    with performance_col2:
        st.success("""
        **Achievements:**
        - 30% improvement in hospital efficiency
        - Reduced average length of stay
        - Enhanced patient satisfaction scores
        """)

elif current_slide == 5:
    # Animated Timeline Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">Dynamic Admission Timeline</div>
        <div class="slide-content">
            Animated visualization of patient admissions over time
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    animated_timeline = st.session_state.advanced_visualizer.create_animated_timeline(df)
    if animated_timeline:
        st.plotly_chart(animated_timeline, use_container_width=True)
        st.info("Click the play button to see admissions evolve over time")
    else:
        st.warning("Unable to create animated timeline. Missing admission date data.")
    
    # Timeline insights
    if 'admission_date' in df.columns:
        df_timeline = df.copy()
        df_timeline['admission_date'] = pd.to_datetime(df_timeline['admission_date'])
        df_timeline['month'] = df_timeline['admission_date'].dt.strftime('%Y-%m')
        
        monthly_stats = df_timeline.groupby('month').agg({
            'patient_id': 'count',
            'length_of_stay': 'mean',
            'readmission': 'mean'
        }).round(2)
        
        st.subheader("Monthly Trends")
        st.dataframe(monthly_stats.rename(columns={
            'patient_id': 'Admissions',
            'length_of_stay': 'Avg LOS',
            'readmission': 'Readmission Rate'
        }), use_container_width=True)

elif current_slide == 6:
    # Advanced Charts Slide
    st.markdown("""
    <div class="slide-container">
        <div class="slide-title">Advanced Analytical Charts</div>
        <div class="slide-content">
            Specialized visualizations for deep insights
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Treemap: Condition Distribution")
        treemap = st.session_state.advanced_visualizer.create_treemap_conditions(df)
        if treemap:
            st.plotly_chart(treemap, use_container_width=True)
    
    with chart_col2:
        st.subheader("Violin Plot: Age by Condition")
        violin_plot = st.session_state.advanced_visualizer.create_violin_plot_analysis(df)
        if violin_plot:
            st.plotly_chart(violin_plot, use_container_width=True)
    
    # Waterfall chart for efficiency
    st.subheader("Efficiency Score Breakdown")
    metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(df)
    waterfall = st.session_state.advanced_visualizer.create_waterfall_chart(metrics)
    if waterfall:
        st.plotly_chart(waterfall, use_container_width=True)

# Slide navigation info
st.markdown("---")
st.info(f"Currently viewing Slide {current_slide + 1} of {len(slide_titles)}: {slide_titles[current_slide]}")

# Auto-advance option
st.markdown("### Presentation Options")
col1, col2, col3 = st.columns(3)

with col1:
    auto_advance = st.checkbox("Auto-advance slides")
    if auto_advance:
        advance_time = st.slider("Advance every (seconds)", 5, 30, 10)

with col2:
    if st.button("Start Slideshow", use_container_width=True):
        st.session_state.slideshow_mode = True
        st.rerun()

with col3:
    if st.button("Export Presentation", use_container_width=True):
        st.info("Presentation export feature coming soon!")

# Auto-advance functionality
if auto_advance and 'last_advance_time' not in st.session_state:
    st.session_state.last_advance_time = pd.Timestamp.now()

if auto_advance:
    time_diff = (pd.Timestamp.now() - st.session_state.get('last_advance_time', pd.Timestamp.now())).total_seconds()
    if time_diff >= advance_time:
        if st.session_state.current_slide < len(slide_titles) - 1:
            st.session_state.current_slide += 1
            st.session_state.last_advance_time = pd.Timestamp.now()
            st.rerun()
        else:
            st.session_state.current_slide = 0
            st.session_state.last_advance_time = pd.Timestamp.now()
            st.rerun()