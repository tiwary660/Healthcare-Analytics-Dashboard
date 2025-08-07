import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class AdvancedHealthcareVisualizations:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.risk_colors = {
            'Low Risk': '#2ca02c',
            'Medium Risk': '#ff7f0e',
            'High Risk': '#d62728'
        }
    
    def create_advanced_patient_flow_diagram(self, df):
        """Create an advanced patient flow visualization"""
        if 'primary_condition' not in df.columns or 'length_of_stay' not in df.columns:
            return None
        
        df_copy = df.copy()
        df_copy['los_category'] = pd.cut(
            df_copy['length_of_stay'], 
            bins=[0, 2, 5, 10, float('inf')], 
            labels=['Short Stay (1-2)', 'Medium Stay (3-5)', 'Long Stay (6-10)', 'Extended Stay (10+)']
        )
        
        flow_data = df_copy.groupby(['primary_condition', 'los_category'], observed=True).size().reset_index(name='count')
        
        conditions = flow_data['primary_condition'].unique()
        categories = flow_data['los_category'].unique()
        
        condition_indices = {cond: i for i, cond in enumerate(conditions)}
        category_indices = {cat: i + len(conditions) for i, cat in enumerate(categories)}
        
        node_labels = list(conditions) + list(categories)
        node_colors = ['lightblue'] * len(conditions) + ['lightcoral'] * len(categories)
        
        source = []
        target = []
        value = []
        
        for _, row in flow_data.iterrows():
            source.append(condition_indices[row['primary_condition']])
            target.append(category_indices[row['los_category']])
            value.append(row['count'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color='rgba(255, 165, 0, 0.3)'
            )
        )])
        
        fig.update_layout(
            title="Patient Flow: Conditions to Length of Stay",
            font_size=12,
            height=500
        )
        
        return fig
    
    def create_3d_risk_scatter(self, df, risk_predictions=None):
        """Create 3D scatter plot for risk assessment"""
        if 'age' not in df.columns or 'bmi' not in df.columns or 'systolic_bp' not in df.columns:
            return None
        
        if risk_predictions:
            risk_categories = [pred.get('risk_category', 'Low Risk') for pred in risk_predictions]
            df_plot = df.copy()
            df_plot['risk_category'] = risk_categories
        else:
            df_plot = df.copy()
            df_plot['risk_category'] = 'Unknown'
        
        fig = px.scatter_3d(
            df_plot,
            x='age',
            y='bmi',
            z='systolic_bp',
            color='risk_category',
            color_discrete_map=self.risk_colors,
            title="3D Patient Risk Analysis",
            labels={
                'age': 'Age (years)',
                'bmi': 'BMI',
                'systolic_bp': 'Systolic BP (mmHg)'
            },
            opacity=0.7,
            size_max=10
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Age',
                yaxis_title='BMI',
                zaxis_title='Systolic BP',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_heatmap_correlation(self, df):
        """Create correlation heatmap for numerical variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Healthcare Metrics Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(
            height=500,
            margin=dict(t=100, b=50, l=100, r=50)
        )
        
        return fig
    
    def create_animated_timeline(self, df):
        """Create animated timeline of admissions"""
        if 'admission_date' not in df.columns:
            return None
        
        df_copy = df.copy()
        df_copy['admission_date'] = pd.to_datetime(df_copy['admission_date'])
        df_copy['month'] = df_copy['admission_date'].dt.to_period('M').astype(str)
        
        monthly_data = df_copy.groupby(['month', 'primary_condition']).size().reset_index(name='admissions')
        
        fig = px.bar(
            monthly_data,
            x='primary_condition',
            y='admissions',
            animation_frame='month',
            color='primary_condition',
            title="Monthly Admissions by Condition (Animated)",
            labels={'admissions': 'Number of Admissions', 'primary_condition': 'Medical Condition'}
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_gauge_dashboard(self, metrics):
        """Create a comprehensive gauge dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=('Hospital Efficiency', 'Patient Satisfaction', 'Readmission Rate', 'Bed Utilization')
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics.get('efficiency_score', 85),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Efficiency Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        satisfaction_score = np.random.uniform(80, 95)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=satisfaction_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Satisfaction"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 85], 'color': "yellow"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('readmission_rate', 8.7),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Readmission %"},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "lightcoral"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        bed_utilization = np.random.uniform(75, 90)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=bed_utilization,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Bed Utilization %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "lightblue"},
                        {'range': [85, 100], 'color': "lightcoral"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title="Healthcare Performance Dashboard")
        
        return fig
    
    def create_treemap_conditions(self, df):
        """Create treemap visualization for medical conditions"""
        if 'primary_condition' not in df.columns:
            return None
        
        condition_counts = df['primary_condition'].value_counts().reset_index()
        condition_counts.columns = ['condition', 'count']
        
        fig = px.treemap(
            condition_counts,
            path=['condition'],
            values='count',
            title="Medical Conditions Distribution (Treemap)",
            color='count',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_violin_plot_analysis(self, df):
        """Create violin plot for age distribution by condition"""
        if 'age' not in df.columns or 'primary_condition' not in df.columns:
            return None
        
        fig = px.violin(
            df,
            x='primary_condition',
            y='age',
            box=True,
            title="Age Distribution by Medical Condition",
            color='primary_condition'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_radar_chart_patient_profile(self, patient_data):
        """Create radar chart for individual patient profile"""
        if not patient_data:
            return None
        
        # Normalize values to 0-100 scale
        categories = ['Age Score', 'BMI Score', 'BP Score', 'Risk Score', 'LOS Score']
        
        # Calculate normalized scores
        age_score = min(100, (patient_data.get('age', 0) / 100) * 100)
        bmi_score = min(100, (patient_data.get('bmi', 0) / 40) * 100)
        bp_score = min(100, (patient_data.get('systolic_bp', 0) / 200) * 100)
        risk_score = patient_data.get('risk_score', 0)
        los_score = min(100, (patient_data.get('predicted_los', 0) / 30) * 100)
        
        values = [age_score, bmi_score, bp_score, risk_score, los_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Patient Profile',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Patient Risk Profile Radar Chart",
            height=500
        )
        
        return fig
    
    def create_waterfall_chart(self, metrics):
        """Create waterfall chart for efficiency metrics"""
        categories = ['Base Score', 'LOS Impact', 'Readmission Impact', 'Volume Impact', 'Final Score']
        
        base_score = 70
        los_impact = 15 if metrics.get('avg_length_of_stay', 5) < 4 else -10
        readmission_impact = 10 if metrics.get('readmission_rate', 10) < 8 else -15
        volume_impact = 5
        
        values = [base_score, los_impact, readmission_impact, volume_impact, 
                 base_score + los_impact + readmission_impact + volume_impact]
        
        fig = go.Figure(go.Waterfall(
            name="Efficiency Calculation",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=categories,
            textposition="outside",
            text=[f"+{v}" if v > 0 else str(v) for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Hospital Efficiency Score Breakdown",
            showlegend=False,
            height=500
        )
        
        return fig