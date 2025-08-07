import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class HealthcareVisualizations:
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
    
    def create_patient_demographics_chart(self, df):
        """Create patient demographics visualization"""
        if 'age' not in df.columns:
            return None
        
        # Age distribution
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ['<18', '18-29', '30-44', '45-59', '60-74', '75+']
        df_copy = df.copy()
        df_copy['age_group'] = pd.cut(df_copy['age'], bins=age_bins, labels=age_labels, right=False)
        age_dist = df_copy['age_group'].value_counts().sort_index()
        
        fig = px.bar(
            x=age_dist.index,
            y=age_dist.values,
            title="Patient Age Distribution",
            labels={'x': 'Age Group', 'y': 'Number of Patients'},
            color=age_dist.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_condition_distribution_chart(self, df):
        """Create primary condition distribution chart"""
        if 'primary_condition' not in df.columns:
            return None
        
        condition_counts = df['primary_condition'].value_counts()
        
        fig = px.pie(
            values=condition_counts.values,
            names=condition_counts.index,
            title="Distribution of Primary Medical Conditions",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_risk_assessment_chart(self, risk_data):
        """Create risk assessment visualization"""
        if not risk_data:
            return None
        
        # Extract risk distribution
        risk_counts = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
        
        if isinstance(risk_data, list):
            for patient in risk_data:
                risk_category = patient.get('risk_category', 'Low Risk')
                risk_counts[risk_category] += 1
        else:
            risk_category = risk_data.get('risk_category', 'Low Risk')
            risk_counts[risk_category] = 1
        
        categories = list(risk_counts.keys())
        values = list(risk_counts.values())
        colors = [self.risk_colors[cat] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Patient Risk Assessment Distribution",
            xaxis_title="Risk Category",
            yaxis_title="Number of Patients",
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_length_of_stay_analysis(self, df):
        """Create length of stay analysis chart"""
        if 'length_of_stay' not in df.columns:
            return None
        
        # Create histogram
        fig = px.histogram(
            df,
            x='length_of_stay',
            nbins=20,
            title="Length of Stay Distribution",
            labels={'length_of_stay': 'Length of Stay (days)', 'count': 'Number of Patients'},
            color_discrete_sequence=[self.color_palette['primary']]
        )
        
        # Add mean line
        mean_los = df['length_of_stay'].mean()
        fig.add_vline(
            x=mean_los,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_los:.1f} days"
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_hospital_efficiency_dashboard(self, metrics):
        """Create hospital efficiency metrics dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Efficiency Score Over Time',
                'Length of Stay vs Readmission Rate',
                'Patient Volume Trend',
                'Key Performance Indicators'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Generate sample time series data for efficiency
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='ME')
        efficiency_scores = np.random.normal(metrics.get('efficiency_score', 85), 5, len(dates))
        efficiency_scores = np.clip(efficiency_scores, 70, 95)
        
        # Efficiency score over time
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=efficiency_scores,
                mode='lines+markers',
                name='Efficiency Score',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # LOS vs Readmission scatter
        los_values = np.random.normal(metrics.get('avg_length_of_stay', 4.2), 2, 50)
        readmission_values = np.random.normal(metrics.get('readmission_rate', 8.7), 3, 50)
        
        fig.add_trace(
            go.Scatter(
                x=los_values,
                y=readmission_values,
                mode='markers',
                name='Hospitals',
                marker=dict(
                    size=8,
                    color=self.color_palette['secondary'],
                    opacity=0.7
                )
            ),
            row=1, col=2
        )
        
        # Patient volume trend
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        patient_volumes = np.random.poisson(1000, 12) + 500
        
        fig.add_trace(
            go.Bar(
                x=months,
                y=patient_volumes,
                name='Patient Volume',
                marker_color=self.color_palette['success']
            ),
            row=2, col=1
        )
        
        # KPI Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics.get('efficiency_score', 85),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Efficiency"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title="Hospital Efficiency Dashboard"
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_df):
        """Create feature importance chart for ML model"""
        if importance_df is None or importance_df.empty:
            return None
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Model Feature Importance",
            labels={'importance': 'Importance Score', 'feature': 'Features'},
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_patient_timeline_chart(self, df):
        """Create patient admission timeline"""
        if 'admission_date' not in df.columns:
            return None
        
        # Aggregate by month
        df_copy = df.copy()
        df_copy['admission_month'] = pd.to_datetime(df_copy['admission_date']).dt.to_period('M')
        monthly_admissions = df_copy.groupby('admission_month').size().reset_index(name='admissions')
        monthly_admissions['admission_month'] = monthly_admissions['admission_month'].astype(str)
        
        fig = px.line(
            monthly_admissions,
            x='admission_month',
            y='admissions',
            title="Patient Admissions Over Time",
            labels={'admission_month': 'Month', 'admissions': 'Number of Admissions'},
            markers=True
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_risk_score_distribution(self, risk_predictions):
        """Create risk score distribution chart"""
        if not risk_predictions:
            return None
        
        risk_scores = []
        if isinstance(risk_predictions, list):
            risk_scores = [pred.get('risk_score', 0) for pred in risk_predictions]
        else:
            risk_scores = [risk_predictions.get('risk_score', 0)]
        
        fig = px.histogram(
            x=risk_scores,
            nbins=20,
            title="Risk Score Distribution",
            labels={'x': 'Risk Score (%)', 'y': 'Number of Patients'},
            color_discrete_sequence=[self.color_palette['warning']]
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
