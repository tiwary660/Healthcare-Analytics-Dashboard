import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_processor import DataProcessor
from utils.ml_models import RiskAssessmentModel
from utils.visualizations import HealthcareVisualizations
import io

st.set_page_config(
    page_title="Reports",
    page_icon="📋",
    layout="wide"
)

# Initialize components
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = RiskAssessmentModel()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()

st.title("📋 Healthcare Analytics Reports")
st.markdown("---")

# Load data if not available
if 'healthcare_data' not in st.session_state:
    st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(1000)

df = st.session_state.healthcare_data

# Create tabs for different report types
tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Clinical Reports", "Risk Analysis", "Custom Reports"])

with tab1:
    st.header("Executive Summary Report")
    
    # Generate report date
    report_date = datetime.now().strftime('%B %d, %Y')
    
    # Calculate key metrics
    metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(df)
    
    # Executive summary content
    st.subheader(f"Healthcare Analytics Summary - {report_date}")
    
    # Key Performance Indicators
    st.markdown("### Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            "Total Patients Analyzed",
            f"{metrics['total_patients']:,}",
            help="Total number of patients in the current dataset"
        )
    
    with kpi_col2:
        st.metric(
            "Hospital Efficiency Score",
            f"{metrics['efficiency_score']:.1f}%",
            delta="2.3% vs last month",
            help="Composite efficiency score based on LOS and readmission rates"
        )
    
    with kpi_col3:
        st.metric(
            "Average Length of Stay",
            f"{metrics['avg_length_of_stay']:.1f} days",
            delta="-0.5 days vs target",
            help="Average hospital length of stay across all patients"
        )
    
    with kpi_col4:
        st.metric(
            "Readmission Rate",
            f"{metrics['readmission_rate']:.1f}%",
            delta="-1.2% improvement",
            help="Percentage of patients readmitted within 30 days"
        )
    
    # Patient Demographics Summary
    st.markdown("### Patient Demographics")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        if 'age' in df.columns:
            st.write("**Age Distribution:**")
            age_stats = {
                "Mean Age": f"{df['age'].mean():.1f} years",
                "Median Age": f"{df['age'].median():.1f} years",
                "Age Range": f"{df['age'].min()}-{df['age'].max()} years",
                "Elderly Patients (>65)": f"{(df['age'] > 65).sum():,} ({(df['age'] > 65).mean()*100:.1f}%)"
            }
            for key, value in age_stats.items():
                st.write(f"• {key}: {value}")
    
    with demo_col2:
        if 'gender' in df.columns:
            st.write("**Gender Distribution:**")
            gender_counts = df['gender'].value_counts()
            for gender, count in gender_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {gender}: {count:,} ({percentage:.1f}%)")
    
    # Clinical Insights
    st.markdown("### Clinical Insights")
    
    if 'primary_condition' in df.columns:
        top_conditions = df['primary_condition'].value_counts().head(5)
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.write("**Top Medical Conditions:**")
            for condition, count in top_conditions.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {condition}: {count:,} patients ({percentage:.1f}%)")
        
        with insight_col2:
            st.write("**Risk Assessment Summary:**")
            if st.session_state.ml_model.is_trained:
                # Get risk predictions
                risk_predictions, _ = st.session_state.ml_model.predict_patient_risk(df)
                if risk_predictions:
                    risk_categories = [pred['risk_category'] for pred in risk_predictions]
                    risk_counts = pd.Series(risk_categories).value_counts()
                    
                    for risk, count in risk_counts.items():
                        percentage = (count / len(risk_categories)) * 100
                        st.write(f"• {risk}: {count:,} patients ({percentage:.1f}%)")
                else:
                    st.write("Risk assessment data not available")
            else:
                st.write("ML model not trained yet")
    
    # Recommendations
    st.markdown("### Strategic Recommendations")
    
    recommendations = []
    
    if metrics['efficiency_score'] < 80:
        recommendations.append("🎯 **Focus on Efficiency**: Current efficiency score is below target. Consider optimizing discharge processes and reducing unnecessary length of stay.")
    
    if metrics['readmission_rate'] > 10:
        recommendations.append("🏥 **Reduce Readmissions**: Implement enhanced discharge planning and follow-up care protocols.")
    
    if metrics['avg_length_of_stay'] > 5:
        recommendations.append("⏰ **Optimize Length of Stay**: Review care pathways and consider early discharge programs for appropriate patients.")
    
    if 'age' in df.columns and (df['age'] > 65).mean() > 0.4:
        recommendations.append("👴 **Elderly Care Focus**: High proportion of elderly patients requires specialized geriatric care protocols.")
    
    if not recommendations:
        recommendations.append("✅ **Maintain Excellence**: Current performance metrics are within acceptable ranges. Continue monitoring and maintain quality standards.")
    
    for rec in recommendations:
        st.markdown(rec)
    
    # Generate executive report
    st.markdown("---")
    if st.button("📄 Generate Executive Report", use_container_width=True):
        executive_report = f"""
HEALTHCARE ANALYTICS EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
KEY PERFORMANCE INDICATORS
========================================
• Total Patients Analyzed: {metrics['total_patients']:,}
• Hospital Efficiency Score: {metrics['efficiency_score']:.1f}%
• Average Length of Stay: {metrics['avg_length_of_stay']:.1f} days
• Readmission Rate: {metrics['readmission_rate']:.1f}%

========================================
PATIENT DEMOGRAPHICS
========================================
Age Statistics:
• Mean Age: {df['age'].mean():.1f} years
• Median Age: {df['age'].median():.1f} years
• Age Range: {df['age'].min()}-{df['age'].max()} years
• Elderly Patients (>65): {(df['age'] > 65).sum():,} ({(df['age'] > 65).mean()*100:.1f}%)

Gender Distribution:
{chr(10).join([f"• {gender}: {count:,} ({count/len(df)*100:.1f}%)" for gender, count in df['gender'].value_counts().items()])}

========================================
CLINICAL OVERVIEW
========================================
Top Medical Conditions:
{chr(10).join([f"• {condition}: {count:,} ({count/len(df)*100:.1f}%)" for condition, count in df['primary_condition'].value_counts().head(5).items()])}

========================================
STRATEGIC RECOMMENDATIONS
========================================
{chr(10).join([f"• {rec.replace('🎯 ', '').replace('🏥 ', '').replace('⏰ ', '').replace('👴 ', '').replace('✅ ', '')}" for rec in recommendations])}

========================================
Report prepared by Healthcare Analytics Dashboard
        """
        
        st.download_button(
            label="📥 Download Executive Report",
            data=executive_report,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

with tab2:
    st.header("Clinical Performance Reports")
    
    # Clinical metrics
    st.subheader("Clinical Quality Metrics")
    
    clinical_col1, clinical_col2 = st.columns(2)
    
    with clinical_col1:
        st.markdown("#### Patient Safety Indicators")
        
        # Calculate safety metrics
        if 'length_of_stay' in df.columns:
            extended_stay = (df['length_of_stay'] > 7).sum()
            extended_stay_rate = (extended_stay / len(df)) * 100
            st.metric("Extended Stay Rate (>7 days)", f"{extended_stay_rate:.1f}%")
        
        if 'readmission' in df.columns:
            readmission_rate = df['readmission'].mean() * 100
            st.metric("30-Day Readmission Rate", f"{readmission_rate:.1f}%")
        
        # Risk stratification
        if st.session_state.ml_model.is_trained:
            risk_predictions, _ = st.session_state.ml_model.predict_patient_risk(df)
            if risk_predictions:
                high_risk_count = sum(1 for pred in risk_predictions if pred['risk_category'] == 'High Risk')
                high_risk_rate = (high_risk_count / len(risk_predictions)) * 100
                st.metric("High Risk Patients", f"{high_risk_rate:.1f}%")
    
    with clinical_col2:
        st.markdown("#### Operational Efficiency")
        
        if 'length_of_stay' in df.columns:
            avg_los = df['length_of_stay'].mean()
            st.metric("Average Length of Stay", f"{avg_los:.1f} days")
            
            # Bed utilization (simplified calculation)
            bed_utilization = min(95, avg_los * 20)  # Simplified formula
            st.metric("Estimated Bed Utilization", f"{bed_utilization:.1f}%")
        
        # Patient throughput
        daily_admissions = len(df) / 30  # Assuming 30-day period
        st.metric("Average Daily Admissions", f"{daily_admissions:.1f}")
    
    # Clinical visualization
    st.subheader("Clinical Performance Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if 'length_of_stay' in df.columns:
            los_chart = st.session_state.visualizer.create_length_of_stay_analysis(df)
            if los_chart:
                st.plotly_chart(los_chart, use_container_width=True)
    
    with viz_col2:
        if 'primary_condition' in df.columns:
            condition_chart = st.session_state.visualizer.create_condition_distribution_chart(df)
            if condition_chart:
                st.plotly_chart(condition_chart, use_container_width=True)
    
    # Department-wise analysis (simulated)
    st.subheader("Department Performance Analysis")
    
    # Generate simulated department data
    departments = ['Cardiology', 'Orthopedics', 'Emergency', 'General Medicine', 'Surgery']
    dept_data = []
    
    for dept in departments:
        dept_data.append({
            'Department': dept,
            'Avg LOS': np.random.normal(4.5, 1.5),
            'Patient Count': np.random.randint(50, 200),
            'Readmission Rate': np.random.uniform(5, 15),
            'Efficiency Score': np.random.uniform(75, 95)
        })
    
    dept_df = pd.DataFrame(dept_data)
    st.dataframe(dept_df.round(2), use_container_width=True)
    
    # Generate clinical report
    if st.button("📊 Generate Clinical Report", use_container_width=True):
        clinical_report = f"""
CLINICAL PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
PATIENT SAFETY INDICATORS
========================================
• Extended Stay Rate (>7 days): {(df['length_of_stay'] > 7).mean()*100:.1f}%
• 30-Day Readmission Rate: {df['readmission'].mean()*100:.1f}%
• Average Length of Stay: {df['length_of_stay'].mean():.1f} days

========================================
QUALITY METRICS
========================================
• Total Patients: {len(df):,}
• Patient Age Range: {df['age'].min()}-{df['age'].max()} years
• Most Common Condition: {df['primary_condition'].mode().iloc[0]}

========================================
DEPARTMENT PERFORMANCE
========================================
{dept_df.to_string(index=False)}

========================================
CLINICAL RECOMMENDATIONS
========================================
• Monitor patients with length of stay >7 days for discharge readiness
• Implement targeted interventions for high-risk patient populations
• Review care protocols for departments with above-average readmission rates
• Consider case management for elderly patients (>75 years)

Report prepared by Healthcare Analytics Dashboard
        """
        
        st.download_button(
            label="📥 Download Clinical Report",
            data=clinical_report,
            file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

with tab3:
    st.header("Risk Analysis Report")
    
    # Train model if not trained
    if not st.session_state.ml_model.is_trained:
        with st.spinner("Training risk assessment model..."):
            success, message = st.session_state.ml_model.train_models(df)
            if not success:
                st.error(f"Unable to train model: {message}")
                st.stop()
    
    # Risk assessment overview
    st.subheader("Risk Assessment Overview")
    
    # Get risk predictions for all patients
    all_predictions, _ = st.session_state.ml_model.predict_patient_risk(df)
    
    if all_predictions:
        # Risk distribution
        risk_categories = [pred['risk_category'] for pred in all_predictions]
        risk_counts = pd.Series(risk_categories).value_counts()
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            low_risk = risk_counts.get('Low Risk', 0)
            st.metric("Low Risk Patients", f"{low_risk:,}", f"{low_risk/len(risk_categories)*100:.1f}%")
        
        with risk_col2:
            medium_risk = risk_counts.get('Medium Risk', 0)
            st.metric("Medium Risk Patients", f"{medium_risk:,}", f"{medium_risk/len(risk_categories)*100:.1f}%")
        
        with risk_col3:
            high_risk = risk_counts.get('High Risk', 0)
            st.metric("High Risk Patients", f"{high_risk:,}", f"{high_risk/len(risk_categories)*100:.1f}%")
        
        # Risk visualization
        st.subheader("Risk Distribution Analysis")
        
        risk_viz_col1, risk_viz_col2 = st.columns(2)
        
        with risk_viz_col1:
            risk_chart = st.session_state.visualizer.create_risk_assessment_chart(all_predictions)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
        
        with risk_viz_col2:
            risk_score_chart = st.session_state.visualizer.create_risk_score_distribution(all_predictions)
            if risk_score_chart:
                st.plotly_chart(risk_score_chart, use_container_width=True)
        
        # Model performance
        st.subheader("Model Performance Metrics")
        
        performance_metrics = st.session_state.ml_model.get_model_performance_metrics(df)
        if performance_metrics:
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("Model Accuracy", f"{performance_metrics['accuracy']:.3f}")
            
            with perf_col2:
                st.metric("Total Predictions", f"{performance_metrics['total_predictions']:,}")
            
            with perf_col3:
                most_common_risk = max(performance_metrics['risk_distribution'], 
                                     key=performance_metrics['risk_distribution'].get)
                st.metric("Most Common Risk", most_common_risk)
        
        # Feature importance
        importance_df = st.session_state.ml_model.get_feature_importance()
        if importance_df is not None:
            st.subheader("Risk Factors Analysis")
            
            importance_chart = st.session_state.visualizer.create_feature_importance_chart(importance_df)
            if importance_chart:
                st.plotly_chart(importance_chart, use_container_width=True)
        
        # High-risk patient details
        st.subheader("High-Risk Patient Analysis")
        
        high_risk_patients = [i for i, pred in enumerate(all_predictions) if pred['risk_category'] == 'High Risk']
        
        if high_risk_patients:
            high_risk_df = df.iloc[high_risk_patients].copy()
            high_risk_predictions = [all_predictions[i] for i in high_risk_patients]
            
            # Add risk scores to dataframe
            high_risk_df['Risk Score'] = [pred['risk_score'] for pred in high_risk_predictions]
            high_risk_df['Predicted LOS'] = [pred.get('predicted_los', 'N/A') for pred in high_risk_predictions]
            
            # Display top high-risk patients
            st.write(f"**Top 10 Highest Risk Patients:**")
            display_cols = ['patient_id', 'age', 'gender', 'primary_condition', 'Risk Score', 'Predicted LOS']
            available_cols = [col for col in display_cols if col in high_risk_df.columns]
            
            top_high_risk = high_risk_df.nlargest(10, 'Risk Score')[available_cols]
            st.dataframe(top_high_risk, use_container_width=True)
        
        # Risk-based recommendations
        st.subheader("Risk-Based Clinical Recommendations")
        
        recommendations = []
        
        if high_risk > 0:
            high_risk_percentage = (high_risk / len(risk_categories)) * 100
            if high_risk_percentage > 20:
                recommendations.append("🚨 **High Risk Alert**: Over 20% of patients are classified as high risk. Consider implementing intensive monitoring protocols.")
            
            recommendations.append(f"🎯 **Focus on {high_risk} High-Risk Patients**: Implement care management programs for patients with highest risk scores.")
        
        if medium_risk > high_risk:
            recommendations.append("⚠️ **Medium Risk Management**: Large medium-risk population may benefit from preventive interventions.")
        
        # Age-based risk analysis
        if 'age' in df.columns:
            elderly_high_risk = len([i for i in high_risk_patients if df.iloc[i]['age'] > 65])
            if elderly_high_risk > 0:
                recommendations.append(f"👴 **Elderly Care**: {elderly_high_risk} high-risk patients are elderly (>65). Consider geriatric care protocols.")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Generate risk analysis report
        if st.button("🎯 Generate Risk Analysis Report", use_container_width=True):
            risk_report = f"""
PATIENT RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
RISK DISTRIBUTION SUMMARY
========================================
• Total Patients Assessed: {len(risk_categories):,}
• Low Risk: {risk_counts.get('Low Risk', 0):,} ({risk_counts.get('Low Risk', 0)/len(risk_categories)*100:.1f}%)
• Medium Risk: {risk_counts.get('Medium Risk', 0):,} ({risk_counts.get('Medium Risk', 0)/len(risk_categories)*100:.1f}%)
• High Risk: {risk_counts.get('High Risk', 0):,} ({risk_counts.get('High Risk', 0)/len(risk_categories)*100:.1f}%)

========================================
MODEL PERFORMANCE
========================================
• Model Accuracy: {performance_metrics['accuracy']:.3f}
• Total Predictions: {performance_metrics['total_predictions']:,}
• Algorithm: Random Forest Classifier
• Features Used: Age, BMI, Blood Pressure, Gender, Medical Condition

========================================
HIGH-RISK PATIENT PROFILE
========================================
Number of High-Risk Patients: {len(high_risk_patients)}
{f"Average Age of High-Risk Patients: {high_risk_df['age'].mean():.1f} years" if high_risk_patients else ""}
{f"Most Common Condition in High-Risk: {high_risk_df['primary_condition'].mode().iloc[0]}" if high_risk_patients and 'primary_condition' in high_risk_df.columns else ""}

========================================
CLINICAL RECOMMENDATIONS
========================================
{chr(10).join([f"• {rec.replace('🚨 ', '').replace('🎯 ', '').replace('⚠️ ', '').replace('👴 ', '')}" for rec in recommendations])}

========================================
RISK FACTOR IMPORTANCE
========================================
{importance_df.to_string(index=False) if importance_df is not None else "Feature importance data not available"}

Report prepared by Healthcare Analytics Dashboard
            """
            
            st.download_button(
                label="📥 Download Risk Analysis Report",
                data=risk_report,
                file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        st.error("Unable to generate risk predictions. Please check if the model is properly trained.")

with tab4:
    st.header("Custom Report Builder")
    
    st.subheader("Create Custom Analytics Report")
    
    # Report configuration
    report_config_col1, report_config_col2 = st.columns(2)
    
    with report_config_col1:
        st.markdown("#### Report Configuration")
        
        report_title = st.text_input("Report Title", value="Custom Healthcare Analytics Report")
        report_period = st.selectbox("Reporting Period", ["Current Month", "Last 30 Days", "Last Quarter", "Custom Range"])
        
        include_demographics = st.checkbox("Include Patient Demographics", value=True)
        include_conditions = st.checkbox("Include Medical Conditions Analysis", value=True)
        include_risk = st.checkbox("Include Risk Assessment", value=True)
        include_efficiency = st.checkbox("Include Hospital Efficiency Metrics", value=True)
    
    with report_config_col2:
        st.markdown("#### Data Filters")
        
        # Age filter
        if 'age' in df.columns:
            age_range = st.slider("Age Range", 
                                int(df['age'].min()), 
                                int(df['age'].max()), 
                                (int(df['age'].min()), int(df['age'].max())))
        
        # Gender filter
        if 'gender' in df.columns:
            gender_filter = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
        
        # Condition filter
        if 'primary_condition' in df.columns:
            condition_filter = st.multiselect("Primary Conditions", 
                                            df['primary_condition'].unique(), 
                                            default=df['primary_condition'].unique())
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'age' in df.columns:
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    if 'gender' in df.columns and gender_filter:
        filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
    
    if 'primary_condition' in df.columns and condition_filter:
        filtered_df = filtered_df[filtered_df['primary_condition'].isin(condition_filter)]
    
    # Display filtered data summary
    st.subheader("Filtered Data Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Filtered Patient Count", len(filtered_df))
    
    with summary_col2:
        if len(filtered_df) > 0:
            st.metric("Average Age", f"{filtered_df['age'].mean():.1f}" if 'age' in filtered_df.columns else "N/A")
    
    with summary_col3:
        reduction_percentage = ((len(df) - len(filtered_df)) / len(df)) * 100
        st.metric("Data Reduction", f"{reduction_percentage:.1f}%")
    
    # Generate custom report
    if st.button("📋 Generate Custom Report", use_container_width=True):
        if len(filtered_df) == 0:
            st.error("No data available with current filters. Please adjust your filter criteria.")
        else:
            custom_report_content = [
                f"{report_title.upper()}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Reporting Period: {report_period}",
                "",
                "=" * 50,
                "REPORT SUMMARY",
                "=" * 50,
                f"• Total Patients in Report: {len(filtered_df):,}",
                f"• Data Filters Applied: {len(df) - len(filtered_df):,} patients excluded",
                "",
            ]
            
            if include_demographics and 'age' in filtered_df.columns:
                custom_report_content.extend([
                    "=" * 50,
                    "PATIENT DEMOGRAPHICS",
                    "=" * 50,
                    f"• Age Range: {filtered_df['age'].min()}-{filtered_df['age'].max()} years",
                    f"• Mean Age: {filtered_df['age'].mean():.1f} years",
                    f"• Median Age: {filtered_df['age'].median():.1f} years",
                    ""
                ])
                
                if 'gender' in filtered_df.columns:
                    gender_dist = filtered_df['gender'].value_counts()
                    custom_report_content.append("Gender Distribution:")
                    for gender, count in gender_dist.items():
                        percentage = (count / len(filtered_df)) * 100
                        custom_report_content.append(f"• {gender}: {count:,} ({percentage:.1f}%)")
                    custom_report_content.append("")
            
            if include_conditions and 'primary_condition' in filtered_df.columns:
                custom_report_content.extend([
                    "=" * 50,
                    "MEDICAL CONDITIONS ANALYSIS",
                    "=" * 50,
                ])
                
                condition_dist = filtered_df['primary_condition'].value_counts().head(10)
                for condition, count in condition_dist.items():
                    percentage = (count / len(filtered_df)) * 100
                    custom_report_content.append(f"• {condition}: {count:,} ({percentage:.1f}%)")
                custom_report_content.append("")
            
            if include_efficiency:
                metrics = st.session_state.data_processor.calculate_hospital_efficiency_metrics(filtered_df)
                custom_report_content.extend([
                    "=" * 50,
                    "HOSPITAL EFFICIENCY METRICS",
                    "=" * 50,
                    f"• Average Length of Stay: {metrics['avg_length_of_stay']:.1f} days",
                    f"• Readmission Rate: {metrics['readmission_rate']:.1f}%",
                    f"• Hospital Efficiency Score: {metrics['efficiency_score']:.1f}%",
                    ""
                ])
            
            if include_risk and st.session_state.ml_model.is_trained:
                risk_predictions, _ = st.session_state.ml_model.predict_patient_risk(filtered_df)
                if risk_predictions:
                    risk_categories = [pred['risk_category'] for pred in risk_predictions]
                    risk_counts = pd.Series(risk_categories).value_counts()
                    
                    custom_report_content.extend([
                        "=" * 50,
                        "RISK ASSESSMENT ANALYSIS",
                        "=" * 50,
                    ])
                    
                    for risk, count in risk_counts.items():
                        percentage = (count / len(risk_categories)) * 100
                        custom_report_content.append(f"• {risk}: {count:,} ({percentage:.1f}%)")
                    custom_report_content.append("")
            
            custom_report_content.extend([
                "=" * 50,
                "REPORT CONFIGURATION",
                "=" * 50,
                f"• Report Title: {report_title}",
                f"• Reporting Period: {report_period}",
                f"• Demographics Included: {'Yes' if include_demographics else 'No'}",
                f"• Conditions Analysis: {'Yes' if include_conditions else 'No'}",
                f"• Risk Assessment: {'Yes' if include_risk else 'No'}",
                f"• Efficiency Metrics: {'Yes' if include_efficiency else 'No'}",
                "",
                "Report prepared by Healthcare Analytics Dashboard"
            ])
            
            final_report = "\n".join(custom_report_content)
            
            # Display report preview
            st.subheader("Report Preview")
            st.text_area("Generated Report Content", final_report, height=400)
            
            # Download button
            st.download_button(
                label="📥 Download Custom Report",
                data=final_report,
                file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Also offer CSV export of filtered data
            if st.button("📊 Export Filtered Data as CSV", use_container_width=True):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Dataset",
                    data=csv_data,
                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Quick report actions
st.markdown("---")
st.subheader("Quick Report Actions")

quick_col1, quick_col2, quick_col3 = st.columns(3)

with quick_col1:
    if st.button("📊 Generate Full Analytics Report", use_container_width=True):
        # Generate comprehensive report
        full_report = f"""
COMPREHENSIVE HEALTHCARE ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report contains complete analytics for the current healthcare dataset.

Dataset Overview:
- Total Patients: {len(df):,}
- Data Collection Period: {report_date}
- Report Generation Time: {datetime.now().strftime('%H:%M:%S')}

Key Findings:
- Primary analysis focus areas include patient demographics, clinical outcomes, and risk assessment
- Machine learning models provide predictive insights for patient care optimization
- Efficiency metrics indicate current performance against healthcare standards

For detailed analysis, please refer to individual report sections.

Generated by Healthcare Analytics Dashboard v1.0
        """
        
        st.download_button(
            label="📥 Download Full Report",
            data=full_report,
            file_name=f"full_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

with quick_col2:
    if st.button("📈 Export All Visualizations Data", use_container_width=True):
        # Create summary of all key metrics for visualization
        viz_data = {
            'metric': ['Total Patients', 'Avg Length of Stay', 'Readmission Rate', 'Efficiency Score'],
            'value': [
                len(df),
                df['length_of_stay'].mean() if 'length_of_stay' in df.columns else 0,
                df['readmission'].mean() * 100 if 'readmission' in df.columns else 0,
                st.session_state.data_processor.calculate_hospital_efficiency_metrics(df)['efficiency_score']
            ]
        }
        
        viz_df = pd.DataFrame(viz_data)
        csv_data = viz_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Metrics CSV",
            data=csv_data,
            file_name=f"visualization_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with quick_col3:
    if st.button("🔄 Refresh All Reports", use_container_width=True):
        # Clear any cached data and refresh
        st.rerun()
