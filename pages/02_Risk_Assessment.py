import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import RiskAssessmentModel
from utils.visualizations import HealthcareVisualizations

st.set_page_config(
    page_title="Risk Assessment",
    page_icon="🎯",
    layout="wide"
)

# Initialize components
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = RiskAssessmentModel()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()

st.title("🎯 Patient Risk Assessment")
st.markdown("---")

# Load data if not available
if 'healthcare_data' not in st.session_state:
    st.session_state.healthcare_data = st.session_state.data_processor.generate_sample_patient_data(1000)

df = st.session_state.healthcare_data

# Train model if not trained
if not st.session_state.ml_model.is_trained:
    with st.spinner("Training machine learning models..."):
        success, message = st.session_state.ml_model.train_models(df)
        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Individual Assessment", "Batch Analysis", "Model Performance", "Feature Importance"])

with tab1:
    st.header("Individual Patient Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        
        # Patient input form
        with st.form("patient_risk_form"):
            patient_age = st.number_input("Age", min_value=18, max_value=100, value=45)
            patient_gender = st.selectbox("Gender", ["Male", "Female"])
            patient_bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            patient_bp = st.number_input("Systolic Blood Pressure", min_value=90, max_value=200, value=120)
            patient_condition = st.selectbox(
                "Primary Medical Condition",
                ["None", "Hypertension", "Diabetes", "Heart Disease", "Asthma", "Arthritis"]
            )
            patient_los = st.number_input("Expected Length of Stay (days)", min_value=1, max_value=30, value=3)
            
            assess_button = st.form_submit_button("🔍 Assess Risk", use_container_width=True)
    
    with col2:
        st.subheader("Risk Assessment Results")
        
        if assess_button:
            # Prepare patient data
            patient_data = {
                'age': patient_age,
                'gender': patient_gender,
                'bmi': patient_bmi,
                'systolic_bp': patient_bp,
                'primary_condition': patient_condition,
                'length_of_stay': patient_los,
                'readmission': 0  # Default value
            }
            
            # Make prediction
            prediction, message = st.session_state.ml_model.predict_patient_risk(patient_data)
            
            if prediction:
                # Display risk category
                risk_category = prediction['risk_category']
                risk_score = prediction['risk_score']
                
                # Color code based on risk
                if risk_category == "Low Risk":
                    risk_color = "🟢"
                elif risk_category == "Medium Risk":
                    risk_color = "🟡"
                else:
                    risk_color = "🔴"
                
                st.success(f"**Assessment Complete!**")
                
                # Risk metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        label="Risk Category",
                        value=f"{risk_color} {risk_category}"
                    )
                with col_b:
                    st.metric(
                        label="Confidence Score",
                        value=f"{risk_score:.1f}%"
                    )
                
                # Probability breakdown
                st.subheader("Risk Probability Breakdown")
                probs = prediction['probabilities']
                
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                with prob_col1:
                    st.metric("Low Risk", f"{probs['low']:.1f}%")
                with prob_col2:
                    st.metric("Medium Risk", f"{probs['medium']:.1f}%")
                with prob_col3:
                    st.metric("High Risk", f"{probs['high']:.1f}%")
                
                # Predicted length of stay
                if 'predicted_los' in prediction:
                    st.metric(
                        label="Predicted Length of Stay",
                        value=f"{prediction['predicted_los']} days"
                    )
                
                # Recommendations
                st.subheader("Clinical Recommendations")
                if risk_category == "High Risk":
                    st.warning("""
                    **High Risk Patient - Immediate Attention Required**
                    - Schedule frequent monitoring
                    - Consider preventive interventions
                    - Ensure care coordination
                    - Review discharge planning early
                    """)
                elif risk_category == "Medium Risk":
                    st.info("""
                    **Medium Risk Patient - Monitor Closely**
                    - Regular check-ins recommended
                    - Monitor for changes in condition
                    - Consider additional assessments
                    """)
                else:
                    st.success("""
                    **Low Risk Patient - Standard Care**
                    - Continue with standard care protocols
                    - Routine monitoring sufficient
                    """)
            else:
                st.error(f"❌ {message}")

with tab2:
    st.header("Batch Risk Analysis")
    
    # Analyze all patients in the dataset
    with st.spinner("Analyzing all patients..."):
        all_predictions, message = st.session_state.ml_model.predict_patient_risk(df)
    
    if all_predictions:
        # Create risk distribution chart
        risk_chart = st.session_state.visualizer.create_risk_assessment_chart(all_predictions)
        if risk_chart:
            st.plotly_chart(risk_chart, use_container_width=True)
        
        # Risk statistics
        col1, col2, col3 = st.columns(3)
        
        risk_categories = [pred['risk_category'] for pred in all_predictions]
        
        with col1:
            low_risk_count = risk_categories.count('Low Risk')
            st.metric("Low Risk Patients", low_risk_count, f"{(low_risk_count/len(risk_categories)*100):.1f}%")
        
        with col2:
            medium_risk_count = risk_categories.count('Medium Risk')
            st.metric("Medium Risk Patients", medium_risk_count, f"{(medium_risk_count/len(risk_categories)*100):.1f}%")
        
        with col3:
            high_risk_count = risk_categories.count('High Risk')
            st.metric("High Risk Patients", high_risk_count, f"{(high_risk_count/len(risk_categories)*100):.1f}%")
        
        # Risk score distribution
        st.subheader("Risk Score Distribution")
        risk_score_chart = st.session_state.visualizer.create_risk_score_distribution(all_predictions)
        if risk_score_chart:
            st.plotly_chart(risk_score_chart, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Risk Assessment Results")
        
        # Create results dataframe
        results_data = []
        for i, pred in enumerate(all_predictions):
            results_data.append({
                'Patient ID': df.iloc[i]['patient_id'],
                'Age': df.iloc[i]['age'],
                'Gender': df.iloc[i]['gender'],
                'Risk Category': pred['risk_category'],
                'Risk Score': f"{pred['risk_score']:.1f}%",
                'Predicted LOS': pred.get('predicted_los', 'N/A')
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            risk_filter = st.selectbox("Filter by Risk Category", ["All", "Low Risk", "Medium Risk", "High Risk"])
        with col2:
            show_count = st.selectbox("Show Records", [50, 100, 200, "All"])
        
        # Apply filters
        filtered_df = results_df.copy()
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['Risk Category'] == risk_filter]
        
        if show_count != "All":
            filtered_df = filtered_df.head(show_count)
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export results
        if st.button("📥 Export Risk Assessment Results"):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv_data,
                file_name=f"risk_assessment_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.error("Unable to perform batch analysis")

with tab3:
    st.header("Model Performance Metrics")
    
    # Get model performance
    performance_metrics = st.session_state.ml_model.get_model_performance_metrics(df)
    
    if performance_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Model Accuracy",
                value=f"{performance_metrics['accuracy']:.3f}",
                delta="Target: >0.80"
            )
        
        with col2:
            st.metric(
                label="Total Predictions",
                value=f"{performance_metrics['total_predictions']:,}"
            )
        
        with col3:
            risk_dist = performance_metrics['risk_distribution']
            most_common_risk = max(risk_dist, key=risk_dist.get)
            st.metric(
                label="Most Common Risk",
                value=most_common_risk,
                delta=f"{risk_dist[most_common_risk]} patients"
            )
        
        # Model training information
        st.subheader("Model Information")
        
        model_info_col1, model_info_col2 = st.columns(2)
        
        with model_info_col1:
            st.info("""
            **Risk Classification Model**
            - Algorithm: Random Forest Classifier
            - Features: Age, BMI, Blood Pressure, Gender, Medical Condition
            - Classes: Low, Medium, High Risk
            - Training Method: Stratified sampling
            """)
        
        with model_info_col2:
            st.info("""
            **Length of Stay Prediction**
            - Algorithm: Gradient Boosting Regressor
            - Target: Hospital length of stay (days)
            - Features: Same as risk model
            - Validation: Cross-validation
            """)
        
        # Performance recommendations
        st.subheader("Model Performance Assessment")
        
        accuracy = performance_metrics['accuracy']
        if accuracy >= 0.85:
            st.success("🎯 **Excellent Performance**: Model accuracy is high and reliable for clinical decision support.")
        elif accuracy >= 0.75:
            st.warning("⚠️ **Good Performance**: Model is performing well but could benefit from additional training data.")
        else:
            st.error("❌ **Needs Improvement**: Model accuracy is below clinical standards. Consider retraining with more data.")
    
    else:
        st.error("Unable to calculate model performance metrics")

with tab4:
    st.header("Feature Importance Analysis")
    
    # Get feature importance
    importance_df = st.session_state.ml_model.get_feature_importance()
    
    if importance_df is not None and not importance_df.empty:
        # Create feature importance chart
        importance_chart = st.session_state.visualizer.create_feature_importance_chart(importance_df)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)
        
        # Feature importance table
        st.subheader("Feature Importance Scores")
        
        # Format the importance scores
        importance_display = importance_df.copy()
        importance_display['importance'] = importance_display['importance'].apply(lambda x: f"{x:.4f}")
        importance_display['importance_percentage'] = (importance_df['importance'] * 100).apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(
            importance_display.rename(columns={
                'feature': 'Feature',
                'importance': 'Importance Score',
                'importance_percentage': 'Percentage'
            }),
            use_container_width=True
        )
        
        # Feature interpretation
        st.subheader("Clinical Interpretation")
        
        top_feature = importance_df.iloc[0]['feature']
        top_importance = importance_df.iloc[0]['importance']
        
        st.info(f"""
        **Most Important Feature**: {top_feature} (Importance: {top_importance:.4f})
        
        This analysis shows which patient characteristics have the strongest influence on risk assessment predictions. 
        Features with higher importance scores should be given more attention during clinical evaluations.
        """)
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        
        feature_descriptions = {
            'Age': 'Patient age in years - older patients typically have higher risk',
            'BMI': 'Body Mass Index - higher BMI associated with increased health risks',
            'Systolic BP': 'Systolic blood pressure - elevated levels indicate cardiovascular risk',
            'Gender': 'Patient gender - may influence certain health conditions',
            'Primary Condition': 'Main medical condition - certain conditions carry higher risks',
            'Age-BMI Risk': 'Combined risk factor for patients over 65 with BMI > 30',
            'Hypertension Risk': 'Risk indicator for patients with systolic BP > 140'
        }
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            if feature in feature_descriptions:
                st.write(f"**{feature}**: {feature_descriptions[feature]}")
    
    else:
        st.error("Unable to generate feature importance analysis. Please ensure the model is properly trained.")

# Retrain model button
st.markdown("---")
if st.button("🔄 Retrain Risk Assessment Model", use_container_width=True):
    with st.spinner("Retraining machine learning models..."):
        # Reset model
        st.session_state.ml_model = RiskAssessmentModel()
        success, message = st.session_state.ml_model.train_models(df)
        
        if success:
            st.success(f"✅ Model retrained successfully! {message}")
        else:
            st.error(f"❌ Failed to retrain model: {message}")
        
        st.rerun()
