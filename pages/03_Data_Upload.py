import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import RiskAssessmentModel
from utils.visualizations import HealthcareVisualizations

st.set_page_config(
    page_title="Data Upload",
    page_icon="📤",
    layout="wide"
)

# Initialize components
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = RiskAssessmentModel()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HealthcareVisualizations()

st.title("📤 Data Upload & Management")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Upload New Data", "Current Dataset", "Data Quality"])

with tab1:
    st.header("Upload Healthcare Dataset")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a healthcare dataset containing patient information"
        )
        
        if uploaded_file is not None:
            # Process the uploaded file
            with st.spinner("Processing uploaded file..."):
                processed_data, message = st.session_state.data_processor.process_uploaded_file(uploaded_file)
            
            if processed_data is not None:
                st.success(f"✅ {message}")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(processed_data.head(10), use_container_width=True)
                
                # Data summary
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Records", len(processed_data))
                with col_b:
                    st.metric("Total Columns", len(processed_data.columns))
                with col_c:
                    missing_percentage = (processed_data.isnull().sum().sum() / (len(processed_data) * len(processed_data.columns))) * 100
                    st.metric("Missing Data", f"{missing_percentage:.1f}%")
                
                # Option to use this data
                st.subheader("Use This Dataset")
                
                col_use1, col_use2 = st.columns(2)
                
                with col_use1:
                    if st.button("🔄 Replace Current Dataset", use_container_width=True):
                        st.session_state.healthcare_data = processed_data
                        # Reset ML model to retrain with new data
                        st.session_state.ml_model = RiskAssessmentModel()
                        st.success("Dataset replaced successfully! The ML model will be retrained with the new data.")
                        st.rerun()
                
                with col_use2:
                    if st.button("➕ Append to Current Dataset", use_container_width=True):
                        if 'healthcare_data' in st.session_state:
                            # Combine datasets
                            try:
                                combined_data = pd.concat([st.session_state.healthcare_data, processed_data], ignore_index=True)
                                st.session_state.healthcare_data = combined_data
                                # Reset ML model to retrain with combined data
                                st.session_state.ml_model = RiskAssessmentModel()
                                st.success("Data appended successfully! The ML model will be retrained with the combined dataset.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error combining datasets: {str(e)}")
                        else:
                            st.session_state.healthcare_data = processed_data
                            st.success("Dataset loaded successfully!")
                            st.rerun()
            
            else:
                st.error(f"❌ {message}")
    
    with col2:
        st.subheader("Required Columns")
        st.info("""
        **Mandatory Fields:**
        - patient_id
        - age
        - gender
        
        **Optional Fields:**
        - primary_condition
        - length_of_stay
        - bmi
        - systolic_bp
        - readmission
        - admission_date
        """)
        
        st.subheader("Data Format Tips")
        st.info("""
        - Age should be numeric
        - Gender: Male/Female
        - BMI: Numeric (15-50)
        - Blood Pressure: Numeric
        - Readmission: 0 or 1
        - Dates: YYYY-MM-DD format
        """)
    
    # Sample data generation
    st.markdown("---")
    st.header("Generate Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Sample Dataset")
        sample_size = st.number_input("Number of patients", min_value=100, max_value=5000, value=1000, step=100)
        
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample healthcare data..."):
                sample_data = st.session_state.data_processor.generate_sample_patient_data(sample_size)
                st.session_state.healthcare_data = sample_data
                # Reset ML model
                st.session_state.ml_model = RiskAssessmentModel()
                st.success(f"Generated {sample_size} sample patient records!")
                st.rerun()
    
    with col2:
        st.subheader("Download Sample Template")
        st.write("Download a CSV template with the correct column structure:")
        
        # Create sample template
        template_data = {
            'patient_id': [1, 2, 3],
            'age': [45, 62, 33],
            'gender': ['Male', 'Female', 'Male'],
            'primary_condition': ['Hypertension', 'Diabetes', 'None'],
            'length_of_stay': [3, 5, 2],
            'bmi': [25.5, 30.2, 22.1],
            'systolic_bp': [140, 155, 120],
            'readmission': [0, 1, 0],
            'admission_date': ['2024-01-15', '2024-01-20', '2024-01-25']
        }
        
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Template",
            data=csv_template,
            file_name="healthcare_data_template.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab2:
    st.header("Current Dataset Overview")
    
    if 'healthcare_data' in st.session_state:
        df = st.session_state.healthcare_data
        
        # Dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df))
        
        with col2:
            st.metric("Total Columns", len(df.columns))
        
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        with col4:
            duplicate_count = df.duplicated().sum()
            st.metric("Duplicate Records", duplicate_count)
        
        # Column information
        st.subheader("Column Information")
        
        column_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            column_info.append({
                'Column': col,
                'Data Type': col_type,
                'Null Count': null_count,
                'Null %': f"{null_percentage:.1f}%",
                'Unique Values': unique_count
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)
        
        # Data preview
        st.subheader("Data Preview")
        
        preview_options = st.columns([1, 1, 1, 1])
        with preview_options[0]:
            show_rows = st.selectbox("Rows to show", [10, 25, 50, 100], index=0)
        with preview_options[1]:
            if st.button("🔀 Shuffle Data"):
                df_display = df.sample(n=min(show_rows, len(df)))
            else:
                df_display = df.head(show_rows)
        
        st.dataframe(df_display, use_container_width=True)
        
        # Basic statistics
        st.subheader("Statistical Summary")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            st.write("**Numeric Columns:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        if categorical_cols:
            st.write("**Categorical Columns:**")
            cat_summary = []
            for col in categorical_cols:
                top_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                top_count = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': top_value,
                    'Frequency': top_count
                })
            
            cat_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_df, use_container_width=True)
        
        # Export current dataset
        st.subheader("Export Current Dataset")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📥 Export as CSV", use_container_width=True):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"healthcare_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("📊 Export Summary Report", use_container_width=True):
                report = f"""
Healthcare Dataset Summary Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Overview:
- Total Records: {len(df):,}
- Total Columns: {len(df.columns)}
- Memory Usage: {memory_usage:.2f} MB
- Duplicate Records: {duplicate_count}

Column Details:
{column_df.to_string(index=False)}

Data Quality:
- Missing Data Percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%
- Complete Records: {len(df.dropna()):,}

Numeric Summary:
{df[numeric_cols].describe().to_string() if numeric_cols else 'No numeric columns'}
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"dataset_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        st.warning("No dataset loaded. Please upload data or generate sample data.")

with tab3:
    st.header("Data Quality Assessment")
    
    if 'healthcare_data' in st.session_state:
        df = st.session_state.healthcare_data
        
        # Data quality metrics
        st.subheader("Quality Metrics")
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with quality_col2:
            duplicate_rate = (df.duplicated().sum() / len(df)) * 100
            st.metric("Duplicate Rate", f"{duplicate_rate:.1f}%")
        
        with quality_col3:
            # Simple validity check for age column
            valid_age_rate = 100
            if 'age' in df.columns:
                valid_ages = ((df['age'] >= 0) & (df['age'] <= 120)).sum()
                valid_age_rate = (valid_ages / len(df)) * 100
            st.metric("Valid Age Values", f"{valid_age_rate:.1f}%")
        
        # Missing data analysis
        st.subheader("Missing Data Analysis")
        
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percentage.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualize missing data
            if len(missing_df) > 0:
                import plotly.express as px
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing Percentage',
                    title="Missing Data by Column",
                    labels={'Missing Percentage': 'Missing Data (%)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data found in the dataset!")
        
        # Data validation
        st.subheader("Data Validation")
        
        validation_results = []
        
        # Age validation
        if 'age' in df.columns:
            invalid_ages = ((df['age'] < 0) | (df['age'] > 120)).sum()
            validation_results.append({
                'Check': 'Age Range (0-120)',
                'Status': '✅ Pass' if invalid_ages == 0 else f'❌ Fail ({invalid_ages} invalid)',
                'Description': 'Age values should be between 0 and 120'
            })
        
        # BMI validation
        if 'bmi' in df.columns:
            invalid_bmi = ((df['bmi'] < 10) | (df['bmi'] > 60)).sum()
            validation_results.append({
                'Check': 'BMI Range (10-60)',
                'Status': '✅ Pass' if invalid_bmi == 0 else f'❌ Fail ({invalid_bmi} invalid)',
                'Description': 'BMI values should be between 10 and 60'
            })
        
        # Blood pressure validation
        if 'systolic_bp' in df.columns:
            invalid_bp = ((df['systolic_bp'] < 60) | (df['systolic_bp'] > 250)).sum()
            validation_results.append({
                'Check': 'Blood Pressure Range (60-250)',
                'Status': '✅ Pass' if invalid_bp == 0 else f'❌ Fail ({invalid_bp} invalid)',
                'Description': 'Systolic BP should be between 60 and 250 mmHg'
            })
        
        # Gender validation
        if 'gender' in df.columns:
            valid_genders = df['gender'].isin(['Male', 'Female', 'M', 'F']).sum()
            invalid_gender = len(df) - valid_genders
            validation_results.append({
                'Check': 'Gender Values',
                'Status': '✅ Pass' if invalid_gender == 0 else f'❌ Fail ({invalid_gender} invalid)',
                'Description': 'Gender should be Male/Female or M/F'
            })
        
        # Length of stay validation
        if 'length_of_stay' in df.columns:
            invalid_los = ((df['length_of_stay'] < 0) | (df['length_of_stay'] > 365)).sum()
            validation_results.append({
                'Check': 'Length of Stay (0-365)',
                'Status': '✅ Pass' if invalid_los == 0 else f'❌ Fail ({invalid_los} invalid)',
                'Description': 'Length of stay should be between 0 and 365 days'
            })
        
        if validation_results:
            validation_df = pd.DataFrame(validation_results)
            st.dataframe(validation_df, use_container_width=True)
        
        # Data cleaning options
        st.subheader("Data Cleaning Options")
        
        cleaning_col1, cleaning_col2 = st.columns(2)
        
        with cleaning_col1:
            if st.button("🧹 Remove Duplicate Records", use_container_width=True):
                original_count = len(df)
                df_cleaned = df.drop_duplicates()
                st.session_state.healthcare_data = df_cleaned
                removed_count = original_count - len(df_cleaned)
                st.success(f"Removed {removed_count} duplicate records.")
                if removed_count > 0:
                    st.rerun()
        
        with cleaning_col2:
            if st.button("🔧 Fill Missing Values", use_container_width=True):
                df_filled = st.session_state.data_processor.clean_data(df)
                st.session_state.healthcare_data = df_filled
                st.success("Missing values filled with appropriate defaults.")
                st.rerun()
        
        # Overall quality score
        st.subheader("Overall Data Quality Score")
        
        # Calculate quality score
        quality_factors = [
            completeness / 100,  # Completeness weight: 40%
            (100 - duplicate_rate) / 100,  # Uniqueness weight: 20%
            valid_age_rate / 100,  # Validity weight: 40%
        ]
        
        weights = [0.4, 0.2, 0.4]
        overall_quality = sum(factor * weight for factor, weight in zip(quality_factors, weights)) * 100
        
        quality_color = "🟢" if overall_quality >= 90 else "🟡" if overall_quality >= 70 else "🔴"
        
        st.metric(
            label="Data Quality Score",
            value=f"{quality_color} {overall_quality:.1f}/100"
        )
        
        if overall_quality >= 90:
            st.success("Excellent data quality! Your dataset is ready for analysis.")
        elif overall_quality >= 70:
            st.warning("Good data quality with room for improvement. Consider addressing missing values or duplicates.")
        else:
            st.error("Poor data quality detected. Please clean the data before proceeding with analysis.")
    
    else:
        st.warning("No dataset loaded. Please upload data or generate sample data to assess quality.")
