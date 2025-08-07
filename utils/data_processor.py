import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

class DataProcessor:
    def __init__(self):
        self.processed_data = None
        self.data_cache = {}
    
    def generate_sample_patient_data(self, n_patients=1000):
        """Generate sample patient data for demonstration"""
        np.random.seed(42)
        
        # Patient demographics
        ages = np.random.normal(45, 20, n_patients).astype(int)
        ages = np.clip(ages, 18, 90)
        
        genders = np.random.choice(['Male', 'Female'], n_patients)
        
        # Medical conditions
        conditions = np.random.choice([
            'Hypertension', 'Diabetes', 'Heart Disease', 'Asthma', 
            'Arthritis', 'None'
        ], n_patients, p=[0.25, 0.15, 0.1, 0.1, 0.1, 0.3])
        
        # Hospital metrics
        length_of_stay = np.random.exponential(3, n_patients)
        length_of_stay = np.clip(length_of_stay, 1, 30).astype(int)
        
        # Risk factors
        bmi = np.random.normal(26, 5, n_patients)
        bmi = np.clip(bmi, 15, 50)
        
        blood_pressure_systolic = np.random.normal(130, 20, n_patients).astype(int)
        blood_pressure_systolic = np.clip(blood_pressure_systolic, 90, 200)
        
        # Outcomes
        readmission = np.random.choice([0, 1], n_patients, p=[0.9, 0.1])
        
        # Create DataFrame
        data = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'age': ages,
            'gender': genders,
            'primary_condition': conditions,
            'length_of_stay': length_of_stay,
            'bmi': bmi.round(1),
            'systolic_bp': blood_pressure_systolic,
            'readmission': readmission,
            'admission_date': [
                datetime.now() - timedelta(days=np.random.randint(0, 365))
                for _ in range(n_patients)
            ]
        })
        
        return data
    
    def validate_healthcare_data(self, df):
        """Validate uploaded healthcare data"""
        required_columns = ['patient_id', 'age', 'gender']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['age']):
            return False, "Age column must be numeric"
        
        return True, "Data validation successful"
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded CSV or Excel file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                return None, "Unsupported file format. Please upload CSV or Excel files."
            
            # Validate data
            is_valid, message = self.validate_healthcare_data(df)
            if not is_valid:
                return None, message
            
            # Clean and process data
            df = self.clean_data(df)
            
            return df, "File processed successfully"
        
        except Exception as e:
            return None, f"Error processing file: {str(e)}"
    
    def clean_data(self, df):
        """Clean and preprocess healthcare data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median())
        
        if 'length_of_stay' in df.columns:
            df['length_of_stay'] = df['length_of_stay'].fillna(df['length_of_stay'].median())
        
        # Standardize categorical variables
        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.title()
        
        return df
    
    def calculate_hospital_efficiency_metrics(self, df):
        """Calculate key hospital efficiency metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_patients'] = len(df)
        metrics['avg_length_of_stay'] = df['length_of_stay'].mean() if 'length_of_stay' in df.columns else 0
        metrics['readmission_rate'] = df['readmission'].mean() * 100 if 'readmission' in df.columns else 0
        
        # Efficiency score (simplified calculation)
        if 'length_of_stay' in df.columns and 'readmission' in df.columns:
            # Lower LOS and readmission rate = higher efficiency
            avg_los_normalized = 1 - (metrics['avg_length_of_stay'] / 30)  # Normalize to 30 days max
            readmission_normalized = 1 - (metrics['readmission_rate'] / 100)
            metrics['efficiency_score'] = ((avg_los_normalized + readmission_normalized) / 2) * 100
        else:
            metrics['efficiency_score'] = 85.0  # Default value
        
        return metrics
    
    def get_age_distribution(self, df):
        """Get age distribution for visualization"""
        if 'age' in df.columns:
            age_bins = [0, 18, 30, 45, 60, 75, 100]
            age_labels = ['<18', '18-29', '30-44', '45-59', '60-74', '75+']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
            return df['age_group'].value_counts().sort_index()
        return pd.Series()
    
    def get_condition_distribution(self, df):
        """Get primary condition distribution"""
        if 'primary_condition' in df.columns:
            return df['primary_condition'].value_counts()
        return pd.Series()
    
    def export_to_csv(self, df, filename="healthcare_report.csv"):
        """Export DataFrame to CSV"""
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
