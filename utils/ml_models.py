import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import joblib
from datetime import datetime

class RiskAssessmentModel:
    def __init__(self):
        self.risk_classifier = None
        self.los_predictor = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for machine learning models"""
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # Handle missing values
        numeric_columns = ['age', 'bmi', 'systolic_bp', 'length_of_stay']
        for col in numeric_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(features_df[col].median())
        
        # Encode categorical variables
        categorical_columns = ['gender', 'primary_condition']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    # Handle unseen categories
                    try:
                        features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col].astype(str))
                    except ValueError:
                        # Assign default value for unseen categories
                        features_df[f'{col}_encoded'] = 0
        
        # Create risk factors
        if 'age' in features_df.columns and 'bmi' in features_df.columns:
            features_df['age_bmi_risk'] = (features_df['age'] > 65) & (features_df['bmi'] > 30)
        
        if 'systolic_bp' in features_df.columns:
            features_df['hypertension_risk'] = features_df['systolic_bp'] > 140
        
        # Select feature columns
        feature_columns = []
        
        # Add numeric features
        for col in ['age', 'bmi', 'systolic_bp']:
            if col in features_df.columns:
                feature_columns.append(col)
        
        # Add encoded categorical features
        for col in ['gender_encoded', 'primary_condition_encoded']:
            if col in features_df.columns:
                feature_columns.append(col)
        
        # Add derived features
        for col in ['age_bmi_risk', 'hypertension_risk']:
            if col in features_df.columns:
                feature_columns.append(col)
        
        return features_df[feature_columns].fillna(0)
    
    def create_risk_labels(self, df):
        """Create risk labels based on multiple factors"""
        risk_scores = np.zeros(len(df))
        
        # Age factor
        if 'age' in df.columns:
            risk_scores += (df['age'] > 65).astype(int) * 2
            risk_scores += (df['age'] > 75).astype(int) * 1
        
        # BMI factor
        if 'bmi' in df.columns:
            risk_scores += (df['bmi'] > 30).astype(int) * 1
            risk_scores += (df['bmi'] > 35).astype(int) * 1
        
        # Blood pressure factor
        if 'systolic_bp' in df.columns:
            risk_scores += (df['systolic_bp'] > 140).astype(int) * 1
            risk_scores += (df['systolic_bp'] > 160).astype(int) * 1
        
        # Condition factor
        if 'primary_condition' in df.columns:
            high_risk_conditions = ['Heart Disease', 'Diabetes', 'Hypertension']
            risk_scores += df['primary_condition'].isin(high_risk_conditions).astype(int) * 2
        
        # Length of stay factor
        if 'length_of_stay' in df.columns:
            risk_scores += (df['length_of_stay'] > 7).astype(int) * 1
        
        # Readmission factor
        if 'readmission' in df.columns:
            risk_scores += df['readmission'].astype(int) * 3
        
        # Convert scores to risk categories
        risk_labels = np.where(risk_scores <= 2, 0,  # Low risk
                              np.where(risk_scores <= 5, 1,  # Medium risk
                                      2))  # High risk
        
        return risk_labels
    
    def train_models(self, df):
        """Train risk assessment and length of stay prediction models"""
        try:
            # Prepare features
            X = self.prepare_features(df)
            
            if X.empty or len(X.columns) == 0:
                return False, "No valid features found for training"
            
            # Train risk classification model
            y_risk = self.create_risk_labels(df)
            
            if len(np.unique(y_risk)) > 1:  # Ensure we have multiple classes
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
                )
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train risk classifier
                self.risk_classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
                self.risk_classifier.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = self.risk_classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Train length of stay predictor if data is available
                if 'length_of_stay' in df.columns:
                    y_los = df['length_of_stay']
                    X_train_los, X_test_los, y_train_los, y_test_los = train_test_split(
                        X, y_los, test_size=0.2, random_state=42
                    )
                    
                    X_train_los_scaled = self.scaler.transform(X_train_los)
                    X_test_los_scaled = self.scaler.transform(X_test_los)
                    
                    self.los_predictor = GradientBoostingRegressor(
                        n_estimators=100, random_state=42, max_depth=6
                    )
                    self.los_predictor.fit(X_train_los_scaled, y_train_los)
                
                self.is_trained = True
                return True, f"Models trained successfully. Risk classifier accuracy: {accuracy:.3f}"
            
            else:
                return False, "Insufficient data variation for training"
                
        except Exception as e:
            return False, f"Error training models: {str(e)}"
    
    def predict_patient_risk(self, patient_data):
        """Predict risk for a single patient or batch of patients"""
        if not self.is_trained or self.risk_classifier is None:
            return None, "Model not trained yet"
        
        try:
            # Prepare features
            if isinstance(patient_data, dict):
                # Single patient
                patient_df = pd.DataFrame([patient_data])
            else:
                # Multiple patients
                patient_df = patient_data.copy()
            
            X = self.prepare_features(patient_df)
            X_scaled = self.scaler.transform(X)
            
            # Predict risk
            risk_probabilities = self.risk_classifier.predict_proba(X_scaled)
            risk_predictions = self.risk_classifier.predict(X_scaled)
            
            # Convert to readable format
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            
            results = []
            for i, (pred, probs) in enumerate(zip(risk_predictions, risk_probabilities)):
                result = {
                    'risk_category': risk_labels[pred],
                    'risk_score': np.max(probs) * 100,
                    'probabilities': {
                        'low': probs[0] * 100,
                        'medium': probs[1] * 100 if len(probs) > 1 else 0,
                        'high': probs[2] * 100 if len(probs) > 2 else 0
                    }
                }
                
                # Predict length of stay if model is available
                if self.los_predictor is not None:
                    los_pred = self.los_predictor.predict(X_scaled[i:i+1])
                    result['predicted_los'] = max(1, round(los_pred[0]))
                
                results.append(result)
            
            return results if len(results) > 1 else results[0], "Prediction successful"
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained or self.risk_classifier is None:
            return None
        
        try:
            feature_names = [
                'Age', 'BMI', 'Systolic BP', 'Gender', 'Primary Condition',
                'Age-BMI Risk', 'Hypertension Risk'
            ]
            
            importances = self.risk_classifier.feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            return None
    
    def get_model_performance_metrics(self, df):
        """Get performance metrics for the trained models"""
        if not self.is_trained:
            return None
        
        try:
            X = self.prepare_features(df)
            y_risk = self.create_risk_labels(df)
            
            X_scaled = self.scaler.transform(X)
            y_pred = self.risk_classifier.predict(X_scaled)
            
            accuracy = accuracy_score(y_risk, y_pred)
            
            # Risk distribution
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            unique, counts = np.unique(y_pred, return_counts=True)
            risk_distribution = {risk_labels[i]: count for i, count in zip(unique, counts)}
            
            metrics = {
                'accuracy': accuracy,
                'total_predictions': len(y_pred),
                'risk_distribution': risk_distribution
            }
            
            return metrics
            
        except Exception as e:
            return None
