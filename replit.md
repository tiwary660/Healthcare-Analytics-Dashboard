# Healthcare Analytics Dashboard

## Overview

This is an advanced Streamlit-based healthcare analytics dashboard that provides comprehensive patient data analysis, risk assessment, and hospital efficiency monitoring with enhanced interactive visualizations. The application features multi-page navigation with dedicated sections for dashboard analytics, machine learning-powered risk assessment, data upload capabilities, automated report generation, and a new advanced analytics section with interactive presentation slides. It serves healthcare administrators and clinical staff by providing insights into patient outcomes, hospital performance metrics, predictive analytics for readmission risks, and immersive 3D visualizations for complex data analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page structure and enhanced CSS styling
- **Layout Strategy**: Wide layout configuration with expandable sidebar navigation and custom gradient themes
- **Page Organization**: Modular page structure with five main sections (Dashboard, Risk Assessment, Data Upload, Reports, Advanced Analytics)
- **State Management**: Streamlit session state for persistent data and model storage across pages
- **UI Components**: Column-based layouts, tabs, metrics displays, interactive file uploaders, and animated status indicators
- **Presentation Mode**: Interactive slide-based presentation system with navigation controls and auto-advance functionality
- **Enhanced Styling**: Custom CSS with gradient backgrounds, animated elements, and responsive design patterns

### Backend Architecture
- **Data Processing**: Centralized DataProcessor class handling sample data generation, file uploads, and metrics calculation
- **Machine Learning**: Dedicated RiskAssessmentModel class implementing scikit-learn models for risk classification and length-of-stay prediction
- **Visualization Engine**: HealthcareVisualizations class using Plotly for interactive charts and graphs
- **Model Pipeline**: Automated feature engineering with categorical encoding, risk factor creation, and standardization

### Data Processing Strategy
- **Sample Data Generation**: Built-in capability to generate realistic healthcare datasets with patient demographics, medical conditions, and outcomes
- **File Upload Support**: CSV and Excel file processing with data validation and quality checks
- **Feature Engineering**: Automatic creation of risk factors, categorical encoding, and derived metrics
- **Caching Mechanism**: In-memory data caching to improve performance across page navigation

### Machine Learning Components
- **Risk Assessment**: Random Forest classifier for predicting patient readmission risk
- **Length of Stay Prediction**: Gradient Boosting regressor for hospital stay duration forecasting
- **Model Training**: Automated training pipeline with train-test split and performance evaluation
- **Feature Importance**: Built-in analysis of model feature contributions for interpretability

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Machine Learning Stack
- **Scikit-learn**: Machine learning algorithms (RandomForest, GradientBoosting)
- **Scikit-learn Preprocessing**: StandardScaler, LabelEncoder for data preparation
- **Joblib**: Model serialization and persistence (imported but not actively used)

### Visualization
- **Plotly Express**: High-level plotting interface for interactive visualizations
- **Plotly Graph Objects**: Low-level plotting for custom chart configurations
- **Advanced Visualizations**: 3D scatter plots, Sankey diagrams, animated timelines, gauge dashboards
- **Interactive Elements**: Correlation matrices, treemaps, violin plots, waterfall charts, and radar charts
- **Presentation Components**: Slide-based navigation system with CSS animations and transitions

### Data Processing
- **IO Module**: File handling for uploads and data processing
- **Datetime**: Date and time manipulation for report generation and timestamps

### Potential Database Integration
The architecture is designed to accommodate database integration, though currently uses in-memory storage. The DataProcessor class structure suggests readiness for database connectivity upgrades.