import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Air Quality Index Prediction System",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Helper Functions
def get_aqi_category(aqi):
    """Get AQI category and recommendation"""
    if aqi <= 50:
        return "Good", "Air quality is safe. You can freely go outside.", "#00E400"
    elif aqi <= 100:
        return "Satisfactory", "Air quality is acceptable. Slight discomfort for extremely sensitive people.", "#FFFF00"
    elif aqi <= 200:
        return "Moderate", "Sensitive groups may feel discomfort. Consider wearing a mask outside.", "#FF7E00"
    elif aqi <= 300:
        return "Poor", "Avoid prolonged outdoor exposure. People with asthma should be careful.", "#FF0000"
    elif aqi <= 400:
        return "Very Poor", "Avoid outdoor activities. Use N95 masks if necessary.", "#8F3F97"
    else:
        return "Severe", "Stay indoors. Air is hazardous even for healthy people.", "#7E0023"

def load_data(file):
    """Load data from uploaded file"""
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Reset file pointer again after reading
        file.seek(0)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def preprocess_data(df):
    """Preprocess data for training"""
    try:
        # Drop RecordID if exists
        if 'RecordID' in df.columns:
            df_processed = df.drop(columns=['RecordID'])
        else:
            df_processed = df.copy()
        
        # Check if required columns exist
        required_cols = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 
                        'Temperature', 'Humidity', 'WindSpeed', 
                        'RespiratoryCases', 'CardiovascularCases', 
                        'HospitalAdmissions', 'HealthImpactScore']
        
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}. Please ensure your data has these columns.")
            return None, None, None, None, None, None
        
        # Define features and target
        y = df_processed['AQI']
        X = df_processed.drop(columns=['AQI', 'HealthImpactClass'] if 'HealthImpactClass' in df_processed.columns else ['AQI'])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None, None, None

# Main App
st.markdown('<h1 class="main-header">🌬️ Air Quality Index Prediction System</h1>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Data Overview", "📈 Exploratory Data Analysis", "🤖 Model Training", "🔮 Predict AQI", "📉 Model Performance"]
)

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your air quality dataset"
    )
    
    if uploaded_file is not None:
        if st.session_state.df is None or st.sidebar.button("Reload Data"):
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.success("Data loaded successfully!")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Data Info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Data Size", f"{df.size:,}")
            
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)
            
            with col2:
                st.write("**Missing Values:**")
                missing = df.isnull().sum()
                missing_df = missing[missing > 0].to_frame('Missing Count')
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values found!")
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Download processed data
            if st.button("Download Processed Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="air_quality_data.csv",
                    mime="text/csv"
                )
    else:
        st.info("👈 Please upload a data file from the sidebar to get started.")
        st.markdown("""
        ### Expected Data Format:
        Your dataset should contain the following columns:
        - **PM10**: Particulate Matter 10
        - **PM2_5**: Particulate Matter 2.5
        - **NO2**: Nitrogen Dioxide
        - **SO2**: Sulfur Dioxide
        - **O3**: Ozone
        - **Temperature**: Temperature in Celsius
        - **Humidity**: Humidity percentage
        - **WindSpeed**: Wind speed
        - **RespiratoryCases**: Number of respiratory cases
        - **CardiovascularCases**: Number of cardiovascular cases
        - **HospitalAdmissions**: Number of hospital admissions
        - **HealthImpactScore**: Health impact score
        - **AQI**: Air Quality Index (target variable)
        """)

# Page 2: Exploratory Data Analysis
elif page == "📈 Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the 'Data Overview' page first.")
    else:
        df = st.session_state.df
        
        # EDA Options
        st.sidebar.subheader("Visualization Options")
        show_histograms = st.sidebar.checkbox("Show Histograms", value=True)
        show_boxplots = st.sidebar.checkbox("Show Boxplots", value=True)
        show_correlation = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
        show_target_dist = st.sidebar.checkbox("Show Target Distribution", value=True)
        
        # Histograms
        if show_histograms:
            st.subheader("📊 Distribution of Numeric Features")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_cols) > 0:
                cols_per_row = 3
                num_cols = len(numeric_cols)
                rows = (num_cols + cols_per_row - 1) // cols_per_row
                
                for i in range(rows):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i * cols_per_row + j
                        if idx < num_cols:
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.histplot(df[numeric_cols[idx]], kde=True, bins=30, ax=ax)
                                ax.set_title(numeric_cols[idx])
                                st.pyplot(fig)
                                plt.close()
            else:
                st.info("No numeric columns found for histograms.")
        
        # Boxplots
        if show_boxplots:
            st.subheader("📦 Boxplots for Numeric Features")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_cols) > 0:
                cols_per_row = 3
                num_cols = len(numeric_cols)
                rows = (num_cols + cols_per_row - 1) // cols_per_row
                
                for i in range(rows):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i * cols_per_row + j
                        if idx < num_cols:
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.boxplot(x=df[numeric_cols[idx]], color='skyblue', ax=ax)
                                ax.set_title(numeric_cols[idx])
                                st.pyplot(fig)
                                plt.close()
        
        # Correlation Heatmap
        if show_correlation:
            st.subheader("🔥 Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(14, 10))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, square=True)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
                plt.close()
        
        # Target Distribution
        if show_target_dist and 'AQI' in df.columns:
            st.subheader("🎯 Target Variable Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df['AQI'], bins=30, kde=True, ax=ax)
                ax.set_title("AQI Distribution")
                st.pyplot(fig)
                plt.close()
            
            with col2:
                if 'HealthImpactClass' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, x='HealthImpactClass', ax=ax)
                    ax.set_title("Health Impact Class Distribution")
                    st.pyplot(fig)
                    plt.close()
                    
                    st.write("**Health Impact Class Counts:**")
                    st.dataframe(df['HealthImpactClass'].value_counts().to_frame('Count'), use_container_width=True)

# Page 3: Model Training
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the 'Data Overview' page first.")
    else:
        df = st.session_state.df
        
        st.sidebar.subheader("Model Configuration")
        model_choice = st.sidebar.selectbox(
            "Select Model",
            ["Random Forest", "Linear Regression", "Decision Tree", "Compare All Models"]
        )
        
        if st.sidebar.button("🚀 Train Model", type="primary"):
            with st.spinner("Preprocessing data and training model..."):
                X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
                
                if X_train is not None:
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = feature_names
                    
                    # Train models
                    results = {}
                    
                    if model_choice == "Random Forest" or model_choice == "Compare All Models":
                        with st.spinner("Training Random Forest..."):
                            rf = RandomForestRegressor(
                                n_estimators=300,
                                max_depth=10,
                                min_samples_split=5,
                                min_samples_leaf=1,
                                random_state=42
                            )
                            rf.fit(X_train, y_train)
                            y_pred_rf = rf.predict(X_test)
                            
                            results['Random Forest'] = {
                                'model': rf,
                                'mae': mean_absolute_error(y_test, y_pred_rf),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                                'r2': r2_score(y_test, y_pred_rf),
                                'predictions': y_pred_rf
                            }
                            st.session_state.model = rf
                    
                    if model_choice == "Linear Regression" or model_choice == "Compare All Models":
                        with st.spinner("Training Linear Regression..."):
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            lr = LinearRegression()
                            lr.fit(X_train_scaled, y_train)
                            y_pred_lr = lr.predict(X_test_scaled)
                            
                            results['Linear Regression'] = {
                                'model': lr,
                                'mae': mean_absolute_error(y_test, y_pred_lr),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                                'r2': r2_score(y_test, y_pred_lr),
                                'predictions': y_pred_lr
                            }
                            if model_choice == "Linear Regression":
                                st.session_state.model = lr
                    
                    if model_choice == "Decision Tree" or model_choice == "Compare All Models":
                        with st.spinner("Training Decision Tree..."):
                            dt = DecisionTreeRegressor(max_depth=None, random_state=42)
                            dt.fit(X_train, y_train)
                            y_pred_dt = dt.predict(X_test)
                            
                            results['Decision Tree'] = {
                                'model': dt,
                                'mae': mean_absolute_error(y_test, y_pred_dt),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
                                'r2': r2_score(y_test, y_pred_dt),
                                'predictions': y_pred_dt
                            }
                            if model_choice == "Decision Tree":
                                st.session_state.model = dt
                    
                    st.session_state.results = results
                    st.session_state.trained = True
                    st.success("✅ Model training completed!")
        
        # Display Results
        if st.session_state.trained and st.session_state.results is not None:
            results = st.session_state.results
            
            st.subheader("📊 Model Performance Metrics")
            
            if model_choice == "Compare All Models":
                # Comparison Table
                comparison_data = []
                for model_name, metrics in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'MAE': f"{metrics['mae']:.2f}",
                        'RMSE': f"{metrics['rmse']:.2f}",
                        'R² Score': f"{metrics['r2']:.4f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visual Comparison
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                models = list(results.keys())
                mae_values = [results[m]['mae'] for m in models]
                rmse_values = [results[m]['rmse'] for m in models]
                r2_values = [results[m]['r2'] for m in models]
                
                axes[0].bar(models, mae_values, color='skyblue')
                axes[0].set_title('Mean Absolute Error (MAE)')
                axes[0].set_ylabel('MAE')
                axes[0].tick_params(axis='x', rotation=45)
                
                axes[1].bar(models, rmse_values, color='lightcoral')
                axes[1].set_title('Root Mean Squared Error (RMSE)')
                axes[1].set_ylabel('RMSE')
                axes[1].tick_params(axis='x', rotation=45)
                
                axes[2].bar(models, r2_values, color='lightgreen')
                axes[2].set_title('R² Score')
                axes[2].set_ylabel('R² Score')
                axes[2].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                # Single Model Results
                if model_choice in results:
                    metrics = results[model_choice]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f}")
                    with col2:
                        st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.2f}")
                    with col3:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            # Feature Importance (for Random Forest)
            if model_choice == "Random Forest" or (model_choice == "Compare All Models" and 'Random Forest' in results):
                st.subheader("🔍 Feature Importance")
                rf_model = results.get('Random Forest', {}).get('model') or st.session_state.model
                
                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    feature_names = st.session_state.feature_names
                    
                    if feature_names is not None and len(feature_names) == len(importances):
                        feat_imp_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)
                        
                        st.dataframe(feat_imp_df, use_container_width=True)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
                        ax.set_title("Feature Importance (Random Forest)")
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("Feature names not available for visualization.")
            
            # Save Model
            if st.button("💾 Save Model"):
                if st.session_state.model is not None:
                    model_filename = f"aqi_model_{model_choice.lower().replace(' ', '_')}.pkl"
                    joblib.dump(st.session_state.model, model_filename)
                    if st.session_state.scaler is not None:
                        joblib.dump(st.session_state.scaler, "scaler.pkl")
                    st.success(f"Model saved as {model_filename}!")
        else:
            st.info("👈 Click 'Train Model' to start training.")

# Page 4: Predict AQI
elif page == "🔮 Predict AQI":
    st.header("🔮 Predict Air Quality Index")
    
    if not st.session_state.trained or st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the 'Model Training' page.")
        
        # Allow loading saved model
        st.sidebar.subheader("Load Saved Model")
        uploaded_model = st.sidebar.file_uploader("Upload Model File", type=['pkl'])
        if uploaded_model is not None:
            try:
                import io
                model = joblib.load(io.BytesIO(uploaded_model.read()))
                st.session_state.model = model
                st.session_state.trained = True
                # Set default feature names if not available
                if st.session_state.feature_names is None:
                    st.session_state.feature_names = ['PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 
                                                      'Temperature', 'Humidity', 'WindSpeed', 
                                                      'RespiratoryCases', 'CardiovascularCases', 
                                                      'HospitalAdmissions', 'HealthImpactScore']
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    else:
        st.subheader("Enter Air Quality Parameters")
        
        # Input Form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            PM10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=100.0, step=1.0)
            PM2_5 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=50.0, step=1.0)
            NO2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=30.0, step=1.0)
            SO2 = st.number_input("SO2 (µg/m³)", min_value=0.0, value=20.0, step=1.0)
        
        with col2:
            O3 = st.number_input("O3 (µg/m³)", min_value=0.0, value=60.0, step=1.0)
            Temperature = st.number_input("Temperature (°C)", min_value=-50.0, value=25.0, step=1.0)
            Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            WindSpeed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=5.0, step=0.1)
        
        with col3:
            RespiratoryCases = st.number_input("Respiratory Cases", min_value=0, value=4, step=1)
            CardiovascularCases = st.number_input("Cardiovascular Cases", min_value=0, value=2, step=1)
            HospitalAdmissions = st.number_input("Hospital Admissions", min_value=0, value=1, step=1)
            HealthImpactScore = st.number_input("Health Impact Score", min_value=0.0, value=80.0, step=1.0)
        
        # Predict Button
        if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
            try:
                # Prepare input data
                input_data = {
                    'PM10': PM10,
                    'PM2_5': PM2_5,
                    'NO2': NO2,
                    'SO2': SO2,
                    'O3': O3,
                    'Temperature': Temperature,
                    'Humidity': Humidity,
                    'WindSpeed': WindSpeed,
                    'RespiratoryCases': RespiratoryCases,
                    'CardiovascularCases': CardiovascularCases,
                    'HospitalAdmissions': HospitalAdmissions,
                    'HealthImpactScore': HealthImpactScore
                }
                
                # Convert to DataFrame
                df_input = pd.DataFrame([input_data])
                
                # Ensure correct column order
                if st.session_state.feature_names is not None:
                    # Reorder columns to match training data
                    df_input = df_input[st.session_state.feature_names]
                else:
                    # Use default order if feature_names not available
                    default_order = ['PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 
                                    'Temperature', 'Humidity', 'WindSpeed', 
                                    'RespiratoryCases', 'CardiovascularCases', 
                                    'HospitalAdmissions', 'HealthImpactScore']
                    df_input = df_input[default_order]
                
                # Predict
                model = st.session_state.model
                
                # Check if model needs scaling (Linear Regression)
                if isinstance(model, LinearRegression) and st.session_state.scaler is not None:
                    df_input_scaled = st.session_state.scaler.transform(df_input)
                    predicted_aqi = model.predict(df_input_scaled)[0]
                else:
                    predicted_aqi = model.predict(df_input)[0]
                
                # Get category and recommendation
                category, recommendation, color = get_aqi_category(predicted_aqi)
                
                # Display Results
                st.success("Prediction completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>Predicted AQI</h2>
                        <h1 style='color: white; margin: 1rem 0;'>{predicted_aqi:.2f}</h1>
                        <h3 style='color: white; margin: 0;'>{category}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 2rem; border-radius: 10px;'>
                        <h3>💡 Recommendation</h3>
                        <p style='font-size: 1.1rem;'>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Input Summary
                st.subheader("📋 Input Summary")
                input_summary = pd.DataFrame([input_data]).T
                input_summary.columns = ['Value']
                st.dataframe(input_summary, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Make sure you have trained a model first.")

# Page 5: Model Performance
elif page == "📉 Model Performance":
    st.header("📉 Model Performance Analysis")
    
    if not st.session_state.trained or st.session_state.results is None:
        st.warning("⚠️ Please train a model first in the 'Model Training' page.")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        if results is None or len(results) == 0:
            st.warning("⚠️ No model results available. Please train a model first.")
        else:
            st.subheader("📊 Prediction vs Actual Comparison")
            
            # Get the first available model for visualization
            model_name = list(results.keys())[0]
            y_pred = results[model_name]['predictions']
        
            # Scatter Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Scatter plot: Predicted vs Actual
            axes[0].scatter(y_test, y_pred, alpha=0.5, color='blue')
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual AQI')
            axes[0].set_ylabel('Predicted AQI')
            axes[0].set_title('Predicted vs Actual AQI')
            axes[0].grid(True, alpha=0.3)
            
            # Residual Plot
            residuals = y_test - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.5, color='green')
            axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[1].set_xlabel('Predicted AQI')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residual Plot')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Distribution Comparison
            st.subheader("📈 Distribution Comparison")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(y_test, bins=30, alpha=0.5, label='Actual AQI', color='blue')
            ax.hist(y_pred, bins=30, alpha=0.5, label='Predicted AQI', color='red')
            ax.set_xlabel('AQI')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution: Actual vs Predicted AQI')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This application predicts Air Quality Index (AQI) based on various environmental and health parameters.

**Features:**
- Data visualization and EDA
- Multiple ML models
- Real-time predictions
- Performance analysis
""")

