import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import joblib
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import geopandas as gpd
import pydeck as pdk
import threading

# ==============================================
# Configuration
# ==============================================
st.set_page_config(
    page_title="Student Analytics Dashboard", 
    layout="wide",
    page_icon="🎓"
)
st.title("🚀 Student Retention Analytics Dashboard")

# ==============================================
# Data Loading & Preparation
# ==============================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Try multiple possible file locations
        try_paths = [
            "student_data.csv",           # Same directory
            "./data/student_data.csv",    # In a data subfolder
            "https://raw.githubusercontent.com/sheetalN-2003/Excelerator-Project/refs/heads/main/final_dataset.csv"  # From GitHub
        ]
        
        for path in try_paths:
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    st.success(f"Data loaded successfully from: {path}")
                    return process_data(df)
            except Exception as e:
                st.warning(f"Failed to load from {path}: {str(e)}")
                continue
                
        st.error("Could not load data from any attempted path")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Critical error loading data: {e}")
        return pd.DataFrame()

def process_data(df):
    """Data processing and feature engineering"""
    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    # Calculate features
    if 'Last_Activity' in df.columns and 'Signup_Date' in df.columns:
        df['Days_Inactive'] = (pd.to_datetime('today') - df['Last_Activity']).dt.days
        df['Engagement_Score'] = df['Login_Count'] / (df['Days_Inactive'] + 1)
    
    return df

def prepare_retention_data(df):
    """Prepare data for retention prediction"""
    try:
        # Target variable
        df['At_Risk'] = np.where(
            (df['Status'].isin(['Inactive', 'Dropped'])) |
            (df['Days_Inactive'] > 30), 1, 0)
        
        # Feature engineering
        df['Activity_Gap'] = (df['Last_Activity'] - df['Signup_Date']).dt.days
        df['Weekday_Signup'] = df['Signup_Date'].dt.dayofweek
        
        return df
    except Exception as e:
        st.error(f"Data prep error: {e}")
        return df

# ==============================================
# Model Training & Prediction
# ==============================================
def train_retention_model(df):
    """Train Random Forest classifier"""
    try:
        features = [
            'Age', 'Engagement_Score', 'Days_Inactive',
            'Activity_Gap', 'Weekday_Signup', 'Course_Load'
        ]
        
        X = pd.get_dummies(df[features].dropna())
        y = df.dropna(subset=features)['At_Risk']
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42)
        
        # Model training
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        return model, X.columns.tolist(), report
        
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None, None

def predict_risk(df, model, features):
    """Generate risk predictions"""
    try:
        X = pd.get_dummies(df[features])
        
        # Ensure all expected columns exist
        missing_cols = set(features) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[features]
        
        df['Risk_Score'] = model.predict_proba(X)[:, 1]
        return df.sort_values('Risk_Score', ascending=False)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return df

# ==============================================
# Dashboard Components
# ==============================================
def real_time_clock():
    """Display live clock in sidebar"""
    placeholder = st.sidebar.empty()
    while True:
        with placeholder:
            st.sidebar.markdown(f"""
            🕒 **Current Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            ⏳ **Uptime:** {time.time() - st.session_state.get('start_time', time.time()):.1f}s
            """)
        time.sleep(1)

def show_model_metrics(report):
    """Display model evaluation metrics"""
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Classification Report:")
        st.code(report)
    
    with col2:
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

# ==============================================
# Main Dashboard Layout
# ==============================================
def main():
    # Initialize session state
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    if 'interventions' not in st.session_state:
        st.session_state.interventions = pd.DataFrame(columns=[
            'Student_ID', 'Date', 'Intervention', 'Status'
        ])
    
    # Start clock thread
    clock_thread = threading.Thread(target=real_time_clock, daemon=True)
    clock_thread.start()
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data loaded - check your data file")
        return
    
    # Apply filters
    with st.sidebar:
        st.header("🔍 Filters")
        course_filter = st.multiselect(
            "Select Courses",
            options=df['Course'].unique(),
            default=df['Course'].unique()[:2]
        )
        status_filter = st.multiselect(
            "Select Status",
            options=df['Status'].unique(),
            default=['Active', 'At Risk']
        )
    
    filtered_df = df[
        (df['Course'].isin(course_filter)) &
        (df['Status'].isin(status_filter))
    ]
    
    # Prepare retention data
    retention_df = prepare_retention_data(filtered_df)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🎯 Retention Analytics", 
        "🛡️ Interventions", 
        "⚙️ Model Management"
    ])
    
    with tab1:
        # Overview KPIs
        st.header("Student Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(filtered_df))
        col2.metric("At Risk", f"{retention_df['At_Risk'].mean():.1%}")
        col3.metric("Avg Engagement", f"{filtered_df['Engagement_Score'].mean():.2f}")
        
        # Activity trends
        st.subheader("Activity Trends")
        fig = px.line(
            filtered_df.groupby('Signup_Date').size().reset_index(),
            x='Signup_Date',
            y=0,
            title="Daily Signups"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Retention analytics
        st.header("Student Retention Analysis")
        
        if 'risk_model' in st.session_state:
            risk_df = predict_risk(
                retention_df,
                st.session_state.risk_model,
                st.session_state.model_features
            )
            
            st.subheader("High-Risk Students (Top 20)")
            st.dataframe(
                risk_df[['Student_ID', 'Course', 'Status', 'Risk_Score']]
                .head(20)
                .style.background_gradient(
                    subset=['Risk_Score'],
                    cmap='OrRd'
                ),
                use_container_width=True
            )
            
            # Risk distribution
            st.subheader("Risk Distribution")
            fig = px.histogram(
                risk_df,
                x='Risk_Score',
                nbins=20,
                title='Student Risk Scores'
            )
            st.plotly_chart(fig)
        else:
            st.warning("No trained model - train one in the Model Management tab")
    
    with tab3:
        # Intervention system
        st.header("Student Interventions")
        
        # Add new intervention
        with st.expander("➕ Add Intervention"):
            student_id = st.selectbox(
                "Select Student",
                options=filtered_df['Student_ID'].unique()
            )
            intervention = st.text_area("Intervention Plan")
            
            if st.button("Save Intervention"):
                new_intervention = pd.DataFrame([{
                    'Student_ID': student_id,
                    'Date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'Intervention': intervention,
                    'Status': 'Pending'
                }])
                
                st.session_state.interventions = pd.concat([
                    st.session_state.interventions,
                    new_intervention
                ], ignore_index=True)
                st.success("Intervention saved!")
        
        # View interventions
        st.subheader("Active Interventions")
        if not st.session_state.interventions.empty:
            st.dataframe(st.session_state.interventions)
        else:
            st.info("No interventions recorded")
    
    with tab4:
        # Model management
        st.header("Retention Model Management")
        
        if st.button("Train New Model"):
            with st.spinner("Training model..."):
                model, features, report = train_retention_model(retention_df)
                
                if model:
                    st.session_state.risk_model = model
                    st.session_state.model_features = features
                    joblib.dump(model, 'retention_model.joblib')
                    st.success("Model trained successfully!")
                    show_model_metrics(report)
        
        if 'risk_model' in st.session_state:
            st.success("Model loaded and ready for predictions")
            if st.button("View Model Details"):
                show_model_metrics(st.session_state.model_report)

if __name__ == "__main__":
    main()
