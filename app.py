# ==============================================
# Configuration - MUST BE FIRST STREAMLIT COMMAND
# ==============================================
import streamlit as st
st.set_page_config(
    page_title="Student Analytics Dashboard", 
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
import os
from streamlit_autorefresh import st_autorefresh
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Initialize session state at the module level
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'interventions' not in st.session_state:
    st.session_state.interventions = pd.DataFrame(columns=[
        'Student_ID', 'Date', 'Intervention', 'Status'
    ])
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Auto-refresh every 5 minutes
st_autorefresh(interval=5*60*1000, key="data_refresh")

# ==============================================
# Data Loading & Preparation
# ==============================================
def load_data():
    """Load and process student data"""
    try:
        # Try multiple possible file locations
        try_paths = [
            "final_dataset.csv",
            "./data/final_dataset.csv",
            "https://raw.githubusercontent.com/sheetalN-2003/Excelerator-Project/main/final_dataset.csv"
        ]
        
        for path in try_paths:
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    st.toast(f"Data loaded successfully from: {path}", icon="✅")
                    return process_data(df)
            except Exception as e:
                st.warning(f"Failed to load from {path}: {str(e)}")
                continue
                
        # Fallback to file uploader
        uploaded_file = st.file_uploader("Upload student data CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            return process_data(df)
                
        st.error("Could not load data from any attempted path")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Critical error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Processing data...")
def process_data(df):
    """Data processing and feature engineering"""
    try:
        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate features
        if 'Signup_Date' in df.columns and 'Last_Activity' in df.columns:
            df['Days_Inactive'] = (pd.to_datetime('today') - df['Last_Activity']).dt.days
            df['Engagement_Score'] = np.log1p(df.get('Login_Count', 1)) / (df['Days_Inactive'] + 1)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    except Exception as e:
        st.error(f"Data processing error: {e}")
        return df

def prepare_retention_data(df):
    """Prepare data for retention prediction"""
    try:
        # Create target variable
        status_mapping = {
            'Active': 0,
            'Inactive': 1,
            'Dropped': 1,
            'Abandoned': 1
        }
        
        if 'Status Description' in df.columns:
            df['At_Risk'] = df['Status Description'].map(status_mapping).fillna(0)
        
        # Feature engineering
        if 'Signup_Date' in df.columns:
            df['Activity_Gap'] = (df.get('Last_Activity', pd.to_datetime('today')) - df['Signup_Date']).dt.days
            df['Weekday_Signup'] = df['Signup_Date'].dt.dayofweek
            df['Month_Signup'] = df['Signup_Date'].dt.month
        
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
            'Activity_Gap', 'Weekday_Signup', 'Month_Signup'
        ]
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            st.error("No valid features found for modeling")
            return None, None, None, None
            
        X = pd.get_dummies(df[available_features].dropna())
        y = df.dropna(subset=available_features)['At_Risk']
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42)
        
        # Model training with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        for i in range(1, 101):
            time.sleep(0.02)  # Simulate training progress
            progress_bar.progress(i)
            status_text.text(f"Training model... {i}%")
        
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        progress_bar.empty()
        status_text.empty()
        
        return model, X.columns.tolist(), report, (fpr, tpr, roc_auc)
        
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None, None, None

def predict_risk(df, model, features):
    """Generate risk predictions"""
    try:
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        X = pd.get_dummies(df[available_features])
        
        # Ensure all expected columns exist
        missing_cols = set(features) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[features]
        
        df['Risk_Score'] = model.predict_proba(X)[:, 1]
        df['Risk_Category'] = pd.cut(df['Risk_Score'], 
                                   bins=[0, 0.3, 0.7, 1],
                                   labels=['Low', 'Medium', 'High'],
                                   include_lowest=True)
        return df.sort_values('Risk_Score', ascending=False)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return df

# ==============================================
# Visualization Components
# ==============================================
def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Random'
    ))
    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    return fig

def plot_feature_importance(model, features):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = px.bar(
        x=[features[i] for i in indices],
        y=importances[indices],
        title="Feature Importance",
        labels={'x': 'Features', 'y': 'Importance'}
    )
    fig.update_layout(showlegend=False)
    return fig

# ==============================================
# Dashboard Components
# ==============================================
def show_model_metrics(report, roc_data=None):
    """Display model evaluation metrics"""
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Classification Metrics")
        metrics_df = pd.DataFrame(report).transpose()
        st.dataframe(
            metrics_df.style.format("{:.2f}"),
            use_container_width=True
        )
        
        if roc_data:
            st.plotly_chart(plot_roc_curve(*roc_data), use_container_width=True)
    
    with col2:
        st.markdown("##### Feature Importance")
        if 'risk_model' in st.session_state and 'model_features' in st.session_state:
            st.plotly_chart(
                plot_feature_importance(
                    st.session_state.risk_model,
                    st.session_state.model_features
                ),
                use_container_width=True
            )
        else:
            st.warning("No model available for feature importance")

def real_time_metrics():
    """Display real-time metrics in sidebar"""
    try:
        with st.sidebar:
            st.markdown("### 📊 Real-time Metrics")
            with stylable_container(
                key="metrics_container",
                css_styles="""
                {
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px);
                    background-color: white;
                }
                """,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Time", datetime.datetime.now().strftime('%H:%M:%S'))
                with col2:
                    if 'start_time' in st.session_state:
                        st.metric("Uptime", f"{(time.time() - st.session_state.start_time)/60:.1f} min")
                    else:
                        st.metric("Uptime", "N/A")
            
            st.markdown("### 🔄 Data Refresh")
            if st.button("Refresh Data", help="Click to manually refresh all data"):
                st.cache_data.clear()
                st.rerun()
    except Exception as e:
        # Silently fail to avoid thread errors
        pass

# ==============================================
# Main Dashboard Layout
# ==============================================
def main():
    st.title("🎓 Student Retention Analytics Dashboard")
    
    # Initialize real-time metrics
    real_time_metrics()
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data loaded - please upload a CSV file or check your data source")
        return
    
    # Apply filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Dynamic filters based on available columns
        filter_options = {}
        if 'Course' in df.columns:
            filter_options['Course'] = st.multiselect(
                "Select Courses",
                options=df['Course'].unique(),
                default=df['Course'].unique()[:2]
            )
        if 'Status Description' in df.columns:
            filter_options['Status Description'] = st.multiselect(
                "Select Status",
                options=df['Status Description'].unique(),
                default=['Active', 'Inactive'] if 'Active' in df['Status Description'].unique() else df['Status Description'].unique()[:2]
            )
    
    # Apply active filters
    filtered_df = df.copy()
    for col, values in filter_options.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    
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
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Students", len(filtered_df))
        
        if 'At_Risk' in retention_df.columns:
            with cols[1]:
                st.metric("At Risk", f"{retention_df['At_Risk'].mean():.1%}", 
                         delta_color="inverse")
        
        if 'Engagement_Score' in filtered_df.columns:
            with cols[2]:
                st.metric("Avg Engagement", f"{filtered_df['Engagement_Score'].mean():.2f}")
        
        with cols[3]:
            st.metric("Data Freshness", filtered_df['Signup_Date'].max().strftime('%Y-%m-%d') 
                     if 'Signup_Date' in filtered_df.columns else "N/A")
        
        style_metric_cards()
        
        # Interactive data explorer
        st.subheader("Data Explorer")
        filtered_data = dataframe_explorer(filtered_df)
        AgGrid(filtered_data, height=300, fit_columns_on_grid_load=True)
        
        # Activity trends
        if 'Signup_Date' in filtered_df.columns:
            st.subheader("Activity Trends")
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                signup_counts = filtered_df.set_index('Signup_Date').resample('W').size()
                fig = px.line(
                    signup_counts,
                    title="Weekly Signups",
                    labels={'value': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with trend_col2:
                if 'Engagement_Score' in filtered_df.columns:
                    engagement_trend = filtered_df.groupby(
                        filtered_df['Signup_Date'].dt.to_period("M"))['Engagement_Score'].mean().reset_index()
                    fig = px.bar(
                        engagement_trend,
                        x='Signup_Date',
                        y='Engagement_Score',
                        title="Monthly Engagement"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Retention analytics
        st.header("Student Retention Analysis")
        
        if st.session_state.get('model_trained', False):
            risk_df = predict_risk(
                retention_df,
                st.session_state.risk_model,
                st.session_state.model_features
            )
            
            if 'Risk_Score' in risk_df.columns:
                # Risk distribution
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    fig = px.pie(
                        risk_df,
                        names='Risk_Category',
                        title='Risk Category Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with dist_col2:
                    fig = px.histogram(
                        risk_df,
                        x='Risk_Score',
                        nbins=20,
                        title='Risk Score Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # High-risk students table
                st.subheader("High-Risk Students")
                
                # Configure AgGrid
                gb = GridOptionsBuilder.from_dataframe(
                    risk_df[['Learner ID', 'Age', 'Status Description', 'Risk_Score', 'Risk_Category']]
                    .sort_values('Risk_Score', ascending=False)
                    .head(50)
                )
                gb.configure_pagination(paginationPageSize=10)
                gb.configure_selection('single', use_checkbox=True)
                gb.configure_column('Risk_Score', type=["numericColumn","numberColumnFilter","customNumericFormat"], 
                                  precision=2)
                grid_options = gb.build()
                
                grid_response = AgGrid(
                    risk_df,
                    gridOptions=grid_options,
                    height=400,
                    width='100%',
                    data_return_mode='AS_INPUT',
                    update_mode='MODEL_CHANGED',
                    fit_columns_on_grid_load=True,
                    theme='streamlit'
                )
                
                selected = grid_response['selected_rows']
                if selected:
                    st.write("Selected Student:")
                    st.json(selected[0])
            else:
                st.warning("Risk scores not calculated - check your data")
        else:
            st.warning("No trained model - train one in the Model Management tab")
    
    with tab3:
        # Intervention system
        st.header("Student Interventions")
        
        # Add new intervention
        with st.expander("➕ Add New Intervention", expanded=True):
            student_col = 'Learner ID' if 'Learner ID' in df.columns else 'Student_ID'
            if student_col in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    student_id = st.selectbox(
                        "Select Student",
                        options=filtered_df[student_col].unique(),
                        key="intervention_student"
                    )
                
                with col2:
                    status = st.selectbox(
                        "Intervention Status",
                        options=['Pending', 'In Progress', 'Completed', 'Failed'],
                        key="intervention_status"
                    )
                
                intervention = st.text_area(
                    "Intervention Plan Details",
                    placeholder="Describe the intervention plan...",
                    height=150
                )
                
                if st.button("💾 Save Intervention", key="save_intervention"):
                    new_intervention = pd.DataFrame([{
                        'Student_ID': student_id,
                        'Date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'Intervention': intervention,
                        'Status': status
                    }])
                    
                    st.session_state.interventions = pd.concat([
                        st.session_state.interventions,
                        new_intervention
                    ], ignore_index=True)
                    st.success("Intervention saved!")
                    st.balloons()
            else:
                st.warning("Student ID column not found in data")
        
        # View interventions
        st.subheader("Intervention History")
        if not st.session_state.interventions.empty:
            # Add filtering to interventions
            status_filter = st.multiselect(
                "Filter by Status",
                options=st.session_state.interventions['Status'].unique(),
                default=st.session_state.interventions['Status'].unique()
            )
            
            filtered_interventions = st.session_state.interventions
            if status_filter:
                filtered_interventions = filtered_interventions[
                    filtered_interventions['Status'].isin(status_filter)
                ]
            
            # Enhanced table display
            gb = GridOptionsBuilder.from_dataframe(filtered_interventions)
            gb.configure_pagination(paginationPageSize=5)
            gb.configure_default_column(groupable=True, sortable=True, filterable=True)
            grid_options = gb.build()
            
            AgGrid(
                filtered_interventions,
                gridOptions=grid_options,
                height=300,
                theme='streamlit',
                fit_columns_on_grid_load=True
            )
            
            # Export option
            if st.button("📤 Export Interventions to CSV"):
                csv = filtered_interventions.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"interventions_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
        else:
            st.info("No interventions recorded")
    
    with tab4:
        # Model management
        st.header("Retention Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🚀 Train New Model", expanded=True):
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        model, features, report, roc_data = train_retention_model(retention_df)
                        
                        if model:
                            st.session_state.risk_model = model
                            st.session_state.model_features = features
                            st.session_state.model_report = report
                            st.session_state.roc_data = roc_data
                            st.session_state.model_trained = True
                            joblib.dump(model, 'retention_model.joblib')
                            st.success("Model trained successfully!")
                            st.rerun()
        
        with col2:
            with st.expander("📥 Load Existing Model"):
                uploaded_model = st.file_uploader(
                    "Upload trained model (.joblib)",
                    type=["joblib"],
                    key="model_uploader"
                )
                
                if uploaded_model:
                    try:
                        model = joblib.load(uploaded_model)
                        st.session_state.risk_model = model
                        st.session_state.model_trained = True
                        st.success("Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
        
        if st.session_state.get('model_trained', False):
            st.success("✅ Model loaded and ready for predictions")
            
            if st.button("View Model Details"):
                show_model_metrics(
                    st.session_state.model_report,
                    st.session_state.roc_data
                )
            
            # Model deployment options
            with st.expander("🛠️ Model Deployment"):
                st.info("Coming soon: Model deployment to API endpoint")
                st.progress(30)
                st.caption("Feature in development")

if __name__ == "__main__":
    main()
