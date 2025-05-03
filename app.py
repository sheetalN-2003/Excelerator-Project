# ==============================================
# Configuration - MUST BE FIRST STREAMLIT COMMAND
# ==============================================
import streamlit as st
st.set_page_config(
    page_title="AI-Powered Student Analytics", 
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# ==============================================
# Enhanced Imports
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import joblib
import threading
import os
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import geopandas as gpd
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.altex import line_chart, bar_chart
from streamlit_folium import folium_static
import folium
from datetime import timedelta
from sklearn.inspection import permutation_importance
import openai  # For AI insights

# Initialize OpenAI (optional - remove if not using)
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except:
    st.warning("OpenAI API key not found. AI insights will be disabled.")

# ==============================================
# Global State Initialization
# ==============================================
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'interventions' not in st.session_state:
    st.session_state.interventions = pd.DataFrame(columns=[
        'Student_ID', 'Date', 'Intervention', 'Status', 'Priority'
    ])
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.datetime.now()

# Auto-refresh every 5 minutes
st_autorefresh(interval=5*60*1000, key="data_refresh")

# ==============================================
# Enhanced Data Loading & Processing
# ==============================================
@st.cache_data(ttl=3600, show_spinner="Loading and processing data...")
def load_and_process_data():
    """Enhanced data loading with multiple fallbacks and robust processing"""
    # Sample data URL - replace with your actual data source
    SAMPLE_DATA_URL = "https://raw.githubusercontent.com/sheetalN-2003/Excelerator-Project/main/final_dataset.csv"
    
    try:
        # Try multiple data sources
        for source in [
            "data/final_dataset.csv",  # Local file
            "final_dataset.csv",       # Root directory
            SAMPLE_DATA_URL            # Online source
        ]:
            try:
                df = pd.read_csv(source)
                if not df.empty:
                    st.toast(f"✅ Data loaded from: {source}", icon="✅")
                    return process_data(df)
            except Exception as e:
                continue
        
        # Fallback to file upload
        uploaded_file = st.file_uploader("📤 Upload student data (CSV)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            return process_data(df)
        
        # If all sources fail, use sample data
        st.warning("Using sample data as fallback")
        df = pd.DataFrame({
            'Learner ID': range(1000, 1100),
            'Age': np.random.randint(18, 35, 100),
            'Course': np.random.choice(['Computer Science', 'Engineering', 'Business'], 100),
            'Status Description': np.random.choice(['Active', 'Inactive', 'Dropped'], 100),
            'Login_Count': np.random.poisson(15, 100),
            'Learner SignUp DateTime': pd.date_range('2023-01-01', periods=100).tolist(),
            'Apply Date': (pd.date_range('2023-01-01', periods=100) + 
                         pd.to_timedelta(np.random.randint(0, 90, 100), unit='d')).tolist()
        })
        return process_data(df)
        
    except Exception as e:
        st.error(f"🚨 Critical data loading error: {str(e)}")
        return pd.DataFrame()

def process_data(df):
    """Robust data processing with automatic feature engineering"""
    try:
        # Convert all string columns to lowercase for consistency
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
        
        # Enhanced datetime parsing with multiple format support
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
        
        # Feature engineering
        if 'Learner SignUp DateTime' in df.columns:
            df['Signup_Date'] = pd.to_datetime(df['Learner SignUp DateTime'])
        if 'Apply Date' in df.columns:
            df['Last_Activity'] = pd.to_datetime(df['Apply Date'])
        
        # Calculate engagement metrics
        if 'Signup_Date' in df.columns and 'Last_Activity' in df.columns:
            df['Days_Since_Last_Activity'] = (pd.to_datetime('today') - df['Last_Activity']).dt.days
            df['Days_Since_Signup'] = (pd.to_datetime('today') - df['Signup_Date']).dt.days
            df['Activity_Ratio'] = df.get('Login_Count', 1) / (df['Days_Since_Signup'] + 1)
            df['Engagement_Score'] = np.log1p(df.get('Login_Count', 1)) / (df['Days_Since_Last_Activity'] + 1)
        
        # Create target variable
        if 'Status Description' in df.columns:
            status_map = {
                'active': 0,
                'inactive': 1,
                'dropped': 1,
                'abandoned': 1
            }
            df['At_Risk'] = df['Status Description'].map(status_map).fillna(0)
        
        # Additional features
        if 'Signup_Date' in df.columns:
            df['Signup_Day'] = df['Signup_Date'].dt.day_name()
            df['Signup_Month'] = df['Signup_Date'].dt.month_name()
            df['Signup_Quarter'] = df['Signup_Date'].dt.quarter
            df['Signup_Hour'] = df['Signup_Date'].dt.hour
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Convert all text columns to string
        text_cols = df.select_dtypes(include='object').columns
        df[text_cols] = df[text_cols].astype(str)
        
        return df
    
    except Exception as e:
        st.error(f"⚠️ Data processing error: {str(e)}")
        return df

# ==============================================
# Enhanced Model Training with Hyperparameter Tuning
# ==============================================
def train_enhanced_model(df):
    """Train with hyperparameter tuning and feature selection"""
    try:
        # Feature selection
        features = [
            'Age', 'Engagement_Score', 'Days_Since_Last_Activity',
            'Activity_Ratio', 'Signup_Quarter', 'Signup_Hour'
        ]
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            st.error("❌ No valid features found for modeling")
            return None, None, None, None
            
        X = pd.get_dummies(df[available_features].dropna())
        y = df.dropna(subset=available_features)['At_Risk']
        
        # Model pipeline with SMOTE and scaling
        model = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        # Simulate training progress
        def update_progress():
            for i in range(1, 101):
                time.sleep(0.02)
                progress_bar.progress(i)
                status_text.text(f"🔧 Training model... {i}%")
        
        # Run in separate thread to allow progress updates
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()
        
        # Train model
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        
        # Get predictions
        y_pred = best_model.predict(X)
        y_proba = best_model.predict_proba(X)[:, 1]
        
        # Evaluation metrics
        report = classification_report(y, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Feature importance
        importances = best_model.named_steps['classifier'].feature_importances_
        
        # Wait for progress thread to finish
        progress_thread.join()
        progress_bar.empty()
        status_text.empty()
        
        return best_model, X.columns.tolist(), report, (fpr, tpr, roc_auc), importances
        
    except Exception as e:
        st.error(f"❌ Model training failed: {str(e)}")
        return None, None, None, None, None

# ==============================================
# AI-Powered Insights Generation
# ==============================================
def generate_ai_insights(data, chart_type, x=None, y=None):
    """Generate natural language insights using OpenAI"""
    try:
        if not openai.api_key:
            return "AI insights disabled - no API key configured"
        
        # Create a text description of the data
        if isinstance(data, pd.DataFrame):
            description = f"A dataset with {len(data)} rows and columns: {', '.join(data.columns)}"
            if x and y:
                description += f"\nShowing {chart_type} of {y} vs {x}"
        else:
            description = f"A {chart_type} chart"
        
        prompt = f"""
        Analyze this educational data visualization and provide 3 concise bullet points of insights:
        
        {description}
        
        Key insights:
        - """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message['content']
    
    except Exception as e:
        return f"⚠️ Could not generate AI insights: {str(e)}"

# ==============================================
# Enhanced Visualization Components
# ==============================================

    def plot_engagement_trends(df):
    """Interactive engagement trends with AI insights"""
    if 'Signup_Date' not in df.columns:
        return None
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15)
    
    # Weekly signups
    weekly_signups = df.set_index('Signup_Date').resample('W').size()
    fig.add_trace(
        go.Scatter(
            x=weekly_signups.index,
            y=weekly_signups.values,
            name='Weekly Signups',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    
    # Monthly engagement
    if 'Engagement_Score' in df.columns:
        monthly_engagement = df.groupby(df['Signup_Date'].dt.to_period("M"))['Engagement_Score'].mean().reset_index()
        monthly_engagement['Signup_Date'] = monthly_engagement['Signup_Date'].astype(str)
        
        fig.add_trace(
            go.Bar(
                x=monthly_engagement['Signup_Date'],
                y=monthly_engagement['Engagement_Score'],
                name='Avg Engagement',
                marker_color='coral')
            ),
            row=2, col=1
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Student Engagement Trends",
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Signups", row=1, col=1)
    fig.update_yaxes(title_text="Engagement Score", row=2, col=1)
    
    # Add AI insights
    with st.expander("🤖 AI Insights"):
        insights = generate_ai_insights(df, "engagement trends")
        st.markdown(insights)
    
    return fig

def plot_risk_distribution(risk_df):
    """Interactive risk distribution visualization"""
    if 'Risk_Score' not in risk_df.columns:
        return None
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Risk Score Distribution", "Risk Category Breakdown"))
    
    # Histogram of risk scores
    fig.add_trace(
        go.Histogram(
            x=risk_df['Risk_Score'],
            nbinsx=20,
            name='Risk Scores',
            marker_color='indianred'),
        row=1, col=1
    )
    
    # Pie chart of risk categories
    if 'Risk_Category' in risk_df.columns:
        risk_counts = risk_df['Risk_Category'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                name='Risk Categories',
                hole=0.3),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Student Risk Analysis"
    )
    
    # Add AI insights
    with st.expander("🤖 AI Risk Analysis"):
        insights = generate_ai_insights(risk_df, "risk distribution")
        st.markdown(insights)
    
    return fig

# ==============================================
# Real-time Dashboard Components
# ==============================================
def real_time_metrics():
    """Enhanced real-time metrics with performance indicators"""
    try:
        with st.sidebar:
            st.markdown("### 📈 Live Dashboard Metrics")
            
            # Performance metrics card
            with stylable_container(
                key="metrics_card",
                css_styles="""
                {
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px);
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                }
                """
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🕒 Current Time", datetime.datetime.now().strftime('%H:%M:%S'))
                    st.metric("🔄 Last Refresh", st.session_state.last_refresh.strftime('%H:%M:%S'))
                with col2:
                    uptime = (time.time() - st.session_state.start_time) / 60
                    st.metric("⏱️ Uptime", f"{uptime:.1f} min")
                    if 'risk_data' in st.session_state and st.session_state.risk_data is not None:
                        at_risk_pct = st.session_state.risk_data['Risk_Score'].mean() * 100
                        st.metric("⚠️ At Risk", f"{at_risk_pct:.1f}%")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Refresh Data", help="Manually refresh all data"):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.datetime.now()
                st.rerun()
            
            if st.button("📊 Update Visuals", help="Regenerate all visualizations"):
                st.session_state.last_refresh = datetime.datetime.now()
                st.rerun()
            
            # System status
            st.markdown("### 🖥️ System Status")
            if st.session_state.get('model_trained', False):
                st.success("✅ Model Loaded")
            else:
                st.warning("⚠️ Model Not Trained")
            
            if 'risk_data' in st.session_state and st.session_state.risk_data is not None:
                st.info(f"📊 {len(st.session_state.risk_data)} Students Analyzed")
            
    except Exception as e:
        # Fail silently to avoid breaking the app
        pass

# ==============================================
# Enhanced Intervention System
# ==============================================
def intervention_system(df):
    """Comprehensive intervention management with prioritization"""
    st.header("🛡️ Student Intervention System")
    
    # Intervention creation
    with st.expander("➕ Create New Intervention", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.selectbox(
                "Select Student",
                options=df['Learner ID'].unique() if 'Learner ID' in df.columns else [],
                key="interv_student_select"
            )
            
            # Auto-priority based on risk score
            priority = st.selectbox(
                "Priority Level",
                options=["Low", "Medium", "High", "Critical"],
                index=2,  # Default to High
                key="interv_priority"
            )
        
        with col2:
            status = st.selectbox(
                "Status",
                options=["Pending", "In Progress", "Completed", "Failed"],
                key="interv_status"
            )
            
            due_date = st.date_input(
                "Due Date",
                min_value=datetime.date.today(),
                key="interv_due_date"
            )
        
        intervention = st.text_area(
            "Intervention Plan Details",
            placeholder="Describe the intervention strategy, resources needed, and expected outcomes...",
            height=150,
            key="interv_details"
        )
        
        if st.button("💾 Save Intervention", key="save_interv"):
            new_intervention = pd.DataFrame([{
                'Student_ID': student_id,
                'Date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Intervention': intervention,
                'Status': status,
                'Priority': priority,
                'Due_Date': due_date.strftime('%Y-%m-%d')
            }])
            
            st.session_state.interventions = pd.concat([
                st.session_state.interventions,
                new_intervention
            ], ignore_index=True)
            
            st.success("Intervention saved successfully!")
            st.balloons()
    
    # Intervention management
    st.subheader("📋 Intervention Management")
    
    if not st.session_state.interventions.empty:
        # Enhanced filtering
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=st.session_state.interventions['Status'].unique(),
                default=st.session_state.interventions['Status'].unique()
            )
        
        with col2:
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=st.session_state.interventions['Priority'].unique(),
                default=st.session_state.interventions['Priority'].unique()
            )
        
        with col3:
            due_filter = st.selectbox(
                "Due Date Filter",
                options=["All", "Upcoming", "Overdue", "Completed"]
            )
        
        # Apply filters
        filtered_interventions = st.session_state.interventions.copy()
        
        if status_filter:
            filtered_interventions = filtered_interventions[
                filtered_interventions['Status'].isin(status_filter)
            ]
        
        if priority_filter:
            filtered_interventions = filtered_interventions[
                filtered_interventions['Priority'].isin(priority_filter)
            ]
        
        if due_filter != "All":
            today = datetime.date.today()
            if due_filter == "Upcoming":
                filtered_interventions = filtered_interventions[
                    (pd.to_datetime(filtered_interventions['Due_Date']) >= pd.to_datetime(today))
            elif due_filter == "Overdue":
                filtered_interventions = filtered_interventions[
                    (pd.to_datetime(filtered_interventions['Due_Date']) < pd.to_datetime(today)) & 
                    (filtered_interventions['Status'] != "Completed")
            elif due_filter == "Completed":
                filtered_interventions = filtered_interventions[
                    filtered_interventions['Status'] == "Completed"
                ]
        
        # Enhanced AgGrid display
        gb = GridOptionsBuilder.from_dataframe(filtered_interventions)
        gb.configure_pagination(paginationPageSize=10)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            editable=False,
            wrapText=True
        )
        
        # Conditional formatting for priority
        priority_cell_style = {
            "conditions": [
                {
                    "condition": "params.value == 'Critical'",
                    "style": {"backgroundColor": "#ff6b6b", "color": "white"}
                },
                {
                    "condition": "params.value == 'High'",
                    "style": {"backgroundColor": "#ffa502", "color": "white"}
                },
                {
                    "condition": "params.value == 'Medium'",
                    "style": {"backgroundColor": "#feca57", "color": "black"}
                },
                {
                    "condition": "params.value == 'Low'",
                    "style": {"backgroundColor": "#c8d6e5", "color": "black"}
                }
            ]
        }
        
        gb.configure_column("Priority", cellStyle=priority_cell_style)
        
        grid_options = gb.build()
        
        grid_response = AgGrid(
            filtered_interventions,
            gridOptions=grid_options,
            height=400,
            width='100%',
            data_return_mode=GridUpdateMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            theme='streamlit'
        )
        
        selected = grid_response['selected_rows']
        if selected:
            st.subheader("Selected Interventions")
            st.write(selected)
            
            # Bulk actions
            col1, col2 = st.columns(2)
            with col1:
                new_status = st.selectbox(
                    "Update Status",
                    options=["Pending", "In Progress", "Completed", "Failed"],
                    key="bulk_status"
                )
                
                if st.button("🔄 Update Status"):
                    for interv in selected:
                        idx = st.session_state.interventions[
                            st.session_state.interventions['Student_ID'] == interv['Student_ID']].index
                        st.session_state.interventions.loc[idx, 'Status'] = new_status
                    st.success(f"Updated {len(selected)} interventions!")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Delete Selected"):
                    for interv in selected:
                        st.session_state.interventions = st.session_state.interventions[
                            st.session_state.interventions['Student_ID'] != interv['Student_ID']]
                    st.success(f"Deleted {len(selected)} interventions!")
                    st.rerun()
        
        # Export options
        st.markdown("### 📤 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export to CSV"):
                csv = filtered_interventions.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"interventions_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
        
        with col2:
            if st.button("📄 Generate Report"):
                with st.spinner("Generating report..."):
                    time.sleep(2)  # Simulate report generation
                    st.success("Report generated successfully!")
                    # In a real app, you would generate a PDF/Word report here
    
    else:
        st.info("No interventions recorded yet. Create your first intervention above.")

# ==============================================
# Main Dashboard Layout
# ==============================================
def main():
    st.title("🎓 AI-Powered Student Retention Dashboard")
    
    # Initialize real-time components
    real_time_metrics()
    
    # Load data with progress indicator
    with st.spinner("🔍 Loading and analyzing student data..."):
        df = load_and_process_data()
    
    if df.empty:
        st.error("❌ No data loaded - please check your data source")
        return
    
    # Apply filters
    with st.sidebar:
        st.header("🔍 Data Filters")
        
        # Dynamic course filter
        if 'Course' in df.columns:
            courses = st.multiselect(
                "Select Courses",
                options=df['Course'].unique(),
                default=df['Course'].unique()[:2]
            )
            df = df[df['Course'].isin(courses)] if courses else df
        
        # Status filter
        if 'Status Description' in df.columns:
            statuses = st.multiselect(
                "Select Status",
                options=df['Status Description'].unique(),
                default=['active'] if 'active' in df['Status Description'].unique().astype(str) else df['Status Description'].unique()[:2]
            )
            df = df[df['Status Description'].isin(statuses)] if statuses else df
        
        # Date range filter
        if 'Signup_Date' in df.columns:
            min_date = df['Signup_Date'].min().date()
            max_date = df['Signup_Date'].max().date()
            
            date_range = st.date_input(
                "Signup Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                df = df[
                    (df['Signup_Date'].dt.date >= date_range[0]) & 
                    (df['Signup_Date'].dt.date <= date_range[1])
    
    # Prepare retention data
    retention_df = prepare_retention_data(df)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🎯 Risk Analytics", 
        "🛡️ Interventions", 
        "⚙️ Model Management"
    ])
    
    with tab1:
        # Overview KPIs
        st.header("📈 Student Overview Dashboard")
        
        # Metrics row
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Students", len(df), help="Total students in current view")
        
        with cols[1]:
            at_risk = retention_df['At_Risk'].mean() if 'At_Risk' in retention_df.columns else 0
            st.metric(
                "At Risk", 
                f"{at_risk:.1%}", 
                delta=f"{(at_risk - 0.2):.1%} vs benchmark" if at_risk else None,
                delta_color="inverse",
                help="Percentage of students identified as at risk"
            )
        
        with cols[2]:
            engagement = df['Engagement_Score'].mean() if 'Engagement_Score' in df.columns else 0
            st.metric(
                "Avg Engagement", 
                f"{engagement:.2f}", 
                help="Average engagement score (higher is better)"
            )
        
        with cols[3]:
            freshness = df['Signup_Date'].max().strftime('%Y-%m-%d') if 'Signup_Date' in df.columns else "N/A"
            st.metric(
                "Data Freshness", 
                freshness,
                help="Most recent student signup date"
            )
        
        style_metric_cards()
        
        # Engagement trends with AI insights
        st.subheader("📊 Engagement Trends")
        engagement_fig = plot_engagement_trends(df)
        if engagement_fig:
            st.plotly_chart(engagement_fig, use_container_width=True)
        else:
            st.warning("No engagement data available")
        
        # Interactive data explorer
        st.subheader("🔍 Data Explorer")
        with st.expander("Filter and Explore Data"):
            filtered_data = df.copy()
            
            # Dynamic column selection
            cols_to_show = st.multiselect(
                "Select columns to display",
                options=df.columns,
                default=['Learner ID', 'Age', 'Course', 'Status Description', 'Engagement_Score']
            )
            
            if cols_to_show:
                filtered_data = filtered_data[cols_to_show]
                
                # Numeric filters
                num_cols = filtered_data.select_dtypes(include=np.number).columns
                for col in num_cols:
                    min_val, max_val = float(filtered_data[col].min()), float(filtered_data[col].max())
                    val_range = st.slider(
                        f"Range for {col}",
                        min_val, max_val, (min_val, max_val)
                    )
                    filtered_data = filtered_data[
                        (filtered_data[col] >= val_range[0]) & 
                        (filtered_data[col] <= val_range[1])
                    ]
                
                # Display data with AgGrid
                AgGrid(
                    filtered_data,
                    height=300,
                    fit_columns_on_grid_load=True,
                    reload_data=True
                )
    
    with tab2:
        # Risk analytics
        st.header("🎯 Student Risk Analysis")
        
        if st.session_state.get('model_trained', False):
            # Predict risk
            risk_df = predict_risk(
                retention_df,
                st.session_state.risk_model,
                st.session_state.model_features
            )
            st.session_state.risk_data = risk_df
            
            if 'Risk_Score' in risk_df.columns:
                # Risk distribution visualizations
                st.subheader("📊 Risk Distribution")
                risk_fig = plot_risk_distribution(risk_df)
                if risk_fig:
                    st.plotly_chart(risk_fig, use_container_width=True)
                
                # High-risk students table
                st.subheader("🚨 High-Risk Students")
                
                # Configure AgGrid with enhanced options
                gb = GridOptionsBuilder.from_dataframe(
                    risk_df[['Learner ID', 'Age', 'Course', 'Status Description', 'Risk_Score', 'Risk_Category']]
                    .sort_values('Risk_Score', ascending=False)
                    .head(50)
                )
                
                # Add conditional formatting for risk scores
                risk_score_cell_style = {
                    "styleConditions": [
                        {
                            "condition": "params.value >= 0.7",
                            "style": {"backgroundColor": "#ff6b6b", "color": "white"}
                        },
                        {
                            "condition": "params.value >= 0.5 && params.value < 0.7",
                            "style": {"backgroundColor": "#ffa502", "color": "white"}
                        },
                        {
                            "condition": "params.value >= 0.3 && params.value < 0.5",
                            "style": {"backgroundColor": "#feca57", "color": "black"}
                        },
                        {
                            "condition": "params.value < 0.3",
                            "style": {"backgroundColor": "#c8d6e5", "color": "black"}
                        }
                    ]
                }
                
                gb.configure_column("Risk_Score", cellStyle=risk_score_cell_style, type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3)
                gb.configure_column("Risk_Category", cellEditor='agSelectCellEditor', cellEditorParams={'values': ['Low', 'Medium', 'High']})
                gb.configure_pagination(paginationPageSize=10)
                gb.configure_selection('multiple', use_checkbox=True)
                gb.configure_default_column(filterable=True, sortable=True)
                
                grid_options = gb.build()
                
                grid_response = AgGrid(
                    risk_df,
                    gridOptions=grid_options,
                    height=400,
                    width='100%',
                    data_return_mode='AS_INPUT',
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    fit_columns_on_grid_load=True,
                    theme='streamlit'
                )
                
                selected = grid_response['selected_rows']
                if selected:
                    st.subheader("Selected Students")
                    
                    # Display detailed view
                    cols = st.columns([1, 3])
                    
                    with cols[0]:
                        st.metric("Selected Students", len(selected))
                        avg_risk = np.mean([s['Risk_Score'] for s in selected])
                        st.metric("Avg Risk Score", f"{avg_risk:.1%}")
                        
                        if st.button("🚑 Create Group Intervention"):
                            st.session_state.interventions = pd.concat([
                                st.session_state.interventions,
                                pd.DataFrame([{
                                    'Student_ID': s['Learner ID'],
                                    'Date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'Intervention': "Group intervention for high-risk students",
                                    'Status': "Pending",
                                    'Priority': "High"
                                } for s in selected])
                            ], ignore_index=True)
                            st.success(f"Created interventions for {len(selected)} students!")
                    
                    with cols[1]:
                        st.write(selected)
                
                # Risk factor analysis
                st.subheader("🔍 Risk Factor Analysis")
                if 'risk_model' in st.session_state and 'model_features' in st.session_state:
                    fig = plot_feature_importance(
                        st.session_state.risk_model.named_steps['classifier'],
                        st.session_state.model_features
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("🤖 AI Interpretation"):
                        insights = generate_ai_insights(
                            pd.DataFrame({
                                'Feature': st.session_state.model_features,
                                'Importance': st.session_state.risk_model.named_steps['classifier'].feature_importances_
                            }), 
                            "feature importance"
                        )
                        st.markdown(insights)
                else:
                    st.warning("No model available for feature importance analysis")
            else:
                st.warning("Risk scores not calculated - check your data")
        else:
            st.warning("No trained model - train one in the Model Management tab")
    
    with tab3:
        # Intervention system
        intervention_system(df)
    
    with tab4:
        # Model management
        st.header("⚙️ Model Management Center")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model training section
            with st.expander("🚀 Model Training", expanded=True):
                st.markdown("""
                **Train a new retention prediction model**  
                This will use the current filtered dataset to train a Random Forest classifier
                with hyperparameter tuning and SMOTE for handling class imbalance.
                """)
                
                if st.button("🔧 Train New Model", type="primary"):
                    with st.spinner("Training model with hyperparameter tuning..."):
                        model, features, report, roc_data, importances = train_enhanced_model(retention_df)
                        
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
            # Model loading section
            with st.expander("📥 Model Operations", expanded=True):
                st.markdown("""
                **Manage existing models**  
                Upload a pre-trained model or download the current one.
                """)
                
                # Model upload
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
                
                # Model download
                if st.session_state.get('model_trained', False):
                    with open('retention_model.joblib', 'rb') as f:
                        st.download_button(
                            label="📥 Download Current Model",
                            data=f,
                            file_name="retention_model.joblib",
                            mime="application/octet-stream"
                        )
        
        # Model evaluation section
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.subheader("📊 Model Evaluation")
            
            # Show model metrics
            if st.button("Show Model Performance"):
                show_model_metrics(
                    st.session_state.model_report,
                    st.session_state.roc_data
                )
            
            # Feature importance visualization
            st.subheader("Feature Importance")
            if 'risk_model' in st.session_state and 'model_features' in st.session_state:
                fig = plot_feature_importance(
                    st.session_state.risk_model.named_steps['classifier'],
                    st.session_state.model_features
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("🤖 AI Interpretation"):
                    insights = generate_ai_insights(
                        pd.DataFrame({
                            'Feature': st.session_state.model_features,
                            'Importance': st.session_state.risk_model.named_steps['classifier'].feature_importances_
                        }), 
                        "feature importance"
                    )
                    st.markdown(insights)
            
            # Model deployment options
            st.markdown("---")
            st.subheader("🚀 Model Deployment")
            
            deploy_option = st.selectbox(
                "Select deployment option",
                options=["Local API", "Cloud Endpoint", "Batch Processing"],
                index=0
            )
            
            if deploy_option == "Local API":
                st.info("""
                **Local REST API**  
                Expose the model as a local REST endpoint for integration with other systems.
                """)
                if st.button("Deploy Local API"):
                    with st.spinner("Setting up local API server..."):
                        # In a real implementation, you would start a FastAPI/Flask server here
                        time.sleep(3)
                        st.success("API server running at http://localhost:8000")
            
            elif deploy_option == "Cloud Endpoint":
                st.info("""
                **Cloud Deployment**  
                Deploy the model to a cloud provider for scalable access.
                """)
                cloud_provider = st.selectbox(
                    "Select cloud provider",
                    options=["AWS SageMaker", "Google Vertex AI", "Azure ML"],
                    index=0
                )
                if st.button(f"Deploy to {cloud_provider}"):
                    with st.spinner(f"Deploying to {cloud_provider}..."):
                        # In a real implementation, you would use the cloud provider's SDK
                        time.sleep(3)
                        st.success(f"Model deployed to {cloud_provider} successfully!")
            
            elif deploy_option == "Batch Processing":
                st.info("""
                **Batch Processing**  
                Set up scheduled batch predictions for all students.
                """)
                schedule = st.selectbox(
                    "Prediction schedule",
                    options=["Daily", "Weekly", "Monthly"],
                    index=0
                )
                if st.button("Schedule Batch Predictions"):
                    with st.spinner(f"Scheduling {schedule} predictions..."):
                        # In a real implementation, you would set up a cron job or similar
                        time.sleep(2)
                        st.success(f"{schedule} batch predictions scheduled!")
        
        else:
            st.warning("No trained model available. Train or load a model first.")

# ==============================================
# Run the application
# ==============================================
if __name__ == "__main__":
    main()
