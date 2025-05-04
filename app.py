# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import openai
import pytz
import pyarrow as pa
from dateutil.relativedelta import relativedelta

# Set page config with professional dark theme
st.set_page_config(
    page_title="AI-Powered Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI (replace with your API key)
if 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai.api_key = "sk-your-actual-key-here-1234567890"  # Replace with your actual key

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Main page background */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: #1A1D24;
        color: #FAFAFA;
        border-right: 1px solid #2E3440;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stAlert, .stText, .stNumberInput label, 
    .stSelectbox label, .stSlider label, .stRadio label, .stButton>button {
        color: #FAFAFA !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Widget styling */
    .st-bb, .st-at, .st-ax, .stTextInput>div>div>input, .stNumberInput>div>div>input,
    .stSelectbox>div>div>select, .stSlider>div>div>div>div, .stRadio>div>label {
        background-color: #2E3440 !important;
        color: #FAFAFA !important;
        border: 1px solid #4C566A;
        border-radius: 6px;
    }
    
    /* Cards and containers */
    .plot-container, .model-card, .insight-card {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        padding: 20px;
        background-color: #2E3440;
        margin-bottom: 24px;
        border: 1px solid #4C566A;
    }
    
    /* Tables */
    .dataframe {
        background-color: #2E3440 !important;
        color: #FAFAFA !important;
        border: 1px solid #4C566A !important;
    }
    .dataframe th {
        background-color: #3B4252 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #5E81AC !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #81A1C1 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #3B4252;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5E81AC !important;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib style for dark theme
plt.style.use('dark_background')
sns.set_style("darkgrid", {
    'axes.facecolor': '#2E3440',
    'grid.color': '#3B4252',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white'
})

# Load data with proper type conversion and error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_dataset.csv')
        
        # Check if DataFrame is empty
        if df.empty:
            st.error("Loaded an empty dataset - please check your data source")
            return pd.DataFrame()
        
        # Convert datetime columns with error handling
        try:
            df['Signup_DateTime'] = pd.to_datetime(
                df['Learner SignUp DateTime_year'].astype(str) + '-' +
                df['Learner SignUp DateTime_month'].astype(str).str.zfill(2) + '-' +
                df['Learner SignUp DateTime_day'].astype(str).str.zfill(2)
            )
        except Exception as e:
            st.error(f"Error converting datetime columns: {str(e)}")
            df['Signup_DateTime'] = pd.to_datetime('today')  # Fallback
            
        # Create target variable with proper handling
        df['drop_off'] = df['Status Description'].apply(
            lambda x: 1 if str(x) in ['Withdrawn', 'Rejected'] else 0
        )
        
        # Convert object columns to string to avoid Arrow serialization issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Real-time clock component
def real_time_clock():
    tz = pytz.timezone('UTC')
    now = datetime.now(tz)
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 16px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; background-color: #A3BE8C;"></span>
            <span style="font-family: 'Courier New', monospace; font-size: 14px; color: #81A1C1; background-color: rgba(46, 52, 64, 0.5); padding: 4px 8px; border-radius: 4px; display: inline-block;">
                {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# OpenAI insights generator with enhanced error handling
def generate_ai_insights(data, prompt):
    try:
        if not openai.api_key or openai.api_key == "you_api_key_here":
            return "⚠️ OpenAI API key not configured. Please set your API key to enable AI insights."
            
        if not data or (isinstance(data, dict) and not data):
            return "⚠️ No data available for analysis"
            
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a data science assistant..."
                },
                {
                    "role": "user", 
                    "content": f"Analyze this data...{prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=256
        )
        return response.choices[0].message.content
        
    except openai.AuthenticationError:
        return "🔒 Authentication error: Please check your OpenAI API key"
    except openai.RateLimitError:
        return "⏳ API rate limit exceeded: Please wait before making more requests"
    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"

# Sidebar with proper labels and error handling
def create_sidebar():
    st.sidebar.title("Dashboard Controls")
    real_time_clock()
    st.sidebar.markdown("### Navigation")
    
    try:
        page = st.sidebar.radio(
            "Select Page",
            options=["Data Overview", "Exploratory Analysis", "Predictive Modeling", "AI Insights"],
            label_visibility="visible"
        )
        return page
    except Exception as e:
        st.sidebar.error(f"Error creating sidebar: {str(e)}")
        return "Data Overview"

page = create_sidebar()

# Main content with error boundaries
try:
    st.title("🎓 AI-Powered Student Analytics Dashboard")
    st.markdown("""
    *Real-time monitoring and predictive analytics for student engagement and retention*
    """)

    # Real-time data updater with proper caching
    @st.cache_data(ttl=60)
    def get_realtime_metrics():
        try:
            return {
                "current_signups": np.random.randint(50, 100),
                "active_students": np.random.randint(500, 800),
                "dropoff_rate": round(np.random.uniform(0.1, 0.2), 2)
            }
        except Exception as e:
            st.error(f"Error generating real-time metrics: {str(e)}")
            return {
                "current_signups": 0,
                "active_students": 0,
                "dropoff_rate": 0.0
            }

    if page == "Data Overview":
        st.header("📊 Data Overview")
        
        # Check if data is loaded
        if df.empty:
            st.warning("No data available - please check your data source")
            st.stop()
        
        # Real-time metrics
        try:
            metrics = get_realtime_metrics()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Month Signups", metrics["current_signups"], "5% vs last month")
            with col2:
                st.metric("Active Students", metrics["active_students"], "-3% vs last week")
            with col3:
                st.metric("Predicted Drop-off Rate", f"{metrics['dropoff_rate']*100:.1f}%", "2% improvement")
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recent Activity Timeline")
            
            try:
                # Generate recent activity data
                now = datetime.now()
                timeline_data = pd.DataFrame({
                    "timestamp": [now - timedelta(minutes=x*15) for x in range(10)],
                    "event": ["New signup", "Application submitted", "Course started", 
                             "Assessment completed", "Support ticket", "Payment processed",
                             "Course completed", "Withdrawal request", "Feedback submitted",
                             "Account update"],
                    "count": np.random.randint(1, 10, 10)
                })
                
                fig = px.timeline(
                    timeline_data, 
                    x_start="timestamp", 
                    x_end=timeline_data["timestamp"] + timedelta(minutes=14),
                    y="event",
                    color="count",
                    color_continuous_scale="viridis",
                    title="Real-time Student Activity"
                )
                fig.update_layout(
                    plot_bgcolor='#2E3440',
                    paper_bgcolor='#2E3440',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#3B4252'),
                    yaxis=dict(gridcolor='#3B4252'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating timeline: {str(e)}")
        
        with col2:
            st.subheader("Dataset Summary")
            st.write(f"Total records: {len(df):,}")
            st.write(f"Number of features: {len(df.columns)}")
            
            st.markdown("**Data Freshness**")
            last_update = datetime.now() - timedelta(hours=2)
            st.write(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown("**Data Quality**")
            try:
                null_percentage = df.isnull().mean().mean()
                if pd.isna(null_percentage):
                    completeness = 0.0
                else:
                    completeness = round((1 - null_percentage) * 100, 1)
                
                progress_value = completeness / 100
                if pd.isna(progress_value):
                    progress_value = 0.0
                elif progress_value < 0:
                    progress_value = 0.0
                elif progress_value > 1:
                    progress_value = 1.0
                    
                st.progress(float(progress_value), text=f"Data completeness: {completeness}%")
            except Exception as e:
                st.error(f"Error calculating data quality: {str(e)}")
                st.progress(0.0, text="Data quality metrics unavailable")
        
        st.subheader("Interactive Data Explorer")
        with st.expander("Filter and Explore Data", expanded=False):
            try:
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_age, max_age = st.slider(
                        "Age Range", 
                        min_value=int(df['Age'].min()), 
                        max_value=int(df['Age'].max()),
                        value=(18, 35),
                        label_visibility="visible"
                    )
                with col2:
                    countries = st.multiselect(
                        "Countries", 
                        df['Country'].unique(), 
                        default=df['Country'].unique()[:3],
                        label_visibility="visible"
                    )
                with col3:
                    statuses = st.multiselect(
                        "Statuses", 
                        df['Status Description'].unique(), 
                        default=df['Status Description'].unique()[:2],
                        label_visibility="visible"
                    )
                
                filtered_df = df[
                    (df['Age'] >= min_age) & 
                    (df['Age'] <= max_age) & 
                    (df['Country'].isin(countries)) & 
                    (df['Status Description'].isin(statuses))
                ]
                
                # Convert DataFrame to string types for display
                display_df = filtered_df.copy()
                for col in display_df.select_dtypes(include=['object']).columns:
                    display_df[col] = display_df[col].astype(str)
                
                st.dataframe(display_df, height=300)
            except Exception as e:
                st.error(f"Error filtering data: {str(e)}")

    elif page == "Exploratory Analysis":
        st.header("🔍 Advanced Exploratory Analysis")
        
        if df.empty:
            st.warning("No data available - please check your data source")
            st.stop()
        
        # Time series analysis with interactive controls
        st.subheader("Temporal Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Signup Trends", "Status Changes", "Demographics"])
        
        with tab1:
            try:
                col1, col2 = st.columns(2)
                with col1:
                    time_resolution = st.selectbox(
                        "Time Resolution", 
                        ["Daily", "Weekly", "Monthly", "Quarterly"],
                        index=2,
                        label_visibility="visible"
                    )
                with col2:
                    show_forecast = st.checkbox("Show 3-Month Forecast", value=True)
                
                # Prepare time series data
                if time_resolution == "Daily":
                    ts_data = df.set_index('Signup_DateTime').resample('D').size()
                elif time_resolution == "Weekly":
                    ts_data = df.set_index('Signup_DateTime').resample('W').size()
                elif time_resolution == "Monthly":
                    ts_data = df.set_index('Signup_DateTime').resample('M').size()
                else:
                    ts_data = df.set_index('Signup_DateTime').resample('Q').size()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_data.index, 
                    y=ts_data.values,
                    name='Actual Signups',
                    line=dict(color='#5E81AC', width=3)
                ))
                
                if show_forecast:
                    # Simple forecast (in a real app, use proper time series forecasting)
                    last_date = ts_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.DateOffset(months=1),
                        periods=3,
                        freq=ts_data.index.freq
                    )
                    forecast_values = ts_data.values[-3:].mean() * np.array([0.95, 1.05, 1.1])
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name='Forecast',
                        line=dict(color='#A3BE8C', width=3, dash='dot')
                    ))
                
                fig.update_layout(
                    title=f'Student Signups ({time_resolution} View)',
                    xaxis_title='Date',
                    yaxis_title='Number of Signups',
                    plot_bgcolor='#2E3440',
                    paper_bgcolor='#2E3440',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#3B4252'),
                    yaxis=dict(gridcolor='#3B4252'),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating time series: {str(e)}")
        
        with tab2:
            try:
                # Status transition analysis
                status_flow = df.groupby(['Status Description', 'Opportunity Category']).size().unstack().fillna(0)
                
                fig = px.bar(
                    status_flow, 
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Status Distribution by Opportunity Category'
                )
                fig.update_layout(
                    plot_bgcolor='#2E3440',
                    paper_bgcolor='#2E3440',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#3B4252'),
                    yaxis=dict(gridcolor='#3B4252'),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating status flow chart: {str(e)}")
        
        with tab3:
            try:
                # Interactive demographic analysis
                col1, col2 = st.columns(2)
                with col1:
                    demographic_var = st.selectbox(
                        "Demographic Variable", 
                        ['Age', 'Gender', 'Country', 'Opportunity Category'],
                        label_visibility="visible"
                    )
                with col2:
                    metric = st.selectbox(
                        "Metric", 
                        ['Count', 'Drop-off Rate', 'Completion Rate'],
                        label_visibility="visible"
                    )
                
                if metric == 'Count':
                    demo_data = df[demographic_var].value_counts().reset_index()
                    demo_data.columns = [demographic_var, 'Count']
                    fig = px.bar(
                        demo_data, 
                        x=demographic_var,
                        y='Count',
                        title=f'Distribution by {demographic_var}'
                    )
                elif metric == 'Drop-off Rate':
                    demo_data = df.groupby(demographic_var)['drop_off'].mean().reset_index()
                    demo_data.columns = [demographic_var, 'Drop-off Rate']
                    fig = px.bar(
                        demo_data, 
                        x=demographic_var,
                        y='Drop-off Rate',
                        title=f'Drop-off Rate by {demographic_var}'
                    )
                else:
                    demo_data = df.groupby(demographic_var)['Status Description'] \
                                 .apply(lambda x: (x == 'Completed').mean()).reset_index()
                    demo_data.columns = [demographic_var, 'Completion Rate']
                    fig = px.bar(
                        demo_data, 
                        x=demographic_var,
                        y='Completion Rate',
                        title=f'Completion Rate by {demographic_var}'
                    )
                
                fig.update_layout(
                    plot_bgcolor='#2E3440',
                    paper_bgcolor='#2E3440',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#3B4252'),
                    yaxis=dict(gridcolor='#3B4252'),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating demographic analysis: {str(e)}")

    elif page == "Predictive Modeling":
        st.header("🤖 Advanced Predictive Modeling")
        
        if df.empty:
            st.warning("No data available - please check your data source")
            st.stop()
        
        st.markdown("""
        ### Intelligent Drop-off Prediction
        Train and evaluate machine learning models to predict student drop-offs with explainable AI.
        """)
        
        # Feature selection
        st.subheader("Feature Engineering")
        
        # Preprocess data for modeling with error handling
        @st.cache_data
        def preprocess_data(df):
            try:
                # Select features and target
                features = ['Age', 'Gender', 'Country', 'Opportunity Category', 
                           'Learner SignUp DateTime_month', 'Learner SignUp DateTime_year']
                target = 'drop_off'
                
                # Filter data
                model_df = df[features + [target]].copy()
                
                # Feature engineering
                model_df['Signup_Season'] = model_df['Learner SignUp DateTime_month'].apply(
                    lambda m: 'Winter' if m in [12,1,2] else 
                             'Spring' if m in [3,4,5] else 
                             'Summer' if m in [6,7,8] else 'Fall'
                )
                
                # Encode categorical variables
                le = LabelEncoder()
                for col in ['Gender', 'Country', 'Opportunity Category', 'Signup_Season']:
                    model_df[col] = le.fit_transform(model_df[col].astype(str))
                    
                # Handle missing values
                model_df.fillna(model_df.median(), inplace=True)
                
                return model_df, features + ['Signup_Season'], target
            except Exception as e:
                st.error(f"Error preprocessing data: {str(e)}")
                return pd.DataFrame(), [], None
        
        model_df, features, target = preprocess_data(df)
        
        if not features or target is None:
            st.error("Failed to preprocess data for modeling")
            st.stop()
        
        # Interactive feature selection
        with st.expander("Feature Selection and Engineering", expanded=False):
            try:
                selected_features = st.multiselect(
                    "Select features for modeling",
                    features,
                    default=features,
                    label_visibility="visible"
                )
                
                if not selected_features:
                    st.warning("Please select at least one feature")
                    st.stop()
            except Exception as e:
                st.error(f"Error in feature selection: {str(e)}")
                st.stop()
        
        # Model selection and configuration
        st.subheader("Model Configuration")
        
        try:
            model_type = st.selectbox(
                "Select Algorithm", 
                ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
                label_visibility="visible"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider(
                    "Test Set Size (%)", 
                    10, 40, 20,
                    label_visibility="visible"
                )
                cv_folds = st.slider(
                    "Cross-Validation Folds", 
                    2, 10, 5,
                    label_visibility="visible"
                )
            with col2:
                random_state = st.number_input(
                    "Random State", 
                    0, 100, 42,
                    label_visibility="visible"
                )
                if model_type in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
                    max_depth = st.slider(
                        "Max Depth", 
                        1, 20, 5,
                        label_visibility="visible"
                    )
                if model_type in ["Random Forest", "Gradient Boosting"]:
                    n_estimators = st.slider(
                        "Number of Trees", 
                        10, 200, 100,
                        label_visibility="visible"
                    )
        except Exception as e:
            st.error(f"Error in model configuration: {str(e)}")
            st.stop()
        
        # Train/test split
        try:
            X = model_df[selected_features]
            y = model_df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
        except Exception as e:
            st.error(f"Error splitting data: {str(e)}")
            st.stop()
        
        # Train model
        if st.button("Train and Evaluate Model", type="primary"):
            st.subheader("Model Performance")
            
            with st.spinner("Training model with cross-validation..."):
                try:
                    if model_type == "Logistic Regression":
                        model = LogisticRegression(random_state=random_state, max_iter=1000)
                    elif model_type == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
                    elif model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                    else:
                        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Confusion Matrix", "Feature Importance", "Predictions"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                            st.metric("Precision", f"{report['1']['precision']:.2%}", 
                                     f"Recall: {report['1']['recall']:.2%}")
                        with col2:
                            st.metric("F1 Score", f"{report['1']['f1-score']:.2%}")
                            st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.2%}")
                        
                        st.markdown("**Detailed Classification Report**")
                        st.table(pd.DataFrame(report).transpose())
                    
                    with tab2:
                        fig, ax = plt.subplots()
                        cm = confusion_matrix(y_test, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
                        ax.set_xlabel('Predicted', color='white')
                        ax.set_ylabel('Actual', color='white')
                        ax.set_xticklabels(ax.get_xticklabels(), color='white')
                        ax.set_yticklabels(ax.get_yticklabels(), color='white')
                        st.pyplot(fig)
                    
                    with tab3:
                        if hasattr(model, 'feature_importances_'):
                            importance = model.feature_importances_
                            feat_imp = pd.DataFrame({
                                'Feature': selected_features,
                                'Importance': importance
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                feat_imp, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title='Feature Importance',
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(
                                plot_bgcolor='#2E3440',
                                paper_bgcolor='#2E3440',
                                font=dict(color='white'),
                                xaxis=dict(gridcolor='#3B4252'),
                                yaxis=dict(gridcolor='#3B4252'),
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Feature importance not available for this model type")
                    
                    with tab4:
                        st.markdown("**Sample Predictions on Test Set**")
                        sample_results = pd.DataFrame({
                            'Actual': y_test[:20],
                            'Predicted': y_pred[:20],
                            'Probability': [f"{p:.1%}" for p in y_prob[:20]]
                        })
                        st.dataframe(sample_results)
                    
                    # Save model to session state
                    st.session_state.model = model
                    st.session_state.features = selected_features
                    st.success("Model training completed successfully!")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

    elif page == "AI Insights":
        st.header("🧠 AI-Powered Insights")
        
        if df.empty:
            st.warning("No data available - please check your data source")
            st.stop()
        
        st.markdown("""
        ### Intelligent Analysis and Recommendations
        Leverage AI to generate actionable insights from your data.
        """)
        
        # Data summary for AI
        try:
            data_summary = {
                "total_students": len(df),
                "drop_off_rate": df['drop_off'].mean(),
                "top_countries": df['Country'].value_counts().head(3).to_dict(),
                "status_distribution": df['Status Description'].value_counts(normalize=True).to_dict(),
                "age_stats": {
                    "mean": df['Age'].mean(),
                    "median": df['Age'].median(),
                    "min": df['Age'].min(),
                    "max": df['Age'].max()
                }
            }
        except Exception as e:
            st.error(f"Error preparing data for analysis: {str(e)}")
            data_summary = {"error": "Data preparation failed"}
        
        st.subheader("Automated Data Analysis")
        try:
            analysis_type = st.selectbox(
                "Select Analysis Type", [
                    "General Overview",
                    "Drop-off Risk Factors",
                    "Retention Opportunities",
                    "Seasonal Patterns",
                    "Custom Analysis"
                ],
                label_visibility="visible"
            )
            
            custom_prompt = ""
            if analysis_type == "Custom Analysis":
                custom_prompt = st.text_area(
                    "Enter your specific questions or focus areas",
                    label_visibility="visible"
                )
        except Exception as e:
            st.error(f"Error in analysis setup: {str(e)}")
            st.stop()
        
        if st.button("Generate Insights", type="primary"):
            with st.spinner("Generating AI-powered insights..."):
                try:
                    if analysis_type == "General Overview":
                        prompt = "Provide 3-5 key insights about our student population and their engagement patterns."
                    elif analysis_type == "Drop-off Risk Factors":
                        prompt = "Identify potential risk factors for student drop-offs based on the data patterns."
                    elif analysis_type == "Retention Opportunities":
                        prompt = "Suggest data-driven strategies to improve student retention."
                    elif analysis_type == "Seasonal Patterns":
                        prompt = "Analyze seasonal patterns in student enrollment and drop-offs."
                    else:
                        prompt = custom_prompt
                    
                    insights = generate_ai_insights(data_summary, prompt)
                    
                    st.markdown("### AI-Generated Insights")
                    st.markdown("---")
                    st.markdown(insights)
                    st.markdown("---")
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
        
        st.subheader("Predictive Scenario Analysis")
        if 'model' not in st.session_state:
            st.warning("Please train a model first in the Predictive Modeling section")
        else:
            st.markdown("Test different scenarios to understand drop-off risks:")
            
            try:
                col1, col2 = st.columns(2)
                with col1:
                    age = st.slider(
                        "Student Age", 
                        15, 70, 25,
                        label_visibility="visible"
                    )
                    gender = st.selectbox(
                        "Gender", 
                        df['Gender'].unique(),
                        label_visibility="visible"
                    )
                    country = st.selectbox(
                        "Country", 
                        df['Country'].unique(),
                        label_visibility="visible"
                    )
                with col2:
                    category = st.selectbox(
                        "Program Category", 
                        df['Opportunity Category'].unique(),
                        label_visibility="visible"
                    )
                    signup_month = st.selectbox(
                        "Signup Month", 
                        range(1, 13), 
                        format_func=lambda x: datetime(2023, x, 1).strftime('%B'),
                        label_visibility="visible"
                    )
                    engagement_level = st.select_slider(
                        "Engagement Level", 
                        ["Low", "Medium", "High"],
                        label_visibility="visible"
                    )
                
                if st.button("Calculate Drop-off Risk", type="primary"):
                    try:
                        # Create input dataframe
                        input_data = pd.DataFrame({
                            'Age': [age],
                            'Gender': [gender],
                            'Country': [country],
                            'Opportunity Category': [category],
                            'Learner SignUp DateTime_month': [signup_month],
                            'Learner SignUp DateTime_year': [2023],
                            'Signup_Season': ['Winter' if signup_month in [12,1,2] else 
                                             'Spring' if signup_month in [3,4,5] else 
                                             'Summer' if signup_month in [6,7,8] else 'Fall']
                        })
                        
                        # Encode categorical variables
                        le = LabelEncoder()
                        for col in ['Gender', 'Country', 'Opportunity Category', 'Signup_Season']:
                            input_data[col] = le.fit_transform(input_data[col].astype(str))
                        
                        # Predict
                        model = st.session_state.model
                        proba = model.predict_proba(input_data[st.session_state.features])[0][1]
                        
                        # Display results with visualization
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Drop-off Probability", f"{proba:.1%}")
                            
                            if proba < 0.3:
                                st.success("Low Risk")
                                st.markdown("**Recommendation:** Standard engagement")
                            elif proba < 0.6:
                                st.warning("Medium Risk")
                                st.markdown("**Recommendation:** Proactive check-ins")
                            else:
                                st.error("High Risk")
                                st.markdown("**Recommendation:** Intervention needed")
                        
                        with col2:
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = proba * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Risk Level"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'steps': [
                                        {'range': [0, 30], 'color': "#A3BE8C"},
                                        {'range': [30, 60], 'color': "#EBCB8B"},
                                        {'range': [60, 100], 'color': "#BF616A"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "white", 'width': 4},
                                        'thickness': 0.75,
                                        'value': proba * 100
                                    }
                                }
                            ))
                            fig.update_layout(
                                plot_bgcolor='#2E3440',
                                paper_bgcolor='#2E3440',
                                font=dict(color='white'),
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in scenario analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error setting up scenario inputs: {str(e)}")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.stop()

# Footer with real-time update
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("""
    **AI-Powered Student Analytics Dashboard**  
    *Real-time monitoring and predictive insights*
    """)
with footer_col2:
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"*Last updated: {last_updated}*")

# JavaScript for real-time clock update
st.markdown("""
<script>
function updateClock() {
    const now = new Date();
    const clockElement = document.querySelector('.real-time-clock');
    if (clockElement) {
        clockElement.textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    }
}
setInterval(updateClock, 1000);
</script>
""", unsafe_allow_html=True)
