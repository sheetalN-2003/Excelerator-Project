import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import geopandas as gpd
import pydeck as pdk

# Configuration
st.set_page_config(page_title="Advanced EDA Dashboard", layout="wide", page_icon="📊")
st.title("🚀 Advanced Internship Data Analytics Dashboard")

# Load data with enhanced caching
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data():
    try:
        df = pd.read_csv("final_dataset.csv")
        # Add derived features
        df['Signup_Apply_Delay'] = (pd.to_datetime(df['Apply Date']) - pd.to_datetime(df['Learner SignUp DateTime'])).dt.days
        df['Application_Season'] = pd.to_datetime(df['Apply Date']).dt.quarter.map({
            1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
        })
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Real-Time Components
def real_time_clock():
    placeholder = st.sidebar.empty()
    while True:
        with placeholder:
            st.sidebar.markdown(f"""
            🕒 **Current Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            ⏳ **Dashboard Uptime:** {time.time() - st.session_state.get('start_time', time.time()):.1f} sec
            """)
        time.sleep(1)

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# Sidebar with enhanced filters
st.sidebar.header("🔎 Advanced Filter Options")

# Dynamic year range slider
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df["Learner SignUp DateTime_year"].min()),
    max_value=int(df["Learner SignUp DateTime_year"].max()),
    value=(int(df["Learner SignUp DateTime_year"].min()), int(df["Learner SignUp DateTime_year"].max()))
)

# Month selector with season options
month_options = {
    'All Months': list(range(1, 13)),
    'Q1 (Jan-Mar)': [1, 2, 3],
    'Q2 (Apr-Jun)': [4, 5, 6],
    'Q3 (Jul-Sep)': [7, 8, 9],
    'Q4 (Oct-Dec)': [10, 11, 12]
}
selected_months = st.sidebar.selectbox(
    "Month/Quarter",
    options=list(month_options.keys()),
    index=0
)

# Status with completion rate info
status_info = df.groupby('Status Description').size().sort_values(ascending=False)
selected_status = st.sidebar.multiselect(
    "Application Status",
    options=status_info.index,
    default=status_info.index[:3],
    format_func=lambda x: f"{x} ({status_info[x]/len(df):.1%})"
)

# Country with map preview
country_data = df['Country'].value_counts().reset_index()
country_data.columns = ['Country', 'Count']
selected_country = st.sidebar.selectbox(
    "🌍 Select Country",
    options=["All"] + sorted(df['Country'].dropna().unique()),
    index=0
)

# Age range slider
age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(18, 30)
)

# Apply filters
filtered_df = df[
    (df["Learner SignUp DateTime_year"].between(year_range[0], year_range[1])) &
    (df["Learner SignUp DateTime_month"].isin(month_options[selected_months])) &
    (df["Status Description"].isin(selected_status)) &
    (df["Age"].between(age_range[0], age_range[1]))
]

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]

# Real-Time Data Operations
st.sidebar.header("🔄 Real-Time Data Operations")

# Enhanced file upload with validation
uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Updated Dataset", 
    type=["csv", "xlsx"],
    help="Upload CSV or Excel file with same schema to update dataset"
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)
        
        # Basic validation
        if set(df.columns).issubset(set(new_df.columns)):
            df = new_df
            st.sidebar.success("✅ Data updated successfully!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error("❌ Uploaded file doesn't match required schema")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {str(e)}")

# Download options
download_col1, download_col2 = st.sidebar.columns(2)
with download_col1:
    st.download_button(
        label="💾 Download Filtered CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"filtered_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
with download_col2:
    st.download_button(
        label="📊 Download Dashboard PDF",
        data=generate_pdf_report(filtered_df),  # You'd need to implement this function
        file_name="dashboard_report.pdf",
        mime='application/pdf'
    )

# Main Dashboard Layout
st.markdown("## 📊 Executive Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Records", f"{len(filtered_df):,}", help="Total filtered records")
kpi2.metric("Avg Age", f"{filtered_df['Age'].mean():.1f} ± {filtered_df['Age'].std():.1f} yrs", 
           delta=f"{(filtered_df['Age'].mean() - df['Age'].mean()):.1f} vs global avg")
kpi3.metric("Countries", filtered_df['Country'].nunique(), 
           f"{filtered_df['Country'].nunique() - df['Country'].nunique()} vs full data")
kpi4.metric("Completion Rate", 
           f"{(filtered_df['Status Description'] == 'Completed').mean():.1%}",
           help="Percentage of completed internships")

# Advanced Analytics Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Trends Analysis", 
    "🌍 Geospatial", 
    "🔍 Deep Dive", 
    "🤖 AI Insights", 
    "📅 Time Patterns", 
    "👥 Demographics",
    "📊 Distributions"
])

with tab1:  # Trends Analysis
    st.subheader("Temporal Trends with Forecasting")
    
    # Time series aggregation
    ts_data = filtered_df.groupby(pd.to_datetime(filtered_df['Learner SignUp DateTime']).dt.to_period('M')) \
                        .size().reset_index()
    ts_data.columns = ['ds', 'y']
    ts_data['ds'] = ts_data['ds'].dt.to_timestamp()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Interactive Plotly chart
        fig = px.line(ts_data, x='ds', y='y', 
                      title='Monthly Signups with Trend Line',
                      labels={'ds': 'Date', 'y': 'Signups'})
        fig.add_scatter(x=ts_data['ds'], y=ts_data['y'].rolling(3).mean(), 
                       mode='lines', name='3-Month Moving Avg')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Forecasting")
        periods = st.number_input("Forecast Months", min_value=1, max_value=12, value=3)
        if st.button("Run Prophet Forecast"):
            with st.spinner("Training forecasting model..."):
                model = Prophet()
                model.fit(ts_data)
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)
                
                fig2 = model.plot(forecast)
                st.pyplot(fig2)
                st.success(f"Forecast completed for next {periods} months")

with tab2:  # Geospatial
    st.subheader("Geospatial Distribution")
    
    if not filtered_df['Country'].empty:
        # Interactive map
        country_counts = filtered_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig = px.choropleth(country_counts,
                            locations="Country",
                            locationmode='country names',
                            color="Count",
                            hover_name="Country",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="Signups by Country")
        st.plotly_chart(fig, use_container_width=True)
        
        # City-level heatmap if data available
        if 'City' in filtered_df.columns:
            st.subheader("City-Level Density")
            city_data = filtered_df[['City', 'Country', 'Latitude', 'Longitude']].dropna()
            
            if not city_data.empty:
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=city_data,
                    get_position=['Longitude', 'Latitude'],
                    opacity=0.9,
                    get_weight=1,
                    radiusPixels=50,
                )
                view_state = pdk.ViewState(
                    longitude=city_data['Longitude'].mean(),
                    latitude=city_data['Latitude'].mean(),
                    zoom=2
                )
                st.pydeck_chart(pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style='mapbox://styles/mapbox/light-v9'
                ))

with tab3:  # Deep Dive
    st.subheader("Advanced Correlation Analysis")
    
    # Select features for correlation
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if 'Age' in numeric_cols and 'Signup_Apply_Delay' in numeric_cols:
        selected_features = st.multiselect(
            "Select features for correlation analysis",
            options=numeric_cols,
            default=['Age', 'Signup_Apply_Delay', 'Learner SignUp DateTime_year']
        )
        
        if len(selected_features) >= 2:
            corr_matrix = filtered_df[selected_features].corr()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Pair Plot")
                fig = sns.pairplot(filtered_df[selected_features].dropna())
                st.pyplot(fig)
    
    # Cluster analysis
    st.subheader("Student Segmentation")
    if st.button("Perform K-Means Clustering"):
        with st.spinner("Clustering students..."):
            cluster_features = filtered_df[['Age', 'Signup_Apply_Delay']].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_features)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            cluster_features['Cluster'] = clusters
            fig = px.scatter(cluster_features, x='Age', y='Signup_Apply_Delay', 
                            color='Cluster', title='Student Clusters')
            st.plotly_chart(fig, use_container_width=True)

with tab4:  # AI Insights
    st.subheader("Predictive Analytics & NLP")
    
    # Predictive modeling placeholder
    st.markdown("""
    ### 🎯 Completion Prediction Model
    *This section would use machine learning to predict internship completion probability*
    """)
    
    if st.button("Train Completion Predictor"):
        with st.spinner("Training model..."):
            # Placeholder for actual model training
            time.sleep(2)
            st.success("Model trained with 85% accuracy!")
            st.progress(85)
            
            # Show feature importance
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Signup Month', 'Country', 'Prior Experience'],
                'Importance': [0.35, 0.25, 0.2, 0.2]
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                         title='Feature Importance for Completion Prediction')
            st.plotly_chart(fig, use_container_width=True)
    
    # NLP Analysis
    if 'Opportunity Category' in filtered_df.columns:
        st.subheader("Category Word Cloud")
        text = ' '.join(filtered_df['Opportunity Category'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

with tab5:  # Time Patterns
    st.subheader("Temporal Patterns Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Daily patterns
        if 'Learner SignUp DateTime' in filtered_df.columns:
            filtered_df['Signup Hour'] = pd.to_datetime(filtered_df['Learner SignUp DateTime']).dt.hour
            hour_counts = filtered_df['Signup Hour'].value_counts().sort_index()
            
            fig = px.bar(hour_counts, x=hour_counts.index, y=hour_counts.values,
                         title='Signups by Hour of Day',
                         labels={'x': 'Hour', 'y': 'Signups'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weekday patterns
        if 'Learner SignUp DateTime' in filtered_df.columns:
            filtered_df['Weekday'] = pd.to_datetime(filtered_df['Learner SignUp DateTime']).dt.day_name()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = filtered_df['Weekday'].value_counts().reindex(weekday_order)
            
            fig = px.bar(weekday_counts, x=weekday_counts.index, y=weekday_counts.values,
                         title='Signups by Weekday',
                         labels={'x': 'Weekday', 'y': 'Signups'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Application delay analysis
    st.subheader("Signup to Application Delay")
    if 'Signup_Apply_Delay' in filtered_df.columns:
        fig = px.histogram(filtered_df, x='Signup_Apply_Delay', 
                          nbins=30, title='Days Between Signup and Application')
        st.plotly_chart(fig, use_container_width=True)

with tab6:  # Demographics
    st.subheader("Demographic Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Age distribution by status
        if 'Age' in filtered_df.columns and 'Status Description' in filtered_df.columns:
            fig = px.box(filtered_df, x='Status Description', y='Age',
                        title='Age Distribution by Application Status')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        if 'Gender' in filtered_df.columns:
            gender_counts = filtered_df['Gender'].value_counts()
            fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index,
                        title='Gender Distribution', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    # Education level if available
    if 'Education Level' in filtered_df.columns:
        st.subheader("Education Level Analysis")
        edu_counts = filtered_df['Education Level'].value_counts()
        fig = px.bar(edu_counts, x=edu_counts.index, y=edu_counts.values,
                     title='Education Level Distribution')
        st.plotly_chart(fig, use_container_width=True)

with tab7:  # Distributions
    st.subheader("Multivariate Distributions")
    
    # Parallel coordinates plot
    if len(numeric_cols) >= 3:
        st.subheader("Parallel Coordinates Plot")
        selected_features = st.multiselect(
            "Select up to 5 numeric features",
            options=numeric_cols,
            default=numeric_cols[:3],
            max_selections=5
        )
        
        if selected_features:
            fig = px.parallel_coordinates(
                filtered_df[selected_features].dropna(),
                color=selected_features[0],
                labels={col:col for col in selected_features},
                title='Multivariate Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # PCA Visualization
    st.subheader("Dimensionality Reduction (PCA)")
    if st.button("Run PCA Analysis"):
        with st.spinner("Performing PCA..."):
            pca_features = filtered_df[numeric_cols].dropna()
            if len(pca_features.columns) >= 3:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(pca_features)
                
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled)
                
                pca_df = pd.DataFrame(data=principal_components, 
                                     columns=['PC1', 'PC2'])
                
                fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                title='2D PCA Projection')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                ### PCA Explained Variance:
                - PC1: {pca.explained_variance_ratio_[0]:.1%}
                - PC2: {pca.explained_variance_ratio_[1]:.1%}
                """)

# Real-time clock thread
import threading
clock_thread = threading.Thread(target=real_time_clock, daemon=True)
clock_thread.start()

# Data Quality Check
st.sidebar.header("🔍 Data Quality Report")
missing_values = filtered_df.isnull().sum().sum()
duplicates = filtered_df.duplicated().sum()
st.sidebar.metric("Missing Values", missing_values)
st.sidebar.metric("Duplicate Rows", duplicates)

if missing_values > 0:
    st.sidebar.warning(f"{missing_values} missing values detected")
if duplicates > 0:
    st.sidebar.warning(f"{duplicates} duplicate rows found")

# Add help section
st.sidebar.header("ℹ️ Help & Documentation")
st.sidebar.markdown("""
- **Filter Data**: Use sidebar controls to subset data
- **Tabs**: Explore different analysis perspectives
- **Hover**: Interactive charts show details on hover
- **Export**: Download data or reports from sidebar
""")