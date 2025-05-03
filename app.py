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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Student Drop-off Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #e9ecef;
    }
    .st-ax {
        background-color: #f8f9fa;
    }
    .css-18e3th9 {
        padding: 2rem 1rem 10rem;
    }
    .css-1d391kg {
        padding: 2rem 1rem 1rem;
    }
    .plot-container {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        background-color: white;
        margin-bottom: 20px;
    }
    .model-card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        background-color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('final_dataset.csv')
    
    # Create target variable for drop-off prediction
    # Assuming 'Status Description' indicates drop-off (simplified for example)
    df['drop_off'] = df['Status Description'].apply(lambda x: 1 if x in ['Withdrawn', 'Rejected'] else 0)
    
    return df

df = load_data()

# Sidebar
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", ["Data Overview", "Exploratory Analysis", "Predictive Modeling", "Model Insights"])

# Main content
st.title("🎓 Student Drop-off Prediction Dashboard")
st.markdown("""
This interactive dashboard helps analyze student enrollment patterns and predict potential drop-offs 
using machine learning models.
""")

if page == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Dataset Summary")
        st.write(f"Total records: {len(df)}")
        st.write(f"Number of features: {len(df.columns)}")
        
        # Basic stats
        st.markdown("**Basic Statistics**")
        st.write(df.describe())
    
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Feature', 'Missing Count']
    missing_data['Missing %'] = (missing_data['Missing Count'] / len(df)) * 100
    st.dataframe(missing_data.sort_values('Missing %', ascending=False))
    
    st.subheader("Data Types")
    st.write(df.dtypes.reset_index().rename(columns={'index': 'Feature', 0: 'Type'}))

elif page == "Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    st.subheader("Monthly Signups Trend")
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_signups = df.groupby(['Learner SignUp DateTime_year', 'Learner SignUp DateTime_month']).size()
    monthly_signups.index = [f"{int(y)}-{int(m):02}" for y, m in monthly_signups.index]
    monthly_signups = monthly_signups.sort_index()
    sns.lineplot(data=monthly_signups, marker="o", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Status Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        counts = df["Status Description"].value_counts(dropna=False)
        donut_width = 0.4
        wedges, texts, autotexts = ax.pie(
            counts, labels=counts.index, autopct='%1.1f%%', 
            startangle=0, pctdistance=1-donut_width/2,
            wedgeprops={'width': donut_width}
        )
        centre_circle = plt.Circle((0, 0), donut_width, fc='white')
        ax.add_artist(centre_circle)
        ax.set_title('Distribution of Status Description')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        counts = df["Opportunity Category"].value_counts(dropna=False)
        donut_width = 0.4
        wedges, texts, autotexts = ax.pie(
            counts, labels=counts.index, autopct='%1.1f%%', 
            startangle=0, pctdistance=1-donut_width/2,
            wedgeprops={'width': donut_width}
        )
        centre_circle = plt.Circle((0, 0), donut_width, fc='white')
        ax.add_artist(centre_circle)
        ax.set_title('Distribution of Opportunity Category')
        st.pyplot(fig)
    
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Age', data=df, orient='h', color='skyblue', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Gender Distribution by Country (Top 5)")
    top_countries = df['Country'].value_counts().head(5).index
    gender_country_data = df[df['Country'].isin(top_countries)].groupby(['Country', 'Gender']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_country_data.plot(kind='bar', figsize=(10, 6), colormap='Pastel1', ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

elif page == "Predictive Modeling":
    st.header("🤖 Predictive Modeling for Student Drop-offs")
    
    st.markdown("""
    ### Model Selection and Training
    This section allows you to train different machine learning models to predict student drop-offs.
    """)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Preprocess data for modeling
    @st.cache_data
    def preprocess_data(df):
        # Select features and target
        features = ['Age', 'Gender', 'Country', 'Opportunity Category', 
                   'Learner SignUp DateTime_month', 'Learner SignUp DateTime_year']
        target = 'drop_off'
        
        # Filter data
        model_df = df[features + [target]].copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in ['Gender', 'Country', 'Opportunity Category']:
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            
        # Handle missing values
        model_df.fillna(model_df.median(), inplace=True)
        
        return model_df, features, target
    
    model_df, features, target = preprocess_data(df)
    
    # Show feature importance (placeholder)
    st.write("Selected Features:", features)
    
    # Model selection
    st.subheader("Model Configuration")
    model_type = st.selectbox("Select Model Type", 
                             ["Logistic Regression", "Decision Tree", "Random Forest"])
    
    # Model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    if model_type == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
    elif model_type == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
    
    # Train/test split
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state
    )
    
    # Train model
    if st.button("Train Model"):
        st.subheader("Model Training Results")
        
        with st.spinner("Training model..."):
            if model_type == "Logistic Regression":
                model = LogisticRegression(random_state=random_state)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model Performance")
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                st.markdown("**Classification Report**")
                st.table(pd.DataFrame(report).transpose())
            
            with col2:
                # Confusion matrix
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            # ROC curve
            st.markdown("### ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   line=dict(dash='dash'), name='Random'))
            fig.update_layout(
                title='Receiver Operating Characteristic',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800, height=500
            )
            st.plotly_chart(fig)
            
            # Save model to session state
            st.session_state.model = model
            st.session_state.features = features
            st.success("Model trained successfully!")

elif page == "Model Insights":
    st.header("🔎 Model Insights and Interpretation")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the Predictive Modeling section.")
    else:
        model = st.session_state.model
        features = st.session_state.features
        
        st.subheader("Feature Importance")
        
        if isinstance(model, LogisticRegression):
            # Coefficients for logistic regression
            importance = model.coef_[0]
            fig = px.bar(x=features, y=importance, 
                         labels={'x': 'Feature', 'y': 'Coefficient'},
                         title='Feature Coefficients (Logistic Regression)')
            st.plotly_chart(fig)
            
            st.markdown("""
            **Interpretation:**
            - Positive coefficients indicate features that increase the likelihood of drop-off
            - Negative coefficients indicate features that decrease the likelihood of drop-off
            """)
        
        elif isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
            # Feature importance for tree-based models
            importance = model.feature_importances_
            fig = px.bar(x=features, y=importance, 
                         labels={'x': 'Feature', 'y': 'Importance'},
                         title='Feature Importance (Tree-based Models)')
            st.plotly_chart(fig)
            
            st.markdown("""
            **Interpretation:**
            - Higher values indicate features that are more important for the model's predictions
            - These features have a stronger influence on the drop-off prediction
            """)
        
        st.subheader("What-If Analysis")
        st.markdown("Predict drop-off probability for a hypothetical student:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=15, max_value=80, value=25)
            gender = st.selectbox("Gender", df['Gender'].unique())
        
        with col2:
            country = st.selectbox("Country", df['Country'].unique())
            category = st.selectbox("Opportunity Category", df['Opportunity Category'].unique())
        
        with col3:
            month = st.selectbox("Signup Month", range(1, 13), format_func=lambda x: datetime(2023, x, 1).strftime('%B'))
            year = st.number_input("Signup Year", min_value=2020, max_value=2025, value=2023)
        
        if st.button("Predict Drop-off Probability"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Country': [country],
                'Opportunity Category': [category],
                'Learner SignUp DateTime_month': [month],
                'Learner SignUp DateTime_year': [year]
            })
            
            # Encode categorical variables (same as training)
            le = LabelEncoder()
            for col in ['Gender', 'Country', 'Opportunity Category']:
                input_data[col] = le.fit_transform(input_data[col].astype(str))
            
            # Predict
            proba = model.predict_proba(input_data)[0][1]
            
            st.metric("Probability of Drop-off", f"{proba:.1%}")
            
            # Interpretation
            if proba < 0.3:
                st.success("Low risk of drop-off")
            elif proba < 0.7:
                st.warning("Medium risk of drop-off")
            else:
                st.error("High risk of drop-off")

# Footer
st.markdown("---")
st.markdown("""
**Student Drop-off Prediction Dashboard**  
Created with Streamlit | Data Science Team
""")
