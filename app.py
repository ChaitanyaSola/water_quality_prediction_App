import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, 
                           confusion_matrix, classification_report)
from sklearn.impute import SimpleImputer
import joblib
import pickle
# ADD THESE TWO LINES HERE
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import streamlit as st

# # ===== BACKGROUND SETUP =====
# def set_background():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-image: url("hhttps://cdn.pixabay.com/photo/2019/10/18/13/08/rain-4559068_1280.jpg");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#         }
#         /* Makes content area readable */
#         .main .block-container {
#             background-color: rgba(255, 255, 255, 0.9);
#             border-radius: 10px;
#             padding: 2rem;
#             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#  Initialize (MUST be right after set_page_config)
# st.set_page_config(layout="wide")
# set_background()  # This applies to all pages automatically

#  ===== YOUR EXISTING CODE BELOW =====
#  (All your current app code goes here)  

# ----------------    
# Set page config
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load or train model
@st.cache_resource
def load_model():
    try:
        # Try to load pre-trained model
        model = joblib.load("Water Quality Prediction.pickle")
        ct = joblib.load('column_transformer.pkl')
        return model, ct
    except:
        # If no saved model, train a new one (using your code)
        df = pd.read_csv('Water_quality_cleaned.csv')  # Assuming you have this file
        
        y = df["Target"]
        x = df[[ 'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate',
            'Conductivity', 'Chlorine', 'Manganese',
            'Total Dissolved Solids', 'Water Temperature', 'Air Temperature']]

        numerical_pl = Pipeline(steps=[("scaler", RobustScaler())])
        ct = ColumnTransformer(transformers=[("scaler", numerical_pl, list(range(17)))])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=23)
        
        final_pl = ImbPipeline(steps=[
            ("columntrans", ct),
            ("smote", SMOTE(random_state=42)),
            ("algorithm", GaussianNB(var_smoothing=6.135506425409834e-06))
        ])
        
        final_pl.fit(x_train, y_train)
        
        # Save the model for future use
        joblib.dump(final_pl, 'Water Quality Prediction.pickle')
        joblib.dump(ct, 'column_transformer.pkl')
        
        return final_pl, ct

model, ct = load_model()

# Navigation
pages = {
    "🏠 Introduction": "intro",
    "📝 Problem Statement": "problem",
    "📊 Data Collection": "data",
    "🔍 EDA": "eda",
    "🤖 Model Development": "model",
    "⚡ Deployment": "deploy"
}

st.sidebar.title("🗂️Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Page 1: Introduction
if selection == "🏠 Introduction":
    st.title("💧 Water Quality Prediction System")

    
    st.markdown("""
    ## Welcome to the Water Quality Prediction System
    
    Access to clean and safe drinking water is essential for human health and well-being. 
    This application uses machine learning to predict water potability based on various 
    physicochemical parameters.
    
    **Why is this important?**
    - Over 2 billion people lack access to safe drinking water (WHO)
    - Contaminated water can transmit diseases like cholera, dysentery, and polio
    - Early detection of water quality issues can prevent health problems
    
    **How it works:**
    1. Input water quality parameters
    2. Our trained model analyzes the data
    3. Get instant prediction on water potability
    
    Navigate through the sections using the sidebar to learn more about our project.
    """)

# Page 2: Problem Statement & Objectives
elif selection == "📝 Problem Statement":
    st.title("Problem Statement & Objectives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝Problem Statement")
        st.markdown("""
        - Millions of people worldwide consume unsafe water daily
        - Traditional water testing methods are time-consuming and expensive
        - Many communities lack access to proper water testing facilities
        - There's a need for quick, affordable water quality assessment tools
        
        **Current Challenges:**
        - Complex relationships between water parameters and potability
        - Imbalanced datasets (more safe water samples than contaminated)
        - Non-linear patterns in water quality data
        """)
    
    with col2:
        st.header("🎯Project Objectives")
        st.markdown("""
        - Develop a machine learning model to predict water potability
        - Create an accessible interface for water quality assessment
        - Provide insights into key water quality parameters
        - Help communities identify potential water contamination
        
        **Key Features:**
        - Uses 17 different water quality parameters
        - Handles imbalanced data with SMOTE technique
        - Robust preprocessing pipeline
        - High accuracy prediction model
        """)
    
    st.markdown("""
    ### Expected Outcomes
    - A reliable prediction system for water potability
    - Reduced time and cost for water quality assessment
    - Increased awareness about water quality parameters
    """)

# Page 3: Data Collection & Understanding
elif selection == "📊 Data Collection":
    st.title("📂Data Collection & Understanding")
    
    st.markdown("""
    ## About the Dataset
    
    Our model is trained on a comprehensive water quality dataset containing over 1 million samples.
    Each sample includes measurements of 17 different parameters that affect water potability.
    """)
    
    st.header("📊Data Features")
    
    features = {
        "pH": "Measure of water acidity/alkalinity (0-14 scale)",
        "Iron": "Iron content in water (mg/L)",
        "Nitrate": "Nitrate concentration (mg/L)",
        "Chloride": "Chloride content (mg/L)",
        "Lead": "Lead concentration (mg/L)",
        "Zinc": "Zinc content (mg/L)",
        "Turbidity": "Water clarity measurement (NTU)",
        "Fluoride": "Fluoride concentration (mg/L)",
        "Copper": "Copper content (mg/L)",
        "Odor": "Odor measurement (threshold odor number)",
        "Sulfate": "Sulfate concentration (mg/L)",
        "Conductivity": "Electrical conductivity (μS/cm)",
        "Chlorine": "Residual chlorine (mg/L)",
        "Manganese": "Manganese concentration (mg/L)",
        "Total Dissolved Solids": "TDS level (mg/L)",
        "Water Temperature": "Temperature of water (°C)",
        "Air Temperature": "Ambient temperature (°C)"
    }
    
    for feature, desc in features.items():
        st.markdown(f"**{feature}**: {desc}")
    
    st.header("Data Characteristics")
    st.markdown("""
    - **Size**: 1,048,575 samples
    - **Features**: 17 water quality parameters
    - **Target**: Binary classification (0 = Not Potable, 1 = Potable)
    - **Missing Values**: None (complete dataset)
    - **Data Types**: All numerical (float64 and int64)
    """)

# Page 4: Exploratory Data Analysis
elif selection == "🔍 EDA":
    st.title("🔎Exploratory Data Analysis")
    
    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('Water_quality_cleaned.csv')
    
    df = load_data()
    
    # Section 1: Dataset Sample
    st.header("1. 📋Dataset Sample")
    st.write("First 10 rows of the dataset:")
    st.dataframe(df.head(10))
    
    # Section 2: Missing Values Analysis
    st.header("2. ❓Missing Values Analysis")
    missing_values = df.isnull().sum().to_frame(name="Missing Values")
    missing_values["Percentage"] = (missing_values["Missing Values"] / len(df)) * 100
    st.dataframe(missing_values)
    
    if missing_values["Missing Values"].sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
        ax.set_title("Missing Values Heatmap")
        st.pyplot(fig)
    else:
        st.success("✅ No missing values found in the dataset")
    
    # Section 3: Univariate Analysis
    st.header("3. Univariate Analysis")
    
    # Numerical Columns Histograms
    st.subheader("Histograms of Numerical Columns")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Select a feature for histogram", num_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)
    
    # Section 4: Outlier Detection
    st.header("4. Outlier Detection")
    st.subheader("Boxplots of Numerical Features")
    
    outlier_col = st.selectbox("Select a feature for boxplot", num_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[outlier_col], ax=ax)
    ax.set_title(f"Boxplot of {outlier_col}")
    st.pyplot(fig)
    
    
    # Section 6: Multivariate Analysis
    st.header("6. Multivariate Analysis")
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, ax=ax, mask=np.triu(corr_matrix))
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    
    
    # Section 7: Key Insights
    st.header("7. 💡Key Insights from EDA")
    
    insights = """
    **1. Data Quality:**
    - Dataset contains 1,048,575 samples with 17 features + 1 target
    - No missing values found (complete dataset)
    
    **2. Target Distribution:**
    - Slight class imbalance observed (more non-potable samples)
    
    **3. Feature Distributions:**
    - Most features show non-normal distributions
    - Several features contain outliers (especially Lead, Iron, Manganese)
    
    **4. Outlier Observations:**
    - Extreme values detected in heavy metal concentrations (Lead, Iron)
    - Some pH values at extreme ends of scale (very acidic/alkaline)
    
    **5. Correlation Findings:**
    - Moderate correlation between Conductivity and Total Dissolved Solids
    - Some negative correlation between pH and metal concentrations
    - Air and Water Temperature show expected positive correlation
    
    **6. Potability Patterns:**
    - Potable water tends to have:
      - pH closer to neutral (6.5-8.5)
      - Lower heavy metal concentrations
      - Moderate TDS and conductivity levels
    """
    
    st.markdown(insights)
# Page 5: Model Development
elif selection == "🤖 Model Development":
    st.title("🤖Model Development")
    
    st.header("⚙️Model Pipeline")
    st.markdown("""
    1. **Data Preprocessing**:
       - Robust scaling of all numerical features
       - Handles outliers effectively
    
    2. **Class Imbalance Handling**:
       - SMOTE (Synthetic Minority Oversampling Technique)
       - Creates synthetic samples of minority class
    
    3. **Model Selection**:
       - Gaussian Naive Bayes classifier
       - Tuned with var_smoothing = 6.135506425409834e-06
    """)
    
    st.header("📊Model Performance")
    st.markdown("""
    - **Accuracy**:  82.5% (on test set)
    - **Precision**: 58.6% for potable water
    - **Recall**:    80.5% for potable water
    - **F1-score**:  67.8% for both classes
    """)
    
     # ===== MODEL TRAINING AND EVALUATION =====
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    def train_and_evaluate_model():
        # Load your data (replace with your actual data loading)
        df = pd.read_csv('Water_quality_cleaned.csv')
        
        # Split features and target
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define preprocessing pipeline
        numerical_pl = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        ct = ColumnTransformer(transformers=[
            ('num', numerical_pl, X.columns)
        ])
        
        # Define complete pipeline with SMOTE and model
        final_pl = ImbPipeline(steps=[
            ('preprocessor', ct),
            ('smote', SMOTE(random_state=42)),
            ('classifier', GaussianNB(var_smoothing=6.135506425409834e-06))
        ])
        
        # Train model
        final_pl.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = final_pl.predict(X_test)
        
        return final_pl, X_test, y_test, y_pred

    # Display section
    st.header("🧠 Model Training and Evaluation")

    if st.button("Train Model"):
        with st.spinner("Training model and evaluating performance..."):
            model, X_test, y_test, y_pred = train_and_evaluate_model()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Display metrics
            st.success("Model trained successfully!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.1%}")
            col2.metric("Precision", f"{precision:.1%}")
            col3.metric("Recall", f"{recall:.1%}")
            col4.metric("F1 Score", f"{f1:.1%}")
        

# Page 6: Deployment
elif selection == "⚡ Deployment":
    st.title("🌐 Water Potability Prediction")
    
    st.markdown("""
    ## Test Water Samples
    
    Enter the water quality parameters below to get a potability prediction.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
        iron = st.number_input("Iron (mg/L)", min_value=0.0, value=0.1)
        nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, value=5.0)
        chloride = st.number_input("Chloride (mg/L)", min_value=0.0, value=100.0)
        lead = st.number_input("Lead (mg/L)", min_value=0.0, value=0.001)
        zinc = st.number_input("Zinc (mg/L)", min_value=0.0, value=0.1)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=1.0)
        fluoride = st.number_input("Fluoride (mg/L)", min_value=0.0, value=0.7)
        copper = st.number_input("Copper (mg/L)", min_value=0.0, value=0.3)
        
    with col2: 
        
        odor = st.number_input("Odor (TON)", min_value=0.0, value=1.5)
        sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=150.0)
        conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=200.0)
        chlorine = st.number_input("Chlorine (mg/L)", min_value=0.0, value=2.0)
        manganese = st.number_input("Manganese (mg/L)", min_value=0.0, value=0.01)
        tds = st.number_input("Total Dissolved Solids (mg/L)", min_value=0.0, value=300.0)
        temp = st.number_input("Water Temperature (°C)", min_value=0.0, value=15.0)
        air_temp = st.number_input("Air Temperature (°C)", min_value=-20.0, value=25.0)
    
    if st.button("Predict Potability"):
        input_data = [[
            ph, iron, nitrate, chloride, lead, zinc, turbidity, fluoride,copper, odor, sulfate, conductivity, chlorine, manganese,tds, temp, air_temp]]
        
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.success("✅ Potable Water (Safe for drinking)")
            st.balloons()
        else:
            st.error("❌ Not Potable Water (Not safe for drinking)")
        
        st.markdown(f"""
        **Confidence**: {max(proba)*100:.1f}%
        
        **Parameters Used**:
        - pH: {ph}
        - Iron: {iron} mg/L
        - Nitrate: {nitrate} mg/L
        - Chloride: {chloride} mg/L
        - Lead: {lead} mg/L
        - Zinc: {zinc} mg/L
        - Turbidity: {turbidity} NTU
        - Fluoride: {fluoride} mg/L
        - Copper: {copper} mg/L
        - Odor: {odor} TON
        - Sulfate: {sulfate} mg/L
        - Conductivity: {conductivity} μS/cm
        - Chlorine: {chlorine} mg/L
        - Manganese: {manganese} mg/L
        - TDS: {tds} mg/L
        - Water Temp: {temp} °C
        - Air Temp: {air_temp} °C
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Water Quality Prediction System**  
Developed with ❤️ using Python  
Data Science Project  
""")