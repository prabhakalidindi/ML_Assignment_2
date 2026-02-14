import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Page Configuration
st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

st.title("🌱 Dry Bean Variety Classifier")
st.markdown("M.Tech (AIML) Assignment 2 | **Predicting 7 different bean types**")

# --- SIDEBAR ---
st.sidebar.header("1. Configuration")
model_name = st.sidebar.selectbox("Select Model", 
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"))

# --- DOWNLOAD BUTTON FOR SAMPLE DATA ---
st.sidebar.header("2. Test Data")
st.sidebar.markdown("Don't have data? Download the sample dataset used for evaluation:")

# Helper to load the CSV for the download button
@st.cache_data
def load_test_data():
    # Reads 'sample_test_data.csv' from the local directory
    return pd.read_csv("sample_test_data.csv").to_csv(index=False).encode('utf-8')

try:
    csv_data = load_test_data()
    st.sidebar.download_button(
        label="📥 Download sample_test_data.csv",
        data=csv_data,
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Click to download the sample dataset to verify the model."
    )
except FileNotFoundError:
    st.sidebar.warning("sample_test_data.csv not found in repo.")

# --- FILE UPLOAD ---
st.sidebar.header("3. User Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# --- MAIN APP LOGIC ---
try:
    # Load Assets
    model = joblib.load(f'model/{model_name.replace(" ", "_")}.pkl')
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')
except Exception as e:
