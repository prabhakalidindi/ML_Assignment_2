import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    matthews_corrcoef
)

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
    try:
        return pd.read_csv("sample_test_data.csv").to_csv(index=False).encode('utf-8')
    except FileNotFoundError:
        return None

csv_data = load_test_data()
if csv_data:
    st.sidebar.download_button(
        label="📥 Download sample_test_data.csv",
        data=csv_data,
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Click to download the sample dataset to verify the model."
    )
else:
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
    st.error(f"Error loading models. Make sure .pkl files are uploaded. Details: {e}")
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. Data Preview")
    st.dataframe(df.head(3))
    
    # Preprocessing
    if 'Class' in df.columns:
        y_true_names = df['Class']
        try:
            y_true = le.transform(y_true_names)
        except:
             st.warning("Unknown labels in uploaded file. Skipping evaluation metrics.")
             y_true = None
        X_input = df.drop(columns=['Class'])
    else:
        y_true = None
        X_input = df
        
    # Scale if needed
    if model_name in ["Logistic Regression", "KNN"]:
        try:
            X_processed = scaler.transform(X_input)
        except:
             st.error("Feature mismatch! Please ensure columns match training data.")
             st.stop()
    else:
        X_processed = X_input

    # Predict
    if st.button("Run Classification"):
        preds = model.predict(X_processed)
        probs = model.predict_proba(X_processed)  # Required for AUC
        pred_names = le.inverse_transform(preds)
        
        # Metrics Display [Requirement: Display all 6 evaluation metrics]
        if y_true is not None:
            st.write(f"### 2. Evaluation Metrics ({model_name})")
            
            # Row 1
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy_score(y_true, preds):.4f}")
            c2.metric("Precision (W)", f"{precision_score(y_true, preds, average='weighted'):.4f}")
            c3.metric("Recall (W)", f"{recall_score(y_true, preds, average='weighted'):.4f}")
            
            # Row 2
            c4, c5, c6 = st.columns(3)
            c4.metric("F1 Score (W)", f"{f1_score(y_true, preds, average='weighted'):.4f}")
            # AUC requires multi_class='ovr' for multiclass problems
            try:
                auc = roc_auc_score(y_true, probs, multi_class='ovr', average='weighted')
                c5.metric("AUC Score", f"{auc:.4f}")
            except:
                c5.metric("AUC Score", "N/A")
            
            c6.metric("MCC Score", f"{matthews_corrcoef(y_true, preds):.4f}")
            
            # Confusion Matrix
            st.write("### 3. Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 5))
            cm = confusion_matrix(y_true, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            
        # Results
        st.write("### 4. Prediction Results")
        res_df = X_input.copy()
        res_df['Predicted_Bean'] = pred_names
        st.dataframe(res_df)
else:
    st.info("👋 Upload a CSV file or download `sample_test_data.csv` from the sidebar to begin.")
