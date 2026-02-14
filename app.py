import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("🌱 Dry Bean Variety Classifier")
st.markdown("M.Tech (AIML) Assignment 2 | **Predicting 7 different bean types based on shape**")

# Sidebar
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"))

# Load Assets
try:
    model = joblib.load(f'model/{model_name.replace(" ", "_")}.pkl')
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')
except:
    st.error("Model files missing. Run train_models.py first.")
    st.stop()

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. Data Preview")
    st.dataframe(df.head(3))

    # Prepare Data
    if 'Class' in df.columns:
        y_true_names = df['Class']
        try:
            y_true = le.transform(y_true_names) # Convert string to int
        except:
             st.warning("Unknown labels in uploaded file. Skipping evaluation.")
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
        pred_names = le.inverse_transform(preds) # Convert int back to string

        # Metrics
        if y_true is not None:
            st.write(f"### 2. Evaluation Metrics ({model_name})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy_score(y_true, preds):.4f}")
            c2.metric("Precision (W)", f"{precision_score(y_true, preds, average='weighted'):.4f}")
            c3.metric("Recall (W)", f"{recall_score(y_true, preds, average='weighted'):.4f}")
            c4.metric("F1 Score (W)", f"{f1_score(y_true, preds, average='weighted'):.4f}")

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
    st.info("👋 Upload 'sample_test_data.csv' to begin.")