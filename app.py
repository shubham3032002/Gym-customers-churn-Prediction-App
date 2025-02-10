import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load models (Lazy loading for CNN)
rf_model = joblib.load("model/radomforest_classifier.pkl")
ada_model = joblib.load("model/ada_classifier.pkl")

# Load CNN model only when needed
def load_cnn_model():
    return tf.keras.models.load_model("model/cnn_classifier.h5")

# Custom CSS for Dark Theme and Modern UI
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
        }
        .main {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #ff6600;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 12px;
            width: 100%;
            border-radius: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #e65c00;
        }
        .prediction-box {
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        .churn {
            background-color: #ff4d4d;
            color: white;
        }
        .no-churn {
            background-color: #4CAF50;
            color: white;
        }
        .sidebar {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title with Icon
st.title("🔍 Gym customers churn Prediction App")

st.markdown("""
    <div style="text-align: center;">
        <h4>🤖 Predict if a customer will churn using ML models.</h4>
        <p>🔬 Choose from Random Forest, AdaBoost, or CNN for predictions.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for Model Selection
st.sidebar.markdown('<h2 style="text-align: center;">🧠 Select Model</h2>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("", ["Random Forest", "AdaBoost", "CNN"])

# Create UI Layout
col1, col2 = st.columns(2)

with col1:
    # Gender Selection
    st.subheader("👤 Customer Info")
    gender_map = {"Male": 1, "Female": 0}
    gender_input = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender = gender_map[gender_input]

    partner = st.radio("Has a Partner?", ["Yes", "No"], horizontal=True)
    promo_friends = st.radio("Referred by Promo Friends?", ["Yes", "No"], horizontal=True)
    group_visits = st.radio("Group Visits?", ["Yes", "No"], horizontal=True)

    # Convert Yes/No to 1/0
    partner = 1 if partner == "Yes" else 0
    promo_friends = 1 if promo_friends == "Yes" else 0
    group_visits = 1 if group_visits == "Yes" else 0

with col2:
    # Numeric Inputs with Examples
    st.subheader("📊 Membership Details")
    contract_period = st.slider("Contract Period (Months)", 1, 12, 6, help="How long is the contract? (e.g., 6 months)")
    age = st.number_input("Age (Years) 🎂", min_value=15, max_value=80, value=30, step=1, help="Example: 25 years old")
    avg_additional_charges = st.number_input("Avg Additional Charges 💰", min_value=0.0, value=50.0, step=1.0, help="Example: 50 (Total additional charges)")
    month_to_end_contract = st.slider("Months to End Contract ⏳", 0, 12, 6, help="Example: 3 months left")
    lifetime = st.number_input("Lifetime Membership (Days) 📅", min_value=0, value=365, step=1, help="Example: 365 days")
    avg_class_freq_total = st.number_input("Avg Class Frequency (Total) 🏋️", min_value=0.0, value=5.0, step=0.1, help="Example: 5 sessions per month")
    avg_class_freq_current = st.number_input("Avg Class Frequency (Current Month) 📆", min_value=0.0, value=2.0, step=0.1, help="Example: 2 sessions this month")

# Predict Button with Animation
if st.button("🚀 Predict Churn"):
    features = np.array([[gender, partner, promo_friends, contract_period,
                          group_visits, age, avg_additional_charges, month_to_end_contract, lifetime,
                          avg_class_freq_total, avg_class_freq_current]])

    # Perform Prediction
    if model_choice == "Random Forest":
        prediction = rf_model.predict(features)[0]
        prob = rf_model.predict_proba(features)[0][1]  # Probability of churn
    elif model_choice == "AdaBoost":
        prediction = ada_model.predict(features)[0]
        prob = ada_model.predict_proba(features)[0][1]
    elif model_choice == "CNN":
        cnn_model = load_cnn_model()  # Load CNN only when needed
        prob = cnn_model.predict(features)[0][0]
        prediction = int(prob > 0.5)

    # Display Result with Color Styling
    result = "🚨 Churn" if prediction == 1 else "✅ No Churn"
    prob_percentage = round(prob * 100, 2)

    # Progress Bar for Confidence Score
    st.subheader("📊 Model Confidence")
    st.progress(float(prob))

    # Display Prediction
    st.markdown(f"""
        <div class="prediction-box {'churn' if prediction == 1 else 'no-churn'}">
            <h3>Prediction: {result}</h3>
            <h4>Confidence: {prob_percentage}%</h4>
        </div>
    """, unsafe_allow_html=True)
