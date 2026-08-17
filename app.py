import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .fraud-box {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        background-color: #ffdddd;
        border: 2px solid #ff0000;
    }

    .safe-box {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        background-color: #ddffdd;
        border: 2px solid #00aa00;
    }

    .risk-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #999999;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🔐 Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Financial Transaction Fraud Detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

MODEL_PATH = "fraud_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "fraud_model.pkl was not found."
    )

    st.info(
        "Run train_model.py first to train and save the model."
    )

    st.stop()

try:

    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]

except Exception as e:

    st.error(
        f"Unable to load the model: {e}"
    )

    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("⚙️ System Information")

st.sidebar.success("Model loaded successfully")

st.sidebar.write(
    "Machine Learning Model:"
)

st.sidebar.write(
    "Random Forest Classifier"
)

st.sidebar.write(
    f"Number of features: {len(features)}"
)

st.sidebar.divider()

st.sidebar.info(
    """
    Enter the transaction feature values
    and click the prediction button.
    """
)

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.header("💳 Transaction Details")

st.write(
    "Enter values for the transaction features below."
)

# ---------------------------------------------------------
# CREATE INPUT DICTIONARY
# ---------------------------------------------------------

input_values = {}

# Create two columns
col1, col2 = st.columns(2)

# ---------------------------------------------------------
# INPUT FIELDS
# ---------------------------------------------------------

for index, feature in enumerate(features):

    # Skip if unusual target column somehow appears
    if feature == "Class":
        continue

    if index % 2 == 0:

        with col1:

            input_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                format="%.6f"
            )

    else:

        with col2:

            input_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                format="%.6f"
            )

# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "🔍 CHECK TRANSACTION",
    use_container_width=True,
    type="primary"
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    try:

        # Create dataframe
        input_df = pd.DataFrame(
            [input_values]
        )

        # Ensure correct feature order
        input_df = input_df[features]

        # Scale input
        input_scaled = scaler.transform(
            input_df
        )

        # Prediction
        prediction = model.predict(
            input_scaled
        )[0]

        # Probability
        probability = model.predict_proba(
            input_scaled
        )[0]

        fraud_probability = probability[1] * 100

        genuine_probability = probability[0] * 100

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")

        # FRAUD
        if prediction == 1:

            st.markdown(
                f"""
                <div class="fraud-box">

                <h1>🚨 FRAUDULENT TRANSACTION</h1>

                <h2>Fraud Probability: {fraud_probability:.2f}%</h2>

                <p>
                The machine learning model considers
                this transaction suspicious.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # GENUINE
        else:

            st.markdown(
                f"""
                <div class="safe-box">

                <h1>✅ GENUINE TRANSACTION</h1>

                <h2>Fraud Probability: {fraud_probability:.2f}%</h2>

                <p>
                The machine learning model considers
                this transaction legitimate.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # PROBABILITY METRICS
        # -------------------------------------------------

        st.subheader("Probability Analysis")

        metric1, metric2 = st.columns(2)

        with metric1:

            st.metric(
                "Genuine Probability",
                f"{genuine_probability:.2f}%"
            )

        with metric2:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

        # -------------------------------------------------
        # PROGRESS BAR
        # -------------------------------------------------

        st.subheader("Fraud Risk")

        risk_value = fraud_probability / 100

        st.progress(
            min(max(risk_value, 0.0), 1.0)
        )

        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if fraud_probability < 20:

            risk_level = "LOW RISK"
            st.success(
                f"Risk Level: {risk_level}"
            )

        elif fraud_probability < 50:

            risk_level = "MEDIUM RISK"
            st.warning(
                f"Risk Level: {risk_level}"
            )

        else:

            risk_level = "HIGH RISK"
            st.error(
                f"Risk Level: {risk_level}"
            )

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        with st.expander("View Transaction Data"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Fraud Detection System | "
    "Machine Learning Project"
)