from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("logistic_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run the notebook first to create it."
        )
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("Credit Card Fraud Detection")
st.write(
    "Estimate whether a transaction resembles a fraudulent transaction using the "
    "trained logistic regression model."
)
st.warning(
    "This is a demonstration using an anonymized/sample dataset. Do not use its "
    "predictions to make real financial decisions."
)

model = load_model()
feature_names = list(model.feature_names_in_)
fraud_class_index = list(model.classes_).index(1)

mode = st.radio(
    "Choose input type",
    ("Single transaction", "Upload CSV"),
    horizontal=True,
)

if mode == "Single transaction":
    with st.form("transaction_form"):
        st.subheader("Transaction details")
        basic_columns = st.columns(2)
        values: dict[str, float] = {}
        for index, feature in enumerate(("Time", "Amount")):
            with basic_columns[index]:
                values[feature] = st.number_input(
                    feature,
                    value=1.0 if feature == "Amount" else 0.0,
                    format="%.6f",
                )

        with st.expander("Anonymized transaction features (V1–V28)"):
            feature_columns = st.columns(4)
            for index, feature in enumerate(
                name for name in feature_names if name not in {"Time", "Amount"}
            ):
                with feature_columns[index % len(feature_columns)]:
                    values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.6f",
                    )

        submitted = st.form_submit_button("Analyze transaction", type="primary")

    if submitted:
        transaction = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
        fraud_probability = model.predict_proba(transaction)[0][fraud_class_index]
        if fraud_probability >= 0.5:
            st.error(f"Prediction: Potential fraud — {fraud_probability:.1%} estimated probability")
        else:
            st.success(f"Prediction: Likely legitimate — {fraud_probability:.1%} estimated fraud probability")
        st.progress(float(fraud_probability), text="Estimated fraud probability")
else:
    uploaded_file = st.file_uploader("Upload a CSV with the model's input columns", type="csv")
    if uploaded_file is not None:
        transactions = pd.read_csv(uploaded_file)
        missing_features = [name for name in feature_names if name not in transactions.columns]
        if missing_features:
            st.error(
                "CSV is missing required model columns: " + ", ".join(missing_features)
            )
        elif transactions.empty:
            st.error("The uploaded CSV contains no transactions.")
        else:
            model_inputs = transactions.loc[:, feature_names]
            probabilities = model.predict_proba(model_inputs)[:, fraud_class_index]
            predictions = (probabilities >= 0.5).astype(int)

            results = transactions.copy()
            results["Fraud probability"] = probabilities
            results["Prediction"] = [
                "Potential fraud" if prediction else "Likely legitimate"
                for prediction in predictions
            ]
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
