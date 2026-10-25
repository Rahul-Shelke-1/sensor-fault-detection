import numpy as np
import os
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(ROOT_DIR / "artifacts" / "model.pkl"))
)
SAMPLE_PATH = ROOT_DIR / "data" / "sample" / "sensor_samples.csv"

TARGET_COLUMN = "class"

st.set_page_config(
    page_title="Sensor Fault Detection",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str, modified_time: float):
    """Load the trained model once and cache it."""
    del modified_time

    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as file:
            return pickle.load(file)


def get_feature_names(model):
    """Get feature names when the model exposes them."""
    candidates = [
        model,
        getattr(model, "preprocessing_object", None),
        getattr(model, "preprocessor", None),
        getattr(model, "trained_model_object", None),
        getattr(model, "model", None),
    ]

    for candidate in candidates:
        if candidate is not None:
            names = getattr(candidate, "feature_names_in_", None)
            if names is not None:
                return list(names)

    return None


def prepare_data(data, expected_features=None):
    """Validate and prepare uploaded sensor data."""
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()

    if data.empty:
        raise ValueError("The input dataset is empty.")

    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    # The target is not a model input.
    data = data.drop(columns=[TARGET_COLUMN], errors="ignore")

    # Common missing-value representation in sensor datasets.
    data = data.replace(
        r"^\s*(na|nan|none|null)\s*$",
        float("nan"),
        regex=True,
    )

    if expected_features is not None:
        missing = [
            col for col in expected_features
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required model features: {missing[:10]}"
            )

        # Preserve the exact feature order used during training.
        data = data[expected_features]
    else:
        st.warning(
            "The model does not expose its feature names. "
            "Ensure the CSV columns and their order match training."
        )

    return data


def predict(model, data):
    """Generate predictions using the loaded estimator or wrapper."""
    if not callable(getattr(model, "predict", None)):
        raise TypeError(
            "The loaded artifact must expose predict(DataFrame)."
        )

    return model.predict(data)


def display_label(value):
    """Convert common binary labels to readable output."""
    label = str(value).strip().lower()

    if label in {"1", "pos", "positive", 1.0}:
        return "pos"

    if label in {"0", "neg", "negative", 0.0}:
        return "neg"

    return str(value)


# Sidebar
with st.sidebar:
    st.title("🛠️ Sensor Fault Detection")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Load example data or upload a CSV.\n"
        "2. Preview sensor measurements.\n"
        "3. Run the model.\n"
        "4. Download prediction results."
    )
    st.markdown("---")
    st.caption("pos = predicted fault")
    st.caption("neg = no predicted fault")


# Main page
st.title("Sensor Fault Detection System")
st.write(
    "An interactive machine-learning application for predicting "
    "sensor-system failures from tabular sensor measurements."
)

with st.expander("About this project"):
    st.write(
        "This application demonstrates model inference for predictive "
        "maintenance. Predictions depend on the trained model and the "
        "quality of the supplied sensor data."
    )

# Load model
if not MODEL_PATH.is_file():
    st.error(f"Model artifact not found: {MODEL_PATH}")
    st.info(
        "Configure MODEL_PATH or place your serialized model at "
        "artifacts/model.pkl."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
    expected_features = get_feature_names(model)
    st.success("Trained model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# Input options
st.subheader("1. Select sensor data")

col1, col2 = st.columns(2)

with col1:
    if SAMPLE_PATH.is_file():
        if st.button("Load example data", use_container_width=True):
            st.session_state["sensor_data"] = pd.read_csv(SAMPLE_PATH)
            st.session_state["sensor_filename"] = SAMPLE_PATH.name
    else:
        st.info(
            "Example CSV not found. Add your sample dataset at "
            "data/sample/sensor_samples.csv."
        )

with col2:
    uploaded_file = st.file_uploader(
        "Upload sensor data (CSV)",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            st.session_state["sensor_data"] = pd.read_csv(uploaded_file)
            st.session_state["sensor_filename"] = uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")


# Preview
data = st.session_state.get("sensor_data")

if data is None:
    st.info("Load example data or upload a CSV to get started.")
    st.stop()

st.subheader("2. Preview input data")
st.caption(
    f"{len(data):,} records · {len(data.columns):,} columns"
)
st.dataframe(data.head(10), use_container_width=True)

# Inference
st.subheader("3. Run prediction")


if st.button("Predict sensor faults", type="primary"):
    try:
        features = prepare_data(data, expected_features)

        with st.spinner("Generating predictions..."):
            raw_predictions = predict(model, features)

        # Flatten predictions, including arrays shaped (n, 1).
        raw_predictions = np.asarray(raw_predictions).reshape(-1)

        if len(raw_predictions) != len(data):
            raise ValueError(
                f"Received {len(raw_predictions)} predictions "
                f"for {len(data)} input records."
            )

        labels = [display_label(value) for value in raw_predictions]

        results = data.drop(
            columns=[TARGET_COLUMN],
            errors="ignore",
        ).copy()

        results["predicted_class"] = labels
        results["prediction"] = results["predicted_class"].map(
            {
                "pos": "Fault predicted",
                "neg": "No fault predicted",
            }
        ).fillna("Other model output")

        # Store both raw and formatted predictions.
        results["raw_prediction"] = raw_predictions

        st.session_state["prediction_results"] = results

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# Results
results = st.session_state.get("prediction_results")

if results is not None:
    st.subheader("4. Prediction results")

    total = len(results)

    raw = pd.to_numeric(
        results["raw_prediction"],
        errors="coerce",
    )

    faults = int((raw == 1.0).sum())
    no_faults = int((raw == 0.0).sum())

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Total records", total)
    metric2.metric("Fault predicted (1.0)", faults)
    metric3.metric("No fault predicted (0.0)", no_faults)

    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="sensor_fault_predictions.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Demo application only. Predictions should not be used as the "
    "sole basis for safety-critical maintenance decisions."
)
