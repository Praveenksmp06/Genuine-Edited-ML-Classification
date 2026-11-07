from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Page configuration

st.set_page_config(
    page_title="Edited vs Genuine Image Classification",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results" / "model_comparison.csv"

# Load trained models

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression_model.pkl"
        ),

        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree_model.pkl"
        ),

        "K-Nearest Neighbors": joblib.load(
            MODEL_DIR / "k_nearest_neighbors_model.pkl"
        ),

        "Gaussian Naive Bayes": joblib.load(
            MODEL_DIR / "gaussian_nb_model.pkl"
        ),

        "Support Vector Machine": joblib.load(
            MODEL_DIR / "svm_model.pkl"
        )
    }

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    return models, scaler

try:
    models, scaler = load_models()
except (FileNotFoundError, OSError, EOFError, ImportError, ValueError) as error:
    st.error(f"Unable to load trained models: {error}")
    st.stop()

# Application heading

st.title("Edited vs Genuine Image Classification")

st.write(
    "This application evaluates image authenticity using "
    "numerical image features and five machine learning algorithms."
)

st.divider()

# Dataset upload

st.subheader("Upload Test Dataset")

st.write(
    "Upload a labelled CSV file containing the 22 extracted "
    "image features and the target label."
)

uploaded_file = st.file_uploader(
    "Choose a test CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your test dataset to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
    st.error(f"Unable to read the uploaded file: {error}")
    st.stop()

# Validate dataset

if "label" not in df.columns:
    st.error("The CSV must contain a label column.")
    st.stop()

X = df.drop(
    columns=["image_name", "label"],
    errors="ignore"
)

y = pd.to_numeric(df["label"], errors="coerce")

if X.shape[1] != 22:
    st.error("The dataset must contain exactly 22 image features.")
    st.stop()

X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any() or y.isnull().any():
    st.error("Please check the dataset for missing or invalid values.")
    st.stop()

if not y.isin([0, 1]).all():
    st.error("Labels must be 0 (Genuine) or 1 (Edited).")
    st.stop()

if y.nunique() != 2:
    st.error("The dataset must contain both Genuine and Edited images.")
    st.stop()

# Match the feature order used during training

if hasattr(scaler, "feature_names_in_"):
    X = X[list(scaler.feature_names_in_)]

# Dataset information

st.success(
    f"Dataset loaded: {len(df):,} records and {X.shape[1]} features."
)

with st.expander("Preview uploaded dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total records", f"{len(df):,}")
    col2.metric("Genuine images", f"{(y == 0).sum():,}")
    col3.metric("Edited images", f"{(y == 1).sum():,}")

# Model selection

st.subheader("Select Classification Model")

selected_model = st.selectbox(
    "Choose a machine learning model",
    list(models.keys())
)

st.write("Selected model:", selected_model)

# Prediction

try:

    X_scaled = scaler.transform(X)

    model = models[selected_model]

    predictions = model.predict(X_scaled)

    probabilities = model.predict_proba(X_scaled)

    class_index = list(model.classes_).index(1)

    edited_probabilities = probabilities[:, class_index]

except (ValueError, TypeError, AttributeError, IndexError) as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

# Evaluation metrics

accuracy = accuracy_score(y, predictions)

auc = roc_auc_score(
    y,
    edited_probabilities
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

mcc = matthews_corrcoef(
    y,
    predictions
)

# Display evaluation results

st.subheader("Model Evaluation Results")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

# Prediction summary

st.subheader("Prediction Summary")

predicted_genuine = int((predictions == 0).sum())
predicted_edited = int((predictions == 1).sum())

col1, col2 = st.columns(2)

col1.metric(
    "Predicted Genuine",
    f"{predicted_genuine:,}"
)

col2.metric(
    "Predicted Edited",
    f"{predicted_edited:,}"
)

st.caption(
    "These counts represent the model's predictions, "
    "not the actual class distribution of the dataset."
)

# Confusion matrix

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y,
    predictions,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Genuine", "Actual Edited"],
    columns=["Predicted Genuine", "Predicted Edited"]
)

st.dataframe(
    cm_df.style.background_gradient(
        cmap="Blues"
    ).format("{:,.0f}"),
    use_container_width=True
)

# Classification report

st.subheader("Classification Report")

report = classification_report(
    y,
    predictions,
    labels=[0, 1],
    target_names=["Genuine", "Edited"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# Comparison of all five models

st.subheader("Comparison of All ML Models")

if RESULTS_PATH.exists():

    results = pd.read_csv(RESULTS_PATH)

    results = results.rename(
        columns={
            "Model": "ML Model",
            "F1-score": "F1 Score",
            "ROC-AUC": "AUC"
        }
    )

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    best_model = results.loc[
        results["Accuracy"].idxmax()
    ]

    st.markdown("**Best-performing model based on accuracy**")

    st.success(
        f"{best_model['ML Model']} achieved an accuracy "
        f"of {best_model['Accuracy']:.4f}."
    )

    st.write(
        f"F1 Score: {best_model['F1 Score']:.4f} | "
        f"MCC: {best_model['MCC']:.4f}"
    )

    st.markdown("**Accuracy Comparison**")

    chart_data = results.sort_values(
        by="Accuracy",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.barh(
        chart_data["ML Model"],
        chart_data["Accuracy"],
        color="#52789D",
        height=0.55
    )

    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy of Machine Learning Models")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar in bars:

        value = bar.get_width()

        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2%}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

else:

    st.warning(
        "Model comparison file was not found. "
        "Check results/model_comparison.csv."
    )

# Footer

st.divider()
st.caption(
    "Edited vs Genuine Image Classification"
)