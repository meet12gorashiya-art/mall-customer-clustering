

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import load_data
from src.exceptions import DataLoadError, InvalidInputError, ModelNotFoundError
from src.predict import describe_segment, load_model, predict_segment
from src.preprocessing import select_features
from src.train import DEFAULT_MODEL_PATH, run_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="centered")


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained clustering model, training it once if it doesn't exist yet."""
    try:
        return load_model(DEFAULT_MODEL_PATH)
    except ModelNotFoundError:
        logger.info("No cached model found - training a fresh one.")
        with st.spinner("First run: finding the optimal number of clusters, this takes a moment..."):
            run_pipeline(DATA_PATH, DEFAULT_MODEL_PATH)
        return load_model(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_customer_data():
    return load_data(DATA_PATH)


def main():
    st.title("🛍️ Mall Customer Segmentation")
    st.write(
        "Segment mall customers by annual income and spending score using "
        "KMeans clustering, and see which segment a new customer falls into."
    )

    try:
        model = get_model()
        df = get_customer_data()
    except (DataLoadError, ModelNotFoundError) as exc:
        st.error(f"Could not load or train the model: {exc}")
        logger.exception("Fatal startup error")
        st.stop()

    features = select_features(df)
    df = df.copy()
    df["Cluster"] = model.predict(features)

    st.subheader("Customer segments")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df["Annual_Income"], df["Spending_Score"], c=df["Cluster"], cmap="tab10", alpha=0.7
    )
    centers = model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.6, marker="X", label="Centers")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Classify a new customer")
    with st.form("prediction_form"):
        income = st.number_input("Annual income (k$)", min_value=0, max_value=200, value=60)
        spending = st.number_input("Spending score (1-100)", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button("Assign segment")

    if submitted:
        customer = {"Annual_Income": income, "Spending_Score": spending}
        try:
            result = predict_segment(model, customer)
            st.success(
                f"This customer falls into **Cluster {result['cluster']}** "
                f"({result['description']})"
            )
        except InvalidInputError as exc:
            st.error(f"Invalid input: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected prediction error")
            st.error(f"Something went wrong while predicting: {exc}")

    with st.expander("About this app"):
        st.markdown(
            "Trained on the `mall_customers.csv` dataset using scikit-learn KMeans. "
            "The number of clusters is chosen automatically by comparing silhouette "
            "scores across k=3..8, matching the analysis in the original notebook "
            "(which found k=5 optimal on Annual Income and Spending Score). Built for "
            "the CST2216 Machine Learning 2 term project."
        )


if __name__ == "__main__":
    main()
