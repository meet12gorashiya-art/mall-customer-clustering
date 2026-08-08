# Mall Customer Segmentation

Segments mall customers into groups by annual income and spending score
using KMeans clustering, and classifies new customers into the segment
they most resemble.


## How it works

1. **Data loading** (`src/data_loader.py`) reads `data/mall_customers.csv`
   and validates that all expected columns are present.
2. **Feature selection & k evaluation** (`src/preprocessing.py`) selects
   `Annual_Income` and `Spending_Score` (matching the original notebook)
   and evaluates candidate cluster counts k=3..8 using both the elbow
   method (within-cluster sum of squares) and the silhouette score.
3. **Training** (`src/train.py`) automatically picks the k with the
   highest silhouette score, fits the final `KMeans` model with
   `init='k-means++'`, and saves it to `models/cluster_model.pkl`.
4. **Prediction** (`src/predict.py`) loads the saved model, assigns a new
   customer (income + spending score) to a cluster, and generates a
   human-readable description (e.g. "High income, low spending") from
   that cluster's center.
5. **App** (`app.py`) shows a scatter plot of all customer segments with
   cluster centers marked, plus a form to classify a new customer. On
   first run, if no trained model is found, it trains one automatically.

All modules log to both stdout and a rotating file (`logs/app.log`) via a
shared logger in `src/utils/logger.py`, and raise typed exceptions
(`src/exceptions.py`) instead of failing silently.

## Setup

```bash
cd mall-customer-clustering
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model from the command line:

```bash
python -m src.train
```

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Run the test suite:

```bash
pytest
```

## Deployment

Deployed on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app pointing at `app.py` on the `main`
   branch.
3. Streamlit Cloud installs `requirements.txt` automatically and runs
   `streamlit run app.py`.

**Live app:** https://mall-customer-clustering-mwuk4jmehup5gerizmdxh6.streamlit.app/

## Dataset

`data/mall_customers.csv` — 200 mall customers with ID, gender, age,
annual income (k$), and a spending score (1-100) assigned by the mall
based on customer behavior.


