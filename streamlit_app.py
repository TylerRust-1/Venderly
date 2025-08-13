import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import json

st.set_page_config(page_title="Storecaster: Revenue Forecaster", layout="wide")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PIPELINE_PATH = "models/pipeline.joblib"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        st.error(f"Missing model file: {PIPELINE_PATH}")
        st.stop()
    return joblib.load(PIPELINE_PATH)

def map_business_type(raw):
    raw_l = str(raw).strip().lower()
    mapping = {
        "apparel": "Apparel",
        "clothing": "Apparel",
        "digital": "Digital",
        "ebook": "Digital",
        "event": "Event Services",
        "ticket": "Event Services",
        "food": "Food & Beverage",
    }
    for k, v in mapping.items():
        if k in raw_l:
            return v
    return "Apparel"

def predict_with_ci(pipe, X, n_boot=500, alpha=0.2, seed=RANDOM_SEED):
    base_pred = float(pipe.predict(X)[0])
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=1, size=n_boot)
    preds = np.empty(n_boot)
    for i in range(n_boot):
        Xb = X.copy()
        for col in Xb.select_dtypes(include=[np.number]).columns:
            if Xb[col].iloc[0] == 0:
                continue
            Xb[col].iloc[0] *= (1 + 0.02 * noise[i])
        try:
            preds[i] = float(pipe.predict(Xb)[0])
        except:
            preds[i] = base_pred
    lo = float(np.quantile(preds, alpha / 2))
    hi = float(np.quantile(preds, 1 - alpha / 2))
    return base_pred, (lo, hi)

def sidebar_inputs():
    st.sidebar.header("Inputs")
    state = st.sidebar.text_input("State (2-letter)", value="CA", max_chars=2)
    zip_code = st.sidebar.text_input("ZIP (optional)", value="")
    school_type = st.sidebar.selectbox("School type", ["Middle", "High"])
    student_pop = st.sidebar.number_input("Student population", 50, 5000, 800)
    urbanicity = st.sidebar.selectbox("Community size", ["Urban", "Suburban", "Rural"], index=1)
    pct_frl = st.sidebar.slider("% Free/Reduced Lunch", 0, 100, 40)
    median_income = st.sidebar.number_input("Median household income ($)", 10000, 250000, 70000)
    raw_type = st.sidebar.text_input("Raw product type (free text)", value="apparel")
    business_type_bucket = map_business_type(raw_type)
    months_active = st.sidebar.slider("Months active in year", 1, 12, 9)
    median_price = st.sidebar.number_input("Median item price ($)", 1.0, 500.0, 20.0)
    price_std = st.sidebar.number_input("Price std dev ($)", 0.0, 200.0, 5.0)
    store_age = st.sidebar.slider("Store age (months)", 1, 60, 12)
    cum_rev_first3 = st.sidebar.number_input("Cumulative revenue first 3 months ($)", 0.0, 500000.0, 3000.0)
    params = dict(
        state=state.upper(),
        zip=str(zip_code).strip(),
        school_type=school_type,
        student_population=int(student_pop),
        urbanicity=urbanicity,
        pct_free_lunch=float(pct_frl) / 100,
        median_income=float(median_income),
        business_type_bucket=business_type_bucket,
        store_age_months=int(store_age),
        months_active=int(months_active),
        median_price=float(median_price),
        price_std=float(price_std),
        cum_revenue_first_3m=float(cum_rev_first3),
    )
    return params

def to_model_frame(params, pipe):
    base = pd.DataFrame([params])
    # Align columns to pipeline if possible (optional)
    try:
        ct = None
        for step in getattr(pipe, "steps", []):
            name, obj = step
            from sklearn.compose import ColumnTransformer
            if isinstance(obj, ColumnTransformer):
                ct = obj
                break
        if ct:
            wanted = []
            for _, _, cols in ct.transformers:
                if cols == "drop":
                    continue
                if isinstance(cols, (list, tuple)):
                    wanted.extend(cols)
            for col in wanted:
                if col not in base.columns:
                    base[col] = np.nan
            base = base[wanted].copy()
    except:
        pass
    return base

def main():
    st.title("Storecaster: Revenue Forecaster")
    st.caption("Predict annual revenue for student-run ventures by community and business context.")

    pipe = load_pipeline()

    tabs = st.tabs(["Predict", "Compare business types"])

    with tabs[0]:
        params = sidebar_inputs()
        X = to_model_frame(params, pipe)
        if st.button("Predict annual revenue"):
            y_pred, (lo, hi) = predict_with_ci(pipe, X)
            st.metric("Estimated revenue", f"${y_pred:,.0f}")
            st.metric("80% low", f"${lo:,.0f}")
            st.metric("80% high", f"${hi:,.0f}")

    with tabs[1]:
        st.subheader("Compare multiple business types")
        base_params = sidebar_inputs()
        types = ["Apparel", "Digital", "Event Services", "Food & Beverage", "Accessories"]
        compare_types = st.multiselect("Business types", types, default=types[:3])
        if st.button("Run comparison"):
            results = []
            for bt in compare_types:
                p = {**base_params, "business_type_bucket": bt}
                Xb = to_model_frame(p, pipe)
                mu, (lo, hi) = predict_with_ci(pipe, Xb, n_boot=300)
                results.append({"business_type": bt, "pred": mu, "lo": lo, "hi": hi})
            df = pd.DataFrame(results).sort_values("pred", ascending=False)
            st.dataframe(df)

if __name__ == "__main__":
    main()
