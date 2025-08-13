import streamlit as st
import torch
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder


# Load model and encoder
model = torch.load("model.pth", weights_only=False)  # ⚠ Only if trusted file
model.eval()
encoder = OneHotEncoder(sparse_output=False)

# Define UI
st.title("School-Business Success Predictor")
school = st.selectbox("Select School Type", ["Middle School", "High School"])
btype = st.selectbox("Select Business Type", ["Retail", "Food Service", "Tutoring", "Other"])

# Predict button
if st.button("Predict"):
    # Encode input
    ex_cat = encoder.transform([[school, btype]])  # categorical only
    ex_tensor = torch.tensor(ex_cat, dtype=torch.float32)

    # Run model
    rev_pred, surv_pred = model(ex_tensor)
    
    # Rescale revenue (adjust if min/max saved somewhere)
    rev_min, rev_max = joblib.load("rev_min_max.pkl")
    rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min

    # Output results
    st.write(f"**Predicted Annual Revenue:** ${rev_rescaled:,.2f}")
    st.write(f"**Survival ≥1 month:** {surv_pred[0,0].item()*100:.1f}%")
    st.write(f"**Survival ≥3 months:** {surv_pred[0,1].item()*100:.1f}%")
    st.write(f"**Survival ≥1 year:** {surv_pred[0,2].item()*100:.1f}%")
