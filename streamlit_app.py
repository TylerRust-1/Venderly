import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from MultiTaskModel import MultiTaskModel

# Load saved encoder and model
encoder = joblib.load("encoder.pkl")

# Input dimension from encoder + numeric feature
input_dim = encoder.transform([["middle", "club fundraiser"]]).shape[1] + 1
model = MultiTaskModel(input_dim=input_dim)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load revenue scaling
rev_min, rev_max = joblib.load("rev_min_max.pkl")

st.title("School-Business Revenue & Survival Predictor")

school_level = st.selectbox("Select School Type", ["middle", "high"])
business_type = st.selectbox("Select Business Type", [
    "Club Fundraiser", "School Store & Snack Shop", "Concessions",
    "School Store", "Culinary Shop", "Plant & Flower Fundraiser",
    "Prom & Homecoming tickets"
])
avg_operating_time = st.number_input("Expected Operating Time (days)", min_value=1, value=180)

if st.button("Predict"):
    # Encode categorical inputs
    cat_input = encoder.transform([[school_level, business_type.lower()]])
    num_input = np.array([[avg_operating_time]])
    x_input = np.hstack([cat_input, num_input])
    x_tensor = torch.tensor(x_input, dtype=torch.float32)

    # Run model
    with torch.no_grad():
        rev_pred, surv_pred = model(x_tensor)
        rev_rescaled = rev_pred.item() * (rev_max - rev_min) + rev_min

    st.write(f"**Predicted Annual Revenue:** ${rev_rescaled:,.2f}")
    st.write(f"**Survival ≥1 month:** {surv_pred[0,0].item()*100:.1f}%")
    st.write(f"**Survival ≥3 months:** {surv_pred[0,1].item()*100:.1f}%")
    st.write(f"**Survival ≥1 year:** {surv_pred[0,2].item()*100:.1f}%")
