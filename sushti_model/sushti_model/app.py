import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
import os

# Path helper
BASE_DIR = os.path.dirname(__file__)

# Set page config
st.set_page_config(page_title="Electricity Saving Assistant", page_icon="⚡", layout="centered")

# Load model and encoders
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "appliance_encoder.pkl"))

# Load appliance tips
with open(os.path.join(BASE_DIR, "tips.json"), 'r') as f:
    tips_db = json.load(f)

st.title("Electricity Saving Assistant ⚡")

# Monthly electricity input
units = st.slider("Select your monthly electricity consumption (kWh):", 0, 2000, 200, 100)

# Appliance selection
appliances_list = list(tips_db.keys())
selected_appliances = st.multiselect("Select the appliances you use:", appliances_list)

# Daily usage per appliance
usage_times = {}
if selected_appliances:
    st.subheader("🕒 Daily Usage Time per Appliance")
    for appliance in selected_appliances:
        usage_times[appliance] = st.slider(f"{appliance} (hours/day)", 0, 24, 1)

# Upload past bills
st.subheader("📂 Upload Past Monthly Bills (Optional)")
bill_df = None
uploaded_file = st.file_uploader("Upload your past monthly electricity bills (CSV with 'Month' and 'Units' columns):", type=["csv"])
if uploaded_file:
    bill_df = pd.read_csv(uploaded_file)
    if 'Month' in bill_df.columns and 'Units' in bill_df.columns:
        st.line_chart(data=bill_df.set_index('Month'))
    else:
        st.warning("CSV must contain 'Month' and 'Units' columns")

if st.button("Get Energy Saving Tips"):
    if not selected_appliances or units == 0:
        st.warning("Please provide both electricity usage and appliances used.")
    else:
        user_input = mlb.transform([selected_appliances])
        user_df = pd.DataFrame(user_input, columns=mlb.classes_)

        # Align features
        model_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else model._fit_X.shape[1]
        if user_df.shape[1] + 1 > model_features:
            user_df = user_df.iloc[:, :model_features - 1]
        user_df["Monthly_Consumption_kWh"] = units

        try:
            dist, indices = model.kneighbors(user_df)
        except Exception as e:
            st.error(f"Error in model prediction: {e}")
            st.stop()

        # Load dataset
        df = pd.read_excel(os.path.join(BASE_DIR, "EVS Dataset.xlsx"), sheet_name="Sheet1")
        df["Appliances"] = df["Appliances"].apply(lambda x: [a.strip() for a in str(x).split(",")])

        st.subheader("💡 Recommended Tips:")
        shown = set()
        for idx in indices[0]:
            matched_appliances = df.loc[idx, "Appliances"]
            for appliance in matched_appliances:
                if appliance in tips_db and appliance not in shown:
                    tips = tips_db[appliance]
                    if isinstance(tips, list):
                        for tip in tips:
                            st.markdown(f"- **{appliance}**: {tip}")
                    else:
                        st.markdown(f"- **{appliance}**: {tips}")
                    shown.add(appliance)

        # Smart Suggestion
        st.subheader("🤖 Smart Suggestion")
        if "Monthly_Consumption_kWh" in df.columns:
            high_consumers = df[df['Monthly_Consumption_kWh'] > units]
            if not high_consumers.empty:
                common = pd.Series([a for sublist in high_consumers['Appliances'] for a in sublist]).value_counts().head(3)
                for app in common.index:
                    st.info(f"Consider reducing usage of: **{app}**")

        # Estimated Monthly Bill & Savings
        st.subheader("📉 Estimated Monthly Bill & Savings")
        base_bill = units * 8
        saving_percent = sum(random.randint(10, 30) for _ in selected_appliances) / (len(selected_appliances) or 1)
        saved_units = units * (saving_percent / 100)
        new_units = units - saved_units
        new_bill = new_units * 8

        st.markdown(f"**Original Units**: {units} kWh")
        st.markdown(f"**Estimated Savings**: {int(saved_units)} kWh (~{int(saving_percent)}%)")
        st.markdown(f"**New Estimated Units**: {int(new_units)} kWh")
        st.markdown(f"**Original Bill**: ₹{int(base_bill)}")
        st.markdown(f"**New Estimated Bill**: ₹{int(new_bill)}")
        st.markdown(f"**💰 You Save**: ₹{int(base_bill - new_bill)}")

        # Appliance Usage Distribution
        st.subheader("📊 Appliance Usage Distribution")
        usage_kwh = []
        total_hours = sum(usage_times.values()) or 1
        for appliance in selected_appliances:
            hours = usage_times.get(appliance, 1)
            share = (hours / total_hours) * units
            usage_kwh.append(share)

        usage_df = pd.DataFrame({"Appliance": selected_appliances, "Estimated kWh": usage_kwh})
        st.bar_chart(usage_df.set_index("Appliance"))
