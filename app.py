import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food Waste Prediction & Demand Forecasting", page_icon="🍽️", layout="wide")
st.title("🍽️ Food Waste Prediction & Demand Forecasting")
st.caption("Machine-learning prototype using the public synthetic dataset linked below.")

DATA_URL = "https://raw.githubusercontent.com/kalyanipatange/Waste-Food-Management-Analysis-/main/food_waste_dataset.csv"

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        raise RuntimeError("Could not fetch the CSV automatically. Download food_waste_dataset.csv from the source repository and upload it in the sidebar.") from e

@st.cache_resource(show_spinner=False)
def train_models(df):
    data = df.copy()
    required = ["Date", "Daily_Demand", "Waste_Quantity", "Waste_Level"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    feature_cols = [c for c in ["Business_Type", "Food_Category", "Purchase_Quantity", "Shelf_Life_Days",
                                 "Storage_Temperature", "Promotion", "Weather", "Date"] if c in data.columns]
    # Date-derived features are safe for forecasting; don't use same-day realized sales/waste to predict same-day demand.
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["DayOfYear"] = data["Date"].dt.dayofyear
    feature_cols = [c for c in feature_cols if c != "Date"] + ["DayOfWeek", "Month", "DayOfYear"]
    # Sort chronologically and hold out the latest 20%.
    split = max(1, int(len(data) * 0.8))
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()
    # Explicitly mark known text fields as categorical. Some CSV readers infer
    # a category/text field such as Business_Type as a non-object dtype.
    categorical_names = {"Business_Type", "Food_Category", "Promotion", "Weather"}
    cats = [c for c in feature_cols if c in categorical_names]
    nums = [c for c in feature_cols if c not in cats]
    # Keep numeric inputs numeric even if the CSV stores them as text.
    for c in nums:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")
    for c in cats:
        train[c] = train[c].astype("string").fillna("Unknown")
        test[c] = test[c].astype("string").fillna("Unknown")
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), cats)
    ])
    demand_model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1))])
    waste_model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1))])
    clf_model = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", n_jobs=-1))])
    # Drop rows with missing target only, impute feature missingness.
    dm_train = train.dropna(subset=["Daily_Demand"])
    dm_test = test.dropna(subset=["Daily_Demand"])
    wm_train = train.dropna(subset=["Waste_Quantity"])
    wm_test = test.dropna(subset=["Waste_Quantity"])
    cl_train = train.dropna(subset=["Waste_Level"])
    cl_test = test.dropna(subset=["Waste_Level"])
    demand_model.fit(dm_train[feature_cols], dm_train["Daily_Demand"])
    waste_model.fit(wm_train[feature_cols], wm_train["Waste_Quantity"])
    clf_model.fit(cl_train[feature_cols], cl_train["Waste_Level"].astype(str))
    metrics = {}
    if len(dm_test):
        p = demand_model.predict(dm_test[feature_cols])
        metrics["demand"] = {"MAE": float(mean_absolute_error(dm_test["Daily_Demand"], p)),
                             "RMSE": float(np.sqrt(mean_squared_error(dm_test["Daily_Demand"], p))),
                             "R2": float(r2_score(dm_test["Daily_Demand"], p)) if len(dm_test) > 1 else None}
    if len(wm_test):
        p = np.maximum(0, waste_model.predict(wm_test[feature_cols]))
        metrics["waste"] = {"MAE": float(mean_absolute_error(wm_test["Waste_Quantity"], p)),
                            "RMSE": float(np.sqrt(mean_squared_error(wm_test["Waste_Quantity"], p))),
                            "R2": float(r2_score(wm_test["Waste_Quantity"], p)) if len(wm_test) > 1 else None}
    if len(cl_test):
        p = clf_model.predict(cl_test[feature_cols])
        metrics["risk"] = {"accuracy": float(accuracy_score(cl_test["Waste_Level"].astype(str), p)),
                           "macro_f1": float(f1_score(cl_test["Waste_Level"].astype(str), p, average="macro"))}
    return data, feature_cols, demand_model, waste_model, clf_model, metrics, test

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload food_waste_dataset.csv if auto-load fails", type=["csv"])
try:
    df = load_data(uploaded)
    data, features, demand_model, waste_model, risk_model, metrics, test = train_models(df)
except Exception as e:
    st.error(str(e))
    st.markdown(f"Source repository: [{DATA_URL}]({DATA_URL})")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Records loaded", f"{len(data):,}")
c2.metric("Food categories", str(data["Food_Category"].nunique()) if "Food_Category" in data else "—")
c3.metric("Total waste quantity", f"{pd.to_numeric(data['Waste_Quantity'], errors='coerce').sum():,.0f}")
c4.metric("Average daily demand", f"{pd.to_numeric(data['Daily_Demand'], errors='coerce').mean():.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict", "Model evaluation", "Dataset"])
with tab1:
    st.subheader("Waste quantity by food category")
    if "Food_Category" in data:
        grouped = data.groupby("Food_Category", dropna=False)["Waste_Quantity"].sum().sort_values(ascending=False)
        st.bar_chart(grouped)
    st.subheader("Waste-risk distribution")
    if "Waste_Level" in data:
        st.bar_chart(data["Waste_Level"].value_counts())
    st.subheader("Daily demand trend")
    trend = data.groupby("Date", dropna=False)["Daily_Demand"].mean() if "Date" in data else pd.Series(dtype=float)
    if len(trend):
        st.line_chart(trend)
    st.info("The source dataset is synthetic. Results demonstrate a machine-learning workflow and are not validated operational forecasts.")
with tab2:
    st.subheader("Enter conditions for a prediction")
    col1,col2 = st.columns(2)
    business_options = sorted(data["Business_Type"].dropna().astype(str).unique()) if "Business_Type" in data else ["Restaurant"]
    food_options = sorted(data["Food_Category"].dropna().astype(str).unique()) if "Food_Category" in data else ["Prepared Food"]
    weather_options = sorted(data["Weather"].dropna().astype(str).unique()) if "Weather" in data else ["Sunny"]
    with col1:
        business = st.selectbox("Business type", business_options)
        food = st.selectbox("Food category", food_options)
        purchase = st.number_input("Purchase quantity", min_value=0.0, value=100.0, step=5.0)
        shelf = st.number_input("Shelf life (days)", min_value=1.0, value=3.0, step=1.0)
    with col2:
        temp = st.number_input("Storage temperature (°C)", value=5.0, step=1.0)
        promotion = st.selectbox("Promotion", ["No", "Yes"])
        weather = st.selectbox("Weather", weather_options)
        date = st.date_input("Forecast date")
    if st.button("Predict demand and waste", type="primary"):
        d = pd.Timestamp(date)
        row = {}
        for f in features:
            if f == "Business_Type": row[f] = business
            elif f == "Food_Category": row[f] = food
            elif f == "Purchase_Quantity": row[f] = purchase
            elif f == "Shelf_Life_Days": row[f] = shelf
            elif f == "Storage_Temperature": row[f] = temp
            elif f == "Promotion": row[f] = promotion
            elif f == "Weather": row[f] = weather
            elif f == "DayOfWeek": row[f] = d.dayofweek
            elif f == "Month": row[f] = d.month
            elif f == "DayOfYear": row[f] = d.dayofyear
            else: row[f] = np.nan
        x = pd.DataFrame([row], columns=features)
        demand = max(0, float(demand_model.predict(x)[0]))
        waste = max(0, float(waste_model.predict(x)[0]))
        risk = str(risk_model.predict(x)[0])
        p1,p2,p3 = st.columns(3)
        p1.metric("Predicted daily demand", f"{demand:.1f}")
        p2.metric("Predicted waste quantity", f"{waste:.1f}")
        p3.metric("Predicted waste risk", risk)
        st.caption("Predictions are generated by models trained on synthetic historical records; use as a prototype, not as a real operational decision.")
with tab3:
    st.subheader("Chronological holdout evaluation")
    st.write("The earliest 80% of dated records are used for training; the latest 20% are held out for evaluation.")
    for key, values in metrics.items():
        st.markdown(f"**{key.title()} model**")
        st.json(values)
    st.caption("Metrics are calculated at app runtime from the repository CSV. They may vary if the dataset changes.")
with tab4:
    st.subheader("Dataset preview")
    st.write("Columns:", ", ".join(data.columns))
    st.dataframe(data.head(100), use_container_width=True)
    st.download_button("Download loaded CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="food_waste_dataset.csv", mime="text/csv")
