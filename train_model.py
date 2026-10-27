"""Train demand, waste quantity, and waste-risk models on the repository CSV."""
import os, json
import pandas as pd
import numpy as np
import requests
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import joblib

URL = "https://raw.githubusercontent.com/kalyanipatange/Waste-Food-Management-Analysis-/main/food_waste_dataset.csv"
def load_csv():
    local = "food_waste_dataset.csv"
    if os.path.exists(local):
        return pd.read_csv(local)
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    open(local, "wb").write(r.content)
    return pd.read_csv(local)

df = load_csv()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear
features = [c for c in ["Business_Type","Food_Category","Purchase_Quantity","Shelf_Life_Days","Storage_Temperature","Promotion","Weather","DayOfWeek","Month","DayOfYear"] if c in df.columns]
categorical_names = {"Business_Type", "Food_Category", "Promotion", "Weather"}
cats = [c for c in features if c in categorical_names]
nums = [c for c in features if c not in cats]
for c in nums:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in cats:
    df[c] = df[c].astype("string").fillna("Unknown")
pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), nums),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),("encode", OneHotEncoder(handle_unknown="ignore"))]), cats)
])
split = int(len(df)*0.8)
train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
results = {}
models = {}
for target in ["Daily_Demand", "Waste_Quantity"]:
    tr, te = train.dropna(subset=[target]), test.dropna(subset=[target])
    model = Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1))])
    model.fit(tr[features], tr[target])
    pred = np.maximum(0, model.predict(te[features]))
    results[target] = {"MAE": float(mean_absolute_error(te[target], pred)),
                       "RMSE": float(np.sqrt(mean_squared_error(te[target], pred))),
                       "R2": float(r2_score(te[target], pred)) if len(te)>1 else None}
    models[target] = model
tr, te = train.dropna(subset=["Waste_Level"]), test.dropna(subset=["Waste_Level"])
clf = Pipeline([("prep", pre), ("model", RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", n_jobs=-1))])
clf.fit(tr[features], tr["Waste_Level"].astype(str))
pred = clf.predict(te[features])
results["Waste_Level"] = {"accuracy": float(accuracy_score(te["Waste_Level"].astype(str), pred)),
                          "macro_f1": float(f1_score(te["Waste_Level"].astype(str), pred, average="macro"))}
models["Waste_Level"] = clf
joblib.dump({"models": models, "features": features}, "food_waste_models.joblib")
json.dump({"dataset": "kalyanipatange/Waste-Food-Management-Analysis-", "is_synthetic": True,
           "split": "chronological 80/20", "metrics": results}, open("model_metrics.json","w"), indent=2)
print(json.dumps(results, indent=2))
print("Saved food_waste_models.joblib and model_metrics.json")
