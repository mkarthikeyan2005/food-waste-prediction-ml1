# Food Waste Prediction and Demand Forecasting Using Machine Learning

A Streamlit prototype built around the public dataset from:
https://github.com/kalyanipatange/Waste-Food-Management-Analysis-

## Features
- Exploratory charts for waste and demand
- Regression model for `Daily_Demand`
- Regression model for `Waste_Quantity`
- Classification model for `Waste_Level` (Low/Medium/High)
- Chronological 80/20 train-test split
- Interactive Streamlit prediction page

## Important dataset note
The source dataset is synthetic. The results demonstrate a workflow; they are not evidence of performance on real business operations. The app computes evaluation metrics when it loads the source data.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
If automatic download does not work, download `food_waste_dataset.csv` from the source GitHub repository and place it beside these files. The app also supports uploading the CSV from the sidebar.

## Deploy
Upload `app.py`, `train_model.py`, `requirements.txt`, and `README.md` to a GitHub repository. Streamlit Community Cloud: https://share.streamlit.io/ ; select `app.py` as the entrypoint.

## Models and evaluation
- Random Forest Regressor: demand and waste quantity
- Random Forest Classifier: waste severity
- Regression metrics: MAE, RMSE, R²
- Classification metrics: accuracy, macro F1
- Chronological holdout: earliest 80% train, latest 20% test

## Limitations
- Synthetic dataset, not real operational data.
- Predictors are restricted to features intended to be available before/at forecasting time; actual same-day sales and waste are not used as inputs.
- A chronological split by date is used to reduce optimistic evaluation from random shuffling.
