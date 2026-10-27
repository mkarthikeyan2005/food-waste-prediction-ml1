# Food Waste Prediction and Demand Forecasting Using Machine Learning

## Abstract
Food waste creates operational and sustainability challenges for food businesses. This project develops a machine-learning prototype to estimate daily demand, predict waste quantity, and classify waste severity using inventory and operating-condition features. The implementation uses the public synthetic dataset published in the Waste-Food-Management-Analysis GitHub repository. Data processing and model evaluation use a chronological holdout so the latest records are reserved for testing.

## Objectives
1. Explore food inventory, demand, and waste patterns.
2. Build a regression model for daily demand.
3. Build a regression model for waste quantity.
4. Classify waste severity as Low, Medium, or High.
5. Present visualizations and interactive predictions through a Streamlit dashboard.

## Dataset
Source: https://github.com/kalyanipatange/Waste-Food-Management-Analysis-

The repository describes 8,000 daily inventory records covering 1 January through 31 December 2025, with 16 columns. Important fields include business type, food category, purchase quantity, shelf life, storage temperature, daily demand, promotion, weather, waste quantity, waste reason, donation status, and waste level.

**Dataset limitation:** The source explicitly identifies the dataset as synthetic. Findings are therefore for demonstration and learning; they should not be presented as results measured from real restaurants or supermarkets.

## Methodology
1. Load the CSV and parse dates.
2. Inspect missing values and data types.
3. Derive day-of-week, month, and day-of-year features.
4. Use operational features that can plausibly be known at prediction time.
5. Sort records chronologically and use the first 80% for training and latest 20% for testing.
6. Train Random Forest models for demand and waste quantity, and a Random Forest classifier for waste severity.
7. Evaluate regression with MAE, RMSE, and R²; evaluate classification with accuracy and macro F1.
8. Display summary charts and predictions in Streamlit.

## System Requirements
Python 3.10 or later; pandas; NumPy; scikit-learn; Streamlit; requests; joblib.

## Results
Run `python train_model.py` to calculate the metrics on the repository dataset. The script writes `model_metrics.json` and `food_waste_models.joblib`. The app also computes metrics at runtime.

Do not insert fabricated metric values into the report. Copy the actual output from `model_metrics.json` after running the script.

## Conclusion
The prototype demonstrates a complete workflow from dataset inspection through model training, evaluation, and an interactive dashboard. Further work should validate the approach on real operational records, compare more forecasting baselines, and assess the cost of overproduction versus stockouts.

## Future Scope
- Validate with real daily demand and waste logs.
- Compare Random Forest against seasonal-naive and gradient-boosting baselines.
- Add model explainability and prediction intervals.
- Include inventory recommendations after validating forecast errors with domain experts.
