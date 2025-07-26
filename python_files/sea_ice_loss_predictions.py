import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from scipy.stats import uniform, randint
from prophet import Prophet

df = pd.read_csv("csv_files/cleaned_datasets/final_climate_dataset.csv", encoding='latin1')

summer_columns = [col for col in df.columns if '(Summer)' in col]
september_columns = [col for col in df.columns if '(September)' in col]

results_summer = []
results_september = []

# Extend years to predict
future_years = np.arange(2025, 2031)

# Random Forest model 
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

# Randomized Search parameters for Gradient Boosting
param_dist = {
    "n_estimators": randint(100, 400),
    "learning_rate": uniform(0.01, 0.1),
    "max_depth": randint(2, 5),
    "subsample": uniform(0.7, 0.3),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10)
}

# Helper function to process one region
def model_and_predict(region_name, df, future_years):
    region_df = df[['Year', region_name]].copy().dropna()

    # Prophet setup
    prophet_df = region_df.rename(columns={"Year": "ds", region_name: "y"})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
    prophet = Prophet()
    prophet.fit(prophet_df)
    future = pd.DataFrame({'ds': pd.to_datetime(future_years, format='%Y')})
    prophet_pred = prophet.predict(future)
    future_prophet = prophet_pred['yhat'].values
    train_pred_prophet = prophet.predict(prophet_df[['ds']])['yhat']
    r2_prophet = r2_score(prophet_df['y'], train_pred_prophet)

    # Machine learning setup
    X = region_df[['Year']]
    y = region_df[region_name]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_future = poly.transform(future_years.reshape(-1, 1))

    # Random Forest
    rf = rf_model
    rf.fit(X_poly, y)
    rf_pred = rf.predict(X_future)
    r2_rf = rf.score(X_poly, y)

    # Gradient Boosting with RandomizedSearchCV
    gb = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        gb, param_distributions=param_dist, n_iter=20,
        cv=3, verbose=0, n_jobs=-1, random_state=42, scoring='r2'
    )
    random_search.fit(X_poly, y)
    best_gb = random_search.best_estimator_

    # Predict with best Gradient Boosting model
    gb_pred = best_gb.predict(X_future)
    r2_gb = best_gb.score(X_poly, y)

    # Collect results
    result = []
    for i, year in enumerate(future_years):
        result.append((year, region_name, "Prophet", future_prophet[i], r2_prophet))
        result.append((year, region_name, "Random Forest", rf_pred[i], r2_rf))
        result.append((year, region_name, "Gradient Boosting", gb_pred[i], r2_gb))

    return result

# Process all Summer regions
for region in summer_columns:
    print(f"Processing {region} (Summer)...")
    result = model_and_predict(region, df, future_years)
    results_summer.extend(result)

# Process all September regions
for region in september_columns:
    print(f"Processing {region} (September)...")
    result = model_and_predict(region, df, future_years)
    results_september.extend(result)

# Create DataFrames
summer_predictions = pd.DataFrame(results_summer, columns=['Year', 'Region', 'Model', 'Prediction', 'R2_Score'])
september_predictions = pd.DataFrame(results_september, columns=['Year', 'Region', 'Model', 'Prediction', 'R2_Score'])

# Save to CSV
summer_predictions.to_csv("csv_files/predictions_scores/summer_predictions.csv", index=False)
september_predictions.to_csv("csv_files/predictions_scores/september_predictions.csv", index=False)

print("Summer and September predictions saved")
