import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

co2_df_cleaned = pd.read_csv("csv_files/cleaned_datasets/co2_cleaned.csv")

# Add GDP Category
def categorize_gdp(gdp):
    if gdp > 1_000_000_000_000:  # > 1 trillion
        return 'High Income'
    elif gdp > 100_000_000_000:  # 100B to 1T
        return 'Upper Middle Income'
    elif gdp > 10_000_000_000:   # 10B to 100B
        return 'Lower Middle Income'
    else:
        return 'Low Income'

co2_df_cleaned['GDP_Category'] = co2_df_cleaned['GDP_PPP'].apply(categorize_gdp)

# Add Region 
continent_map = {
    "Canada": "North America",
    "Brazil": "South America",
    "Germany": "Europe",
    "India": "Asia",
    "Australia": "Oceania",
    "South Africa": "Africa",
    "United States": "North America",
    "China": "Asia",
    "Russia": "Europe",
    "Turkey": "Europe",
    "Egypt": "Africa",
    "Japan": "Asia"
}

co2_df_cleaned['Region'] = co2_df_cleaned['Country'].map(continent_map)
co2_df_cleaned['Region'] = co2_df_cleaned['Region'].fillna('Other')  

# Encode Categorical Features
le_country = LabelEncoder()
le_gdp_category = LabelEncoder()
le_region = LabelEncoder()

co2_df_cleaned['Country_Code'] = le_country.fit_transform(co2_df_cleaned['Country'])
co2_df_cleaned['GDP_Category_Code'] = le_gdp_category.fit_transform(co2_df_cleaned['GDP_Category'])
co2_df_cleaned['Region_Code'] = le_region.fit_transform(co2_df_cleaned['Region'])

# Prepare Training Data
X = co2_df_cleaned[['Country_Code', 'Year', 'GDP_PPP', 'GDP_Category_Code', 'Region_Code']]
y = co2_df_cleaned['CO2_Emissions']

# Prepare future years to predict
future_years = np.arange(2025, 2031)

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
rf.fit(X, y)

# Train Gradient Boosting (auto-tuned)
gb = GradientBoostingRegressor(random_state=42)
param_dist = {
    "n_estimators": randint(100, 400),
    "learning_rate": uniform(0.01, 0.1),
    "max_depth": randint(3, 6),
    "subsample": uniform(0.7, 0.3),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10)
}
random_search = RandomizedSearchCV(
    gb, param_distributions=param_dist, n_iter=20,
    cv=3, verbose=0, n_jobs=-1, random_state=42, scoring='r2'
)
random_search.fit(X, y)
best_gb = random_search.best_estimator_

# Predict for each country
results = []

for country in co2_df_cleaned['Country'].unique():
    country_df = co2_df_cleaned[co2_df_cleaned['Country'] == country]

    # Prophet prediction
    prophet_df = country_df[['Year', 'CO2_Emissions']].rename(columns={'Year': 'ds', 'CO2_Emissions': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

    if len(prophet_df) > 5:  # Prophet needs enough data
        model = Prophet()
        model.fit(prophet_df)

        future = pd.DataFrame({'ds': pd.to_datetime(future_years, format='%Y')})
        forecast = model.predict(future)
        prophet_preds = forecast['yhat'].values

        # Prophet R2
        train_pred = model.predict(prophet_df[['ds']])['yhat']
        r2_prophet = r2_score(prophet_df['y'], train_pred)
    else:
        prophet_preds = [np.nan] * len(future_years)
        r2_prophet = np.nan

    # Prepare inputs for RF and GB
    country_code = le_country.transform([country])[0]
    gdp_category_code = le_gdp_category.transform([country_df['GDP_Category'].iloc[0]])[0]
    region_code = le_region.transform([country_df['Region'].iloc[0]])[0]
    avg_gdp = country_df['GDP_PPP'].mean()

    X_future = pd.DataFrame({
        'Country_Code': [country_code] * len(future_years),
        'Year': future_years,
        'GDP_PPP': [avg_gdp] * len(future_years),
        'GDP_Category_Code': [gdp_category_code] * len(future_years),
        'Region_Code': [region_code] * len(future_years)
    })

    # Predict
    rf_preds = rf.predict(X_future)
    r2_rf = rf.score(X, y)

    gb_preds = best_gb.predict(X_future)
    r2_gb = best_gb.score(X, y)

    # Save results
    for i, year in enumerate(future_years):
        results.append((year, country, "Prophet", prophet_preds[i], r2_prophet))
        results.append((year, country, "Random Forest", rf_preds[i], r2_rf))
        results.append((year, country, "Gradient Boosting", gb_preds[i], r2_gb))

# Save results
predictions_df = pd.DataFrame(results, columns=['Year', 'Country', 'Model', 'Prediction', 'R2_Score'])
predictions_df.to_csv("csv_files/predictions_scores/co2_predictions", index=False)

print("CO2 predictions (with GDP Category and Region) are saved!")
