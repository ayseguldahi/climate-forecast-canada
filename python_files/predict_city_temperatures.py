import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from prophet import Prophet

df = pd.read_csv('csv_files/cleaned_datasets/merged_climate_history_final.csv')

# Features and Target
feature_columns = ['Year', 'Temperature Change (°C)', 'CO2 Emissions (per capita)', 'Total Sea Ice Area (millions km²)']
X = df[feature_columns]
Y = df[[col for col in df.columns if col.startswith('MEAN_TEMPERATURE')]]

# Feature Engineering
X['CO2 Growth Rate'] = X['CO2 Emissions (per capita)'].pct_change().fillna(0)
X['Sea Ice % Change'] = X['Total Sea Ice Area (millions km²)'].pct_change().fillna(0)

# Split Data (Train 80%, Test 20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Extrapolate Future Features
def extrapolate_feature(df, feature_name, years_to_predict):
    X_years = df['Year'].values.reshape(-1, 1)
    y_feature = df[feature_name].values
    model = LinearRegression()
    model.fit(X_years, y_feature)
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    future_preds = model.predict(future_years)
    return future_preds

future_years = np.arange(2025, 2031)
future_df_features = pd.DataFrame({
    'Year': future_years,
    'Temperature Change (°C)': [np.nan] * 6,  # Placeholder
    'CO2 Emissions (per capita)': extrapolate_feature(df, 'CO2 Emissions (per capita)', 6),
    'Total Sea Ice Area (millions km²)': extrapolate_feature(df, 'Total Sea Ice Area (millions km²)', 6)
})
future_df_features['CO2 Growth Rate'] = np.gradient(future_df_features['CO2 Emissions (per capita)'])
future_df_features['Sea Ice % Change'] = np.gradient(future_df_features['Total Sea Ice Area (millions km²)'])
future_df_features['Temperature Change (°C)'] = future_df_features['Temperature Change (°C)'].fillna(0)
model_types = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}
# Train Models City by City-
all_predictions = []  

for city in Y.columns:
    print(f"\nTraining models for: {city}")
    
    y_train_city = Y_train[city]
    y_test_city = Y_test[city]

    # ➡️ Random Forest and Gradient Boosting
    for model_name, model in model_types.items():
        model.fit(X_train, y_train_city)
        
        # Evaluation
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test_city, y_pred_test)

        # Future Prediction (2025-2030)
        future_preds = model.predict(future_df_features)

        for year, pred in zip(future_years, future_preds):
            all_predictions.append({
                'Year': year,
                'City': city,
                'Model': model_name,
                'Prediction': pred,
                'R2_Score': r2
            })

    # ➡️ Prophet Model
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df['Year'], format='%Y')
    prophet_df['y'] = df[city]

    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    prophet_model.fit(prophet_df)

    # Evaluation
    future_train = prophet_model.make_future_dataframe(periods=0, freq='Y')
    forecast_train = prophet_model.predict(future_train)
    prophet_yhat = forecast_train['yhat']
    prophet_r2 = r2_score(prophet_df['y'], prophet_yhat)

    # Future Predictions (2025-2030)
    future_years_prophet = pd.DataFrame({'ds': pd.date_range(start='2025', end='2030', freq='Y')})
    forecast_future = prophet_model.predict(future_years_prophet)

    for year, pred in zip(future_years, forecast_future['yhat']):
        all_predictions.append({
            'Year': year,
            'City': city,
            'Model': 'Prophet',
            'Prediction': pred,
            'R2_Score': prophet_r2
        })
# Save to CSV
final_temperature_predictions = pd.DataFrame(all_predictions)
final_temperature_predictions.to_csv('csv_files/predictions_scores/temperature_predictions.csv', index=False)

print("\n✅ All models trained. Predictions saved successfully!")

