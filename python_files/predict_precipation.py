# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset
df = pd.read_csv('csv_files/cleaned_datasets/merged_climate_history_final.csv')

# Select only Precipitation Columns
precipitation_cols = [col for col in df.columns if col.startswith('TOTAL_PRECIPITATION')]

# Prepare Features
X = df[['Year', 'Temperature Change (°C)', 'CO2 Emissions (per capita)', 'Total Sea Ice Area (millions km²)']].copy()

# Fill missing values with 0
X = X.fillna(0)

# Create Growth Rates
X['CO2 Growth Rate'] = X['CO2 Emissions (per capita)'].pct_change().fillna(0)
X['Sea Ice % Change'] = X['Total Sea Ice Area (millions km²)'].pct_change().fillna(0)

# Target: Y
Y = df[precipitation_cols]

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Models to use
model_types = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Extrapolate Future Features (2025-2030)
def extrapolate_feature(df, feature_name):
    model = LinearRegression()
    X_years = df['Year'].values.reshape(-1, 1)
    y_feature = df[feature_name].values
    model.fit(X_years, y_feature)
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    preds = model.predict(future_years)
    return preds

future_years = np.arange(2025, 2031)
future_df_features = pd.DataFrame({
    'Year': future_years,
    'Temperature Change (°C)': [0] * 6,
    'CO2 Emissions (per capita)': extrapolate_feature(df, 'CO2 Emissions (per capita)'),
    'Total Sea Ice Area (millions km²)': extrapolate_feature(df, 'Total Sea Ice Area (millions km²)')
})
future_df_features['CO2 Growth Rate'] = np.gradient(future_df_features['CO2 Emissions (per capita)'])
future_df_features['Sea Ice % Change'] = np.gradient(future_df_features['Total Sea Ice Area (millions km²)'])

# 🛠️ Train and Predict City by City
all_results = []

for city in precipitation_cols:
    print(f"\nTraining models for: {city}")
    y_train_city = Y_train[city]
    y_test_city = Y_test[city]

    # Machine Learning Models
    for model_name, model in model_types.items():
        model.fit(X_train, y_train_city)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test_city, y_pred)

        # Predict Future
        future_preds = model.predict(future_df_features)

        for year, pred in zip(future_years, future_preds):
            all_results.append({
                'Year': year,
                'City': city,
                'Model': model_name,
                'Prediction': pred,
                'R2_Score': r2
            })

    # Prophet Model
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Year'], format='%Y'),
        'y': df[city]
    })

    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    prophet_model.fit(prophet_df)

    # Evaluate on Training
    future_train = prophet_model.make_future_dataframe(periods=0, freq='Y')
    forecast_train = prophet_model.predict(future_train)
    prophet_r2 = r2_score(prophet_df['y'], forecast_train['yhat'])

    # Predict Future
    future_years_prophet = pd.DataFrame({'ds': pd.date_range(start='2025', end='2031', freq='Y')})
    forecast_future = prophet_model.predict(future_years_prophet)

    for year, pred in zip(future_years, forecast_future['yhat']):
        all_results.append({
            'Year': year,
            'City': city,
            'Model': 'Prophet',
            'Prediction': pred,
            'R2_Score': prophet_r2
        })

# Save Final Results
results_df = pd.DataFrame(all_results)
results_df.to_csv('csv_files/predictions_scores/precipitation_predictions.csv', index=False)

print("✅ Precipitation predictions (Random Forest + Gradient Boosting + Prophet) saved successfully!")

# 📈 Example Plot
example_city = 'TOTAL_PRECIPITATION_TORONTO'
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df[example_city], label='Historical Toronto')
predicted_df = results_df[(results_df['City'] == example_city) & (results_df['Model'] == 'Prophet')]
plt.plot(predicted_df['Year'], predicted_df['Prediction'], label='Predicted Toronto (Prophet)', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Total Precipitation')
plt.title('Toronto Precipitation Prediction (2025-2030)')
plt.legend()
plt.grid(True)
plt.show()
