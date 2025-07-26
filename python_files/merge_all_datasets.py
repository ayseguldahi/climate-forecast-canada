import pandas as pd
import os
data_files = {
    "sea_ice_loss": "csv_files/raw_datasets/sea_ice_loss.csv",
    "temperature_change": "csv_files/raw_datasets/Temperature-change-annual-en.csv",
    "co2_emissions": "csv_files/raw_datasets/co-emissions-per-capita.csv"
}

datasets = {name: pd.read_csv(path) for name, path in data_files.items()}

# Standardize Column Names for Consistency
datasets["sea_ice_loss"].rename(columns={
    "Northern Canadian Waters sea ice area (millions of square kilometres)": "Total Sea Ice Area (millions km²)"
}, inplace=True)

datasets["temperature_change"].rename(columns={
    "Temperature departure (degree Celsius)": "Temperature Change (°C)",
    "Warmest year ranking": "Warmest Year Rank"
}, inplace=True)

datasets["co2_emissions"].rename(columns={
    "CO2_emissions_per_capita": "CO2 Emissions (per capita)"
}, inplace=True)

#  Convert "Year" Column to Integer Format in All Datasets
for df in datasets.values():
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

#  Load Previously Merged Summer & September Trends Dataset
merged_trends_path = "csv_files/cleaned_datasets/summer_september_trends.csv"
merged_trends = pd.read_csv(merged_trends_path)

# Ensure "Year" column is numeric
merged_trends["Year"] = pd.to_numeric(merged_trends["Year"], errors="coerce").astype("Int64")

# Merge Additional Datasets with Summer & September Trends
merged_full = merged_trends.copy()

for name, df in datasets.items():
    merged_full = pd.merge(merged_full, df, on="Year", how="outer")

#  Drop Rows Where "Year" < 1968
merged_full = merged_full[merged_full["Year"] >= 1968]

# Check missing values count
missing_values = merged_full.isnull().sum()
print("Missing Values in Each Column:\n")
print(missing_values[missing_values > 0])

# Fill missing values using median for numerical columns
merged_full.fillna({
    "Temperature Change (°C)": merged_full["Temperature Change (°C)"].median(),
    "CO2 Emissions (per capita)": merged_full["CO2 Emissions (per capita)"].median(),
    "Total Sea Ice Area (millions km²)": merged_full["Total Sea Ice Area (millions km²)"].median()
}, inplace=True)

# Forward-fill for time-series data
merged_full.ffill(inplace=True)

# Convert Temperature Change to Fahrenheit & Keep Both
merged_full["Temperature Change (°F)"] = merged_full["Temperature Change (°C)"] * 9/5 + 32

output_dir = "csv_files"
os.makedirs(output_dir, exist_ok=True)  
output_path = os.path.join(output_dir, "cleaned_datasets/final_climate_dataset.csv")

merged_full.to_csv(output_path, index=False)

print(merged_full.info())
print(merged_full.head())

print("Cleaned dataset saved successfully at: {output_path}")

