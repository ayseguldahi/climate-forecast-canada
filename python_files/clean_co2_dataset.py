import pandas as pd
co2_df = pd.read_csv("csv_files/raw_datasets/co2-emissions-and-gdp.csv")

# Drop rows where the target 'annual_CO2_emissions' is missing
co2_df_cleaned = co2_df.dropna(subset=['annual_CO2_emissions'])

# Drop rows where 'GDP, PPP (constant 2021 international $)' is missing
# Because GDP is an important feature for Random Forest and Gradient Boosting
co2_df_cleaned = co2_df_cleaned.dropna(subset=['GDP, PPP (constant 2021 international $)'])

# Drop unnecessary columns
# 'Code' and 'annual_consumption_based_CO2_emissions' are not needed for this project
co2_df_cleaned = co2_df_cleaned.drop(columns=['Code', 'annual_consumption_based_CO2_emissions'])

co2_df_cleaned = co2_df_cleaned.drop_duplicates()
co2_df_cleaned = co2_df_cleaned.reset_index(drop=True)
co2_df_cleaned = co2_df_cleaned.rename(columns={
    'GDP, PPP (constant 2021 international $)': 'GDP_PPP',
    'annual_CO2_emissions': 'CO2_Emissions'
})

co2_df_cleaned.info()
print(co2_df_cleaned.head())
co2_df_cleaned.to_csv("csv_files/cleaned_datasets/co2_cleaned.csv", index=False)
