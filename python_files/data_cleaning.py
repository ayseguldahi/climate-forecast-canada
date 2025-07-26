import pandas as pd

# Load Datasets
september_trends_path = "csv_files/raw_datasets/September-trends-Arctic-en.csv"
summer_trends_path = "csv_files/raw_datasets/summer-trends-NCW.csv"

september_trends = pd.read_csv(september_trends_path)
summer_trends = pd.read_csv(summer_trends_path)

print("September Trends Shape:", september_trends.shape)
print("Summer Trends Shape:", summer_trends.shape)

#  Rename Columns (Fixing for Summer and September)
september_trends = september_trends.rename(columns={
    "Foxe Basin": "Foxe Basin (September)",
    "Kane Basin": "Kane Basin (September)",
    "Baffin Bay": "Baffin Bay (September)",
    "Beaufort Sea": "Beaufort Sea (September)",
    "Canadian Arctic Archipelago": "Arctic Archipelago (September)",
    "Canadian Arctic domain (thousands of square kilometres)": "Arctic Domain (September)"
})

summer_trends = summer_trends.rename(columns={
    "Foxe Basin": "Foxe Basin (Summer)",
    "Kane Basin (thousands of square kilometres)": "Kane Basin (Summer)",
    "Baffin Bay (thousands of square kilometres)": "Baffin Bay (Summer)",
    "Beaufort Sea (thousands of square kilometres)": "Beaufort Sea (Summer)",
    "Canadian Arctic Archipelago": "Arctic Archipelago (Summer)",
    "Hudson Bay (thousands of square kilometres)": "Hudson Bay (Summer)",
    "Hudson Strait (thousands of square kilometres)": "Hudson Strait (Summer)",
    "Davis Strait (thousands of square kilometres)": "Davis Strait (Summer)",
    "Northern Labrador Sea (thousands of square kilometres)": "Northern Labrador Sea (Summer)"
})

# Convert Numeric Columns
for df in [september_trends, summer_trends]:
    for col in df.columns[1:]:  # Skip "Year" column
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Merge Datasets on Year
merged_trends = pd.merge(summer_trends, september_trends, on="Year", how="outer")

merged_trends.columns = merged_trends.columns.str.strip()

merged_trends["Year"] = pd.to_numeric(merged_trends["Year"], errors="coerce").astype("Int64")

merged_trends = merged_trends.dropna(subset=["Year"])

merged_trends.to_csv("csv_files/cleaned_datasets/summer_september_trends.csv", index=False)

# Show Summary
print("Final Dataset Info:")
print(merged_trends.info())
print("Final Dataset Preview:")
print(merged_trends.head())
print("Final cleaned and merged dataset saved successfully!")


