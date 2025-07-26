# 🌳 Climate Change in Canada: Historical Analysis and Future Projections

![MIT License](https://img.shields.io/badge/license-MIT-green)

## 🧾 Project Overview

The goal of this project is to predict key climate indicators for Canada and the world using advanced machine learning techniques. I used historical data from 1968 to 2024 (except for CO₂ emissions data, which covers a different range) to train models and generate predictions for the future period of 2025 to 2030.

The project focuses on forecasting:
- 🧊 **Sea ice loss** (for Canada)
- 🌡️ **Temperature change** (for Canada)
- 🌧️ **Precipitation trends** (for Canada)
- 🌍 **CO₂ emissions** (globally)
## 🧠 Project Insights & Reflections

This project explores the historical patterns and future projections of climate change with a particular focus on Canada, using machine learning models to predict temperature changes, sea ice loss, precipitation trends, and global CO₂ emissions. 

By analyzing data from **1968 to 2024** and forecasting trends for **2025–2030**, I aim to provide a **data-driven view** that can support policymakers, researchers, and environmental organizations in making informed decisions about climate action.

The project combines:
- 🌲 **Traditional ML models**: Random Forest, Gradient Boosting  
- ⏳ **Time series forecasting**: Prophet  
- ⚙️ **Enhancements**: Regularization, feature engineering, and hyperparameter tuning

🔍 **Key Highlights**:
- 📈 Interactive dashboards with animated maps and trendlines
- 🌍 Global comparison of CO₂ emissions
- ❄️ Arctic sea ice loss as a critical concern
- 🌧️ Regional precipitation forecasting showed mixed accuracy
- 📊 Best prediction scores achieved in CO₂ emissions forecasting

This work demonstrates the power of combining **multiple modeling techniques** with contextual domain knowledge to better understand and visualize climate risk.
---

## 📂 Project Structure

```
📁 climate/                            # Virtual environment folder

📁 csv_files/
│   📂 cleaned_datasets/              # Cleaned datasets used in modeling
│   ├─ final_climate_dataset.csv
│   ├─ co2_cleaned.csv
│   ├─ merged_climate_history_final.csv
│   ├─ summer_semptember_trends.csv
│
│   📂 predictions_scores/            # Predictions and model performance metrics
│   ├─ co2_predictions_with_clusters.csv
│   ├─ precipitation_predictions.csv
│   ├─ september_predictions.csv
│   ├─ summer_predictions.csv
│   └─ temperature_predictions.csv
│
│   📂 raw_datasets/                  # Original datasets without editing
│   ├─ climate_history.csv
│   ├─ co2-emissions-and-gdp.csv
│   ├─ co-emissions-per-capita.csv
│   ├─ sea_ice_loss.csv
│   ├─ September-trends-Arctic-en.csv
│   ├─ summer-trends-NCW.csv
│   └─ Temperature-change-annual-en.csv

📁 python_files/                       # Python scripts
├─ clean_co2_dataset.py               # CO₂ emissions data cleaning
├─ dashboard.py                       # Main Streamlit dashboard
├─ data_cleaning.py                   # Scripts for data preprocessing
├─ merge_all_datasets.py              # Merge datasets
├─ predict_city_temperatures.py       # Temperature prediction models
├─ predict_co2_emissions.py           # CO₂ emissions prediction models
├─ predict_precipitation.py           # Precipitation prediction models
└─ sea_ice_loss_predictions.py        # Sea ice loss prediction models

📄 READ_ME.txt                         # Important details of the project
📄 requirements.txt                    # How to run this project
📄 dataset_resources.xlsx              # Dataset sources and descriptions
📄 Streamlit_Dashboard.pdf             # Dashboard screenshots
```

---

## 🚀 Environment Setup

1. **Create a virtual environment (optional but recommended).**
2. **Activate the environment**:

- On **Windows**:
```bash
.\climate\Scripts\Activate
```

- On **Mac/Linux**:
```bash
source climate/bin/activate
```

3. **Install the required packages**:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Streamlit Dashboard?

```bash
streamlit run python_files/dashboard.py
```

---

## 📊 Dashboard Features

- 📈 **Exploratory Data Analysis (EDA)**  
  Visualize trends in Canada's historical climate patterns.

- ❄️ **Sea Ice Loss Predictions**  
  Forecast summer and September sea ice area loss across Northern Canadian Waters.

- 🌡️ **Temperature Predictions**  
  Predict average temperature changes across major Canadian cities.

- 🌧️ **Precipitation Predictions**  
  Estimate precipitation patterns and extreme weather trends.

- 🌍 **CO₂ Emissions Forecast**  
  Model future CO₂ emissions by country and visualize results on a global map.

- 🎥 **Interactive Animations**  
  Year-by-year animated maps showing environmental changes over time.

---

## 👤 Author
Aysegul Dahi
🔗 [LinkedIn](https://linkedin.com/in/ayseguldahi)
---

> ⭐ Star this repository if you found the project valuable!
---

---


