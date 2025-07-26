# 🌳 Climate Change in Canada: Historical Analysis and Future Projections

## 🧾 Project Overview

This project analyzes historical climate data from Canada and uses machine learning models (Random Forest, Prophet, and Gradient Boosting) to predict future trends in temperature change, sea ice loss, precipitation patterns, and CO₂ emissions. The goal is to empower policymakers, researchers, and civil society organizations to make data-driven decisions to combat climate change over the next five years.

---

## 📂 Project Structure

```

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
📄 Aysegul_Dahi_Presentation.pptx      # Presentation slides
📄 Aysegul_Dahi_Presentation.pdf       # PDF version of presentation
📄 Streamlit_Dashboard.pdf             # Dashboard screenshots
```

---

## 🚀 Environment Setup

1. **Create a virtual environment (optional but recommended).**
2. **Activate the environment**:

- On **Windows**:
```bash
.\climate\Scriptsctivate
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
