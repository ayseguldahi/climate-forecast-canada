import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


st.sidebar.title("🍁 Climate Change in Canada: Historical Analysis and Future Projections")
section = st.sidebar.radio("Navigate to", [
    "📊 EDA Analysis", 
    "❄️ Ice Loss Predictions", 
    "🌡️ Temperature Predictions", 
    "🌧️ Precipitation Predictions", 
    "🌍 CO₂ Emissions Predictions"
    ])
st.sidebar.markdown("""
**Project Goal:**  
This project analyzes historical climate data from Canada and uses machine learning models (Random Forest, Prophet and Gradient Boosting) to predict future trends in temperature change, sea ice loss precipitation patterns, CO₂ emissions. The goal is to empower policymakers, researchers, and civil society organizations to make data-driven decisions to combat climate change over the next five years.

---

**Prepared by:**  
*Aysegul Dahi*

---
                    
**Student Number:**
*300387536*

---
 
""")

if section == "📊 EDA Analysis":

    # Global CO₂ Emissions EDA
    
    @st.cache_data
    def load_co2_data():
        df_co2 = pd.read_csv("csv_files/cleaned_datasets/co2_cleaned.csv")
        return df_co2

    co2_df = load_co2_data()

    st.title("📊 EDA Analysis: Global CO₂ Emissions")

    st.subheader("🔎 CO₂ Emissions Overview")
    world_co2 = co2_df.groupby('Year')['CO2_Emissions'].sum().reset_index()

    fig = px.line(
        world_co2,
        x='Year',
        y='CO2_Emissions',
        title='🌎 Total Global CO₂ Emissions Over Time',
        labels={'CO2_Emissions': 'Total CO₂ Emissions (tons)'},
        template='plotly_white'
    )
    st.plotly_chart(fig)

    st.subheader("🏆 Top 10 CO₂ Emitting Countries (Latest Year)")
    latest_year = co2_df['Year'].max()
    latest_data = co2_df[co2_df['Year'] == latest_year]

    top_10_emitters = latest_data.sort_values('CO2_Emissions', ascending=False).head(10)

    fig2 = px.bar(
        top_10_emitters,
        x='Country',
        y='CO2_Emissions',
        title=f'🏅 Top 10 CO₂ Emitting Countries in {latest_year}',
        text_auto='.2s',
        template='plotly_white',
        color='CO2_Emissions'
    )
    st.plotly_chart(fig2)

    st.subheader("📈 CO₂ Emission Distribution")
    fig3, ax = plt.subplots(figsize=(8,5))
    sns.histplot(co2_df["CO2_Emissions"], bins=50, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of CO₂ Emissions')
    ax.set_xlabel('CO₂ Emissions (tons)')
    st.pyplot(fig3)

    st.subheader("🗺️ Global CO₂ Emissions Map")
    fig4 = px.choropleth(
        latest_data,
        locations="Country",
        locationmode="country names",
        color="CO2_Emissions",
        hover_name="Country",
        color_continuous_scale="Reds",
        title=f'🗺️ Global CO₂ Emissions Map - {latest_year}',
        template='plotly_white'
    )
    st.plotly_chart(fig4)

    st.info("""
    - Global CO₂ emissions are rising steadily over time.
    - China, the United States, and India are consistently the top CO₂ emitters.
    - The global distribution of CO₂ emissions is heavily skewed toward a few countries.
    """)
    # Canadian Climate EDA
    st.subheader("🇨🇦 Canadian Cities Climate Overview (1968–2024)")

    canada_df = pd.read_csv("csv_files/cleaned_datasets/merged_climate_history_final.csv")
    canada_df.columns = canada_df.columns.str.replace('Â', '', regex=True)

    st.write("Dataset Shape:", canada_df.shape)

    st.subheader("📈 Average Temperature Change in Canadian Cities Over Time")
    city_temp_cols = [col for col in canada_df.columns if "MEAN_TEMPERATURE" in col]
    avg_temp = canada_df[city_temp_cols].mean(axis=1)

    fig = px.line(
        x=canada_df["Year"], 
        y=avg_temp,
        labels={"x": "Year", "y": "Average Temperature (°C)"},
        title="Average Temperature Across Canadian Cities (1968–2024)"
    )
    st.plotly_chart(fig)

    st.subheader("🌧️ Average Precipitation in Canadian Cities Over Time")
    city_precip_cols = [col for col in canada_df.columns if "TOTAL_PRECIPITATION" in col]
    avg_precip = canada_df[city_precip_cols].mean(axis=1)

    fig = px.line(
        x=canada_df["Year"], 
        y=avg_precip,
        labels={"x": "Year", "y": "Average Precipitation (mm)"},
        title="Average Total Precipitation Across Canadian Cities (1968–2024)"
    )
    st.plotly_chart(fig)

    # Animated Maps for Temperature and Precipitation
    city_coords = {
        "CALGARY": (51.0447, -114.0719),
        "EDMONTON": (53.5461, -113.4938),
        "HALIFAX": (44.6488, -63.5752),
        "MONCTON": (46.0878, -64.7782),
        "MONTREAL": (45.5017, -73.5673),
        "OTTAWA": (45.4215, -75.6995),
        "QUEBEC": (46.8139, -71.2082),
        "SASKATOON": (52.1579, -106.6702),
        "STJOHNS": (47.5615, -52.7126),
        "TORONTO": (43.65107, -79.347015),
        "VANCOUVER": (49.2827, -123.1207),
        "WHITEHORSE": (60.7212, -135.0568),
        "WINNIPEG": (49.8951, -97.1384)
    }

    # Prepare and plot temperature map
    map_data = []
    for _, row in canada_df.iterrows():
        for city in city_coords.keys():
            city_col = f"MEAN_TEMPERATURE_{city}"
            if city_col in row:
                map_data.append({
                    "City": city.title(),
                    "Year": row["Year"],
                    "Temperature (°C)": row[city_col],
                    "Latitude": city_coords[city][0],
                    "Longitude": city_coords[city][1]
                })

    map_df = pd.DataFrame(map_data)

    fig_temp_change = px.scatter_mapbox(
        map_df,
        lat="Latitude",
        lon="Longitude",
        size=map_df["Temperature (°C)"].abs(),
        color="Temperature (°C)",
        hover_name="City",
        animation_frame="Year",
        color_continuous_scale="thermal",
        zoom=3,
        height=600,
        title="Canadian Cities: Temperature Change Over Years"
    )

    fig_temp_change.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_temp_change)

    # Prepare and plot precipitation map
    precip_map_data = []
    for _, row in canada_df.iterrows():
        for city in city_coords.keys():
            city_col = f"TOTAL_PRECIPITATION_{city}"
            if city_col in row:
                precip_map_data.append({
                    "City": city.title(),
                    "Year": row["Year"],
                    "Precipitation (mm)": row[city_col],
                    "Latitude": city_coords[city][0],
                    "Longitude": city_coords[city][1]
                })

    precip_map_df = pd.DataFrame(precip_map_data)

    fig_precip_change = px.scatter_mapbox(
        precip_map_df,
        lat="Latitude",
        lon="Longitude",
        size=precip_map_df["Precipitation (mm)"].abs(),
        color="Precipitation (mm)",
        hover_name="City",
        animation_frame="Year",
        color_continuous_scale="blues",
        zoom=3,
        height=600,
        title="Canadian Cities: Precipitation Change Over Years"
    )

    fig_precip_change.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_precip_change)

    st.subheader("🔎 Key Insights for Canadian Climate")
    st.write("""
    - Temperatures in Canadian cities show an increasing trend since the 1970s.
    - Precipitation levels have moderately fluctuated across years, with slight increases in some regions.
    - Warmer cities like Vancouver and Toronto consistently remain above national averages.
    - Northern cities like Whitehorse show more extreme fluctuations in temperature and precipitation.
    """)

    # Arctic Sea Ice Loss EDA

    st.subheader("🧊 Sea Ice Loss and Global Temperature Trends (1968–2024)")

    seaice_df = pd.read_csv("csv_files/cleaned_datasets/final_climate_dataset.csv")
    seaice_df.columns = seaice_df.columns.str.replace('Â', '', regex=True)

    st.write("Dataset Shape:", seaice_df.shape)

    st.subheader("📉 Total Arctic Sea Ice Area Over Time")
    fig_seaice = px.line(
        seaice_df, 
        x="Year", 
        y="Total Sea Ice Area (millions km²)",
        title="Total Arctic Sea Ice Area (1968–2024)"
    )
    st.plotly_chart(fig_seaice)

    st.subheader("🌡️ Global Temperature Change Over Time")
    fig_temp = px.line(
        seaice_df, 
        x="Year", 
        y="Temperature Change (°C)",
        title="Global Temperature Change (1968–2024)"
    )
    st.plotly_chart(fig_temp)

    st.subheader("📊 Sea Ice Area vs Global Temperature")
    fig_scatter = px.scatter(
        seaice_df,
        x="Temperature Change (°C)",
        y="Total Sea Ice Area (millions km²)",
        color="Year",
        trendline="ols",
        title="Sea Ice Area vs. Temperature Change"
    )
    st.plotly_chart(fig_scatter)

    st.subheader("🧩 Correlation Heatmap")
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(seaice_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

    st.subheader("🔎 Key Insights on Sea Ice Loss")
    st.write("""
    - Arctic Sea Ice Area shows a steady decline over the past five decades.
    - Global Temperatures have consistently increased.
    - There is a strong negative correlation between global temperature rise and Arctic sea ice loss.
    """)
# If Section is "Ice Loss Predictions"
elif section == "❄️ Ice Loss Predictions":
    st.title("🧊 Sea Ice Loss Predictions: Historical Trends & Future Outlook")

    # Load updated datasets
    historical_df = pd.read_csv("csv_files/cleaned_datasets/final_climate_dataset.csv")
    summer_predictions = pd.read_csv("csv_files/predictions_scores/summer_predictions.csv")
    september_predictions = pd.read_csv("csv_files/predictions_scores/september_predictions.csv")

    # Updated Arctic Region Coordinates
    region_coords = {
        "Foxe Basin": (66.5, -78.0),
        "Kane Basin": (79.5, -72.0),
        "Baffin Bay": (74.0, -67.5),
        "Beaufort Sea": (72.0, -140.0),
        "Canadian Arctic Archipelago": (75.0, -90.0),
        "Hudson Bay": (60.0, -85.0),
        "Hudson Strait": (62.0, -70.0),
        "Davis Strait": (65.0, -55.0),
        "Northern Labrador Sea": (62.5, -55.0),
        "Arctic Archipelago": (75.0, -90.0),
        "Arctic Domain": (75.0, -100.0),
    }

    # Tabs
    tab1, tab2, tab3 = st.tabs(["☀️ Summer Predictions", "🍁 September Predictions", "📊 Model Scores"])

    # Summer Predictions Tab
    with tab1:
        st.subheader("☀️ Summer Sea Ice Predictions")

        summer_models = summer_predictions["Model"].unique()
        selected_summer_model = st.selectbox("Select Summer Prediction Model:", summer_models)

        summer_filtered = summer_predictions[summer_predictions["Model"] == selected_summer_model].copy()

        # Clean region names
        summer_filtered["Region Clean"] = summer_filtered["Region"].str.replace(r"\(Summer\)", "", regex=True).str.strip()

        # Map coordinates
        summer_filtered["Latitude"] = summer_filtered["Region Clean"].map(lambda x: region_coords.get(x, (None, None))[0])
        summer_filtered["Longitude"] = summer_filtered["Region Clean"].map(lambda x: region_coords.get(x, (None, None))[1])

        # Convert Predictions to km²
        summer_filtered["Prediction (km²)"] = summer_filtered["Prediction"] * 1000
        summer_filtered["Bubble Size"] = summer_filtered["Prediction (km²)"].clip(lower=0)

        # Summer Map
        fig_summer_pred = px.scatter_mapbox(
            summer_filtered.dropna(subset=["Latitude", "Longitude"]),
            lat="Latitude",
            lon="Longitude",
            size="Bubble Size",
            color="Prediction (km²)",
            hover_name="Region Clean",
            animation_frame="Year",
            color_continuous_scale="Blues_r",
            size_max=40,
            zoom=2.5,
            height=650,
            title=f"Future Summer Sea Ice Predictions ({selected_summer_model})"
        )

        fig_summer_pred.update_layout(mapbox_style="carto-positron", margin={"r": 0, "t": 30, "l": 0, "b": 0})
        st.plotly_chart(fig_summer_pred)

        # Summer Ice Loss Interpretation
        st.subheader("🧊 Summer Ice Loss Interpretation and Warnings")

        critical_loss_summer = summer_filtered[(summer_filtered["Year"] == 2030) & (summer_filtered["Prediction (km²)"] < 5)]

        if not critical_loss_summer.empty:
            st.warning("⚠️ Critical Summer Sea Ice Loss Detected in 2030! Some regions are projected to almost completely lose summer sea ice.")

            for idx, row in critical_loss_summer.iterrows():
                st.write(f"🔹 **Region:** {row['Region Clean']} - **Predicted Summer Sea Ice Area in 2030:** {row['Prediction (km²)']:.2f} km²")

            st.info("""
            **Meaning of Summer Trends:**
            
            Summer trends represent the minimum sea ice extent during the warmest months. 
            A near-complete loss of summer ice means severe ecosystem stress, 
            loss of ice-dependent species, and faster warming (positive feedback loops).
            """)
        else:
            st.success("✅ No critical summer sea ice loss detected for 2030. Significant ice remains in all regions.")
    # September Predictions Tab
    with tab2:
        st.subheader("🍁 September Sea Ice Predictions")

        september_models = september_predictions["Model"].unique()
        selected_september_model = st.selectbox("Select September Prediction Model:", september_models)

        september_filtered = september_predictions[september_predictions["Model"] == selected_september_model].copy()

        # Clean region names
        september_filtered["Region Clean"] = september_filtered["Region"].str.replace(r"\(September\)", "", regex=True).str.strip()

        # Map coordinates
        september_filtered["Latitude"] = september_filtered["Region Clean"].map(lambda x: region_coords.get(x, (None, None))[0])
        september_filtered["Longitude"] = september_filtered["Region Clean"].map(lambda x: region_coords.get(x, (None, None))[1])

        # Convert Predictions to km²
        september_filtered["Prediction (km²)"] = september_filtered["Prediction"] * 1000
        september_filtered["Bubble Size"] = september_filtered["Prediction (km²)"].clip(lower=0)

        # September Map
        fig_september_pred = px.scatter_mapbox(
            september_filtered.dropna(subset=["Latitude", "Longitude"]),
            lat="Latitude",
            lon="Longitude",
            size="Bubble Size",
            color="Prediction (km²)",
            hover_name="Region Clean",
            animation_frame="Year",
            color_continuous_scale="Reds_r",
            size_max=40,
            zoom=2.5,
            height=650,
            title=f"Future September Sea Ice Predictions ({selected_september_model})"
        )

        fig_september_pred.update_layout(mapbox_style="carto-positron", margin={"r": 0, "t": 30, "l": 0, "b": 0})
        st.plotly_chart(fig_september_pred)

        # September Ice Loss Interpretation
        st.subheader("🍁 September Ice Loss Interpretation and Warnings")

        critical_loss_september = september_filtered[(september_filtered["Year"] == 2030) & (september_filtered["Prediction (km²)"] < 5)]

        if not critical_loss_september.empty:
            st.warning("⚠️ Critical September Sea Ice Loss Detected in 2030! Some regions are projected to almost completely lose September sea ice.")

            for idx, row in critical_loss_september.iterrows():
                st.write(f"🔹 **Region:** {row['Region Clean']} - **Predicted September Sea Ice Area in 2030:** {row['Prediction (km²)']:.2f} km²")

            st.info("""
            **Meaning of September Trends:**
            
            September trends show the end-of-melt-season minimum sea ice.
            A near-total loss of September ice indicates major permanent changes in the Arctic system,
            signaling severe climate tipping points.
            """)
        else:
            st.success("✅ No critical September sea ice loss detected for 2030. Significant ice remains in all regions.")

    # Model Scores Tab
    with tab3:
        st.subheader("📊 Model Performance (R² Scores)")

        st.write("### ☀️ Summer Prediction R² Scores by Region and Model")
        summer_scores = summer_predictions[["Region", "Model", "R2_Score"]].drop_duplicates()
        st.dataframe(summer_scores.sort_values(["Region", "Model"]))

        st.write("### 🍁 September Prediction R² Scores by Region and Model")
        september_scores = september_predictions[["Region", "Model", "R2_Score"]].drop_duplicates()
        st.dataframe(september_scores.sort_values(["Region", "Model"]))

        st.info("✅ A higher R² score indicates better model performance and prediction accuracy.")

# 🌡️ Temperature Predictions Section

elif section == "🌡️ Temperature Predictions":
    st.title("🌡️ Temperature Predictions: City-Wise Trends & Model Performance")

    # Load data
    temperature_predictions = pd.read_csv('csv_files/predictions_scores/temperature_predictions.csv')

    # City Coordinates (approximate)
    city_coords = {
        "MEAN_TEMPERATURE_CALGARY": (51.0447, -114.0719),
        "MEAN_TEMPERATURE_EDMONTON": (53.5461, -113.4938),
        "MEAN_TEMPERATURE_HALIFAX": (44.6488, -63.5752),
        "MEAN_TEMPERATURE_MONCTON": (46.0878, -64.7782),
        "MEAN_TEMPERATURE_MONTREAL": (45.5017, -73.5673),
        "MEAN_TEMPERATURE_OTTAWA": (45.4215, -75.6972),
        "MEAN_TEMPERATURE_QUEBEC": (46.8139, -71.2082),
        "MEAN_TEMPERATURE_SASKATOON": (52.1579, -106.6702),
        "MEAN_TEMPERATURE_STJOHNS": (47.5615, -52.7126),
        "MEAN_TEMPERATURE_TORONTO": (43.65107, -79.347015),
        "MEAN_TEMPERATURE_VANCOUVER": (49.2827, -123.1207),
        "MEAN_TEMPERATURE_WHITEHORSE": (60.7212, -135.0568),
        "MEAN_TEMPERATURE_WINNIPEG": (49.8951, -97.1384)
    }

    # Tabs
    tab1, tab2 = st.tabs(["🌡️ Future Temperature Maps", "📊 Model Scores"])

    # Future Temperature Maps
    with tab1:
        st.subheader("🌡️ Future Temperature Predictions (2025-2030)")

        # Model Selector
        temp_models = temperature_predictions["Model"].unique()
        selected_temp_model = st.selectbox("Select Temperature Prediction Model:", temp_models)

        temp_filtered = temperature_predictions[temperature_predictions["Model"] == selected_temp_model].copy()

        # Clean City Names
        temp_filtered["Latitude"] = temp_filtered["City"].map(lambda x: city_coords.get(x, (None, None))[0])
        temp_filtered["Longitude"] = temp_filtered["City"].map(lambda x: city_coords.get(x, (None, None))[1])

        # Bubble Size (clip negatives)
        temp_filtered["Bubble Size"] = temp_filtered["Prediction"].clip(lower=0)

        # Map
        fig_temp_pred = px.scatter_mapbox(
            temp_filtered.dropna(subset=["Latitude", "Longitude"]),
            lat="Latitude",
            lon="Longitude",
            size="Bubble Size",
            color="Prediction",
            hover_name="City",
            animation_frame="Year",
            color_continuous_scale="Reds_r",
            size_max=40,
            zoom=2.5,
            height=650,
            title="Predicted Future Temperatures (2025-2030)"
        )

        fig_temp_pred.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_temp_pred)

        # Optional interpretation if you want:
        st.info("""
        **Interpretation:**  
        🔥 Higher temperatures shown in darker red.  
        🔥 Increasing bubble size indicates warmer average temperatures in Canadian cities by 2030.
        """)

    # 📊 Model Scores
    with tab2:
        st.subheader("📊 Model R² Scores by City")

        temp_scores = temperature_predictions[["City", "Model", "R2_Score"]].drop_duplicates()
        st.dataframe(temp_scores.sort_values(["City", "Model"]))

        st.info("✅ A higher R² Score means better prediction accuracy for that city and model.")
# Precipitation Predictions Section

elif section == "🌧️ Precipitation Predictions":
    st.title("🌧️ Precipitation Predictions: City-Wise Trends & Model Performance")

    # Load data
    precipitation_predictions = pd.read_csv('csv_files/predictions_scores/precipitation_predictions.csv')

    # City Coordinates (approximate)
    precipitation_coords = {
        "TOTAL_PRECIPITATION_CALGARY": (51.0447, -114.0719),
        "TOTAL_PRECIPITATION_EDMONTON": (53.5461, -113.4938),
        "TOTAL_PRECIPITATION_HALIFAX": (44.6488, -63.5752),
        "TOTAL_PRECIPITATION_MONCTON": (46.0878, -64.7782),
        "TOTAL_PRECIPITATION_MONTREAL": (45.5017, -73.5673),
        "TOTAL_PRECIPITATION_OTTAWA": (45.4215, -75.6972),
        "TOTAL_PRECIPITATION_QUEBEC": (46.8139, -71.2082),
        "TOTAL_PRECIPITATION_SASKATOON": (52.1579, -106.6702),
        "TOTAL_PRECIPITATION_STJOHNS": (47.5615, -52.7126),
        "TOTAL_PRECIPITATION_TORONTO": (43.65107, -79.347015),
        "TOTAL_PRECIPITATION_VANCOUVER": (49.2827, -123.1207),
        "TOTAL_PRECIPITATION_WHITEHORSE": (60.7212, -135.0568),
        "TOTAL_PRECIPITATION_WINNIPEG": (49.8951, -97.1384)
    }

    # Tabs
    tab1, tab2 = st.tabs(["🌧️ Future Precipitation Maps", "📊 Model Scores"])
    # Future Precipitation Maps
    with tab1:
        st.subheader("🌧️ Future Total Precipitation Predictions (2025-2030)")

        # Model Selector
        precipitation_models = precipitation_predictions["Model"].unique()
        selected_precipitation_model = st.selectbox("Select Precipitation Prediction Model:", precipitation_models)

        precipitation_filtered = precipitation_predictions[precipitation_predictions["Model"] == selected_precipitation_model].copy()

        # Clean City Names
        precipitation_filtered["Latitude"] = precipitation_filtered["City"].map(lambda x: precipitation_coords.get(x, (None, None))[0])
        precipitation_filtered["Longitude"] = precipitation_filtered["City"].map(lambda x: precipitation_coords.get(x, (None, None))[1])

        # Bubble Size (clip negatives)
        precipitation_filtered["Bubble Size"] = precipitation_filtered["Prediction"].clip(lower=0)

        # Map
        fig_precipitation_pred = px.scatter_mapbox(
            precipitation_filtered.dropna(subset=["Latitude", "Longitude"]),
            lat="Latitude",
            lon="Longitude",
            size="Bubble Size",
            color="Prediction",
            hover_name="City",
            animation_frame="Year",
            color_continuous_scale="Blues",
            size_max=40,
            zoom=2.5,
            height=650,
            title="Predicted Future Precipitation (2025-2030)"
        )

        fig_precipitation_pred.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_precipitation_pred)

        st.info("""
        **Interpretation:**  
        🌧️ Larger bubbles and darker blue indicate higher predicted precipitation totals in the future.
        """)

    # Model Scores
    with tab2:
        st.subheader("📊 Model R² Scores by City")

        precipitation_scores = precipitation_predictions[["City", "Model", "R2_Score"]].drop_duplicates()
        st.dataframe(precipitation_scores.sort_values(["City", "Model"]))

        st.info("✅ A higher R² Score indicates better model prediction accuracy for each city.")
     
    # Model Scores
#CO₂ Emissions Predictions Section
elif section == "🌍 CO₂ Emissions Predictions":
    st.header("🌍 CO₂ Emissions Predictions")

    # Load CO2 emissions data
    co2_data = pd.read_csv('csv_files/cleaned_datasets/co2_cleaned.csv')
    co2_predictions = pd.read_csv('csv_files/predictions_scores/co2_predictions_with_clusters.csv')

    # Fix column names
    co2_data.columns = [col.strip() for col in co2_data.columns]
    co2_predictions.columns = [col.strip() for col in co2_predictions.columns]

    # Merge Country Names
    country_mapping = co2_data[['Country', 'Year', 'CO2_Emissions']]
    merged_predictions = co2_predictions.merge(country_mapping, on=['Country', 'Year'], how='left')

    # Model Selector
    model_selector = st.selectbox("Select Model for CO₂ Prediction", merged_predictions['Model'].unique())

    # Country Selector
    country_list = ['All'] + sorted(merged_predictions['Country'].unique())
    selected_country = st.selectbox("Select Country (optional)", country_list)

    # Filter only future years (2025-2030)
    future_predictions = merged_predictions[(merged_predictions['Year'] >= 2025) & (merged_predictions['Year'] <= 2030)]
    filtered_predictions = future_predictions[future_predictions['Model'] == model_selector]

    if selected_country != 'All':
        filtered_predictions = filtered_predictions[filtered_predictions['Country'] == selected_country]

    # Ensure positive size values
    filtered_predictions['Prediction_Positive'] = filtered_predictions['Prediction'].clip(lower=0)

    # Tabs: Map and Scores
    tab1, tab2 = st.tabs(["🌍 Map", "📈 Scores"])

    with tab1:
        fig = px.scatter_geo(
            filtered_predictions,
            locations='Country',
            locationmode='country names',
            color='Prediction',
            hover_name='Country',
            size='Prediction_Positive',
            size_max=20,
            animation_frame='Year',
            projection='natural earth',
            color_continuous_scale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
            range_color=[0, filtered_predictions['Prediction_Positive'].max()],
            title=f"CO₂ Emissions Prediction by Country ({model_selector}) (2025-2030)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📈 Model R2 Scores by Country")
        r2_scores = merged_predictions[['Country', 'Model', 'R2_Score']].drop_duplicates()
        r2_scores = r2_scores[r2_scores['Model'] == model_selector]  # Filter based on selected model
        st.dataframe(r2_scores.reset_index(drop=True))

    # Global CO2 Trend Line
    st.subheader("📊 Global Average CO₂ Emissions Trend (Historical + Predicted)")

    # Prepare global average
    global_trend = merged_predictions.groupby('Year')['Prediction'].mean().reset_index()

    fig_trend = px.line(
        global_trend,
        x='Year',
        y='Prediction',
        title="Global Average CO₂ Emissions Over Time",
        labels={'Prediction': 'Average CO₂ Emissions (tons per capita)', 'Year': 'Year'},
    )
    fig_trend.update_traces(mode="lines+markers")
    fig_trend.update_layout(yaxis_title="Average CO₂ Emissions", xaxis_title="Year")
    st.plotly_chart(fig_trend, use_container_width=True)



