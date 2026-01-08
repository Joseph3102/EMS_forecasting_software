import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
import numpy as np
import streamlit_authenticator as stauth
import hashlib
import os
from wastewater_data_retrieval import get_texas_data, DATA_URL2, DATA_URL1, DATA_URL3

Category_ranking = {
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5
}

multiplier_map = {
    1: 0.95,
    2: 1.00,
    3: 1.05,
    4: 1.10,
    5: 1.20
}


def texas_latest_score(df):
    df = df.copy()
    df["Week_Ending_Date"] = pd.to_datetime(df["Week_Ending_Date"])
    latest_week = df["Week_Ending_Date"].max()
    latest = df[df["Week_Ending_Date"] == latest_week]

    scores = latest["WVAL_Category"].map(Category_ranking)
    return round(scores.mean())




if "historical_df" not in st.session_state:
    st.session_state["historical_df"] = pd.DataFrame(columns=["call_time", "facility"])

if "uploaded_hashes" not in st.session_state:
    st.session_state["uploaded_hashes"] = set()


def get_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()


names = ["Joseph", "Admin"]
usernames = ["joseph", "admin"]


passwords = ["$2b$12$dFwPQuX2Oe8n5o4ba2ZydOVyx5WdYcGjgNnwlAaLYQXL/kfG3lESi", "$2b$12$dFwPQuX2Oe8n5o4ba2ZydOVyx5WdYcGjgNnwlAaLYQXL/kfG3lESi"]


credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": passwords[i]
        } for i in range(len(usernames))
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "forecast_cookie",
    "forecast_key",
    cookie_expiry_days=7
)

name, auth_status, username = authenticator.login(
    fields={"Form name": "Login"},
    location="sidebar"
)




if auth_status is False:
    st.error("Incorrect username or password")
    st.stop()

if auth_status is None:
    st.warning("Please enter your username and password")
    st.stop()


if auth_status:
    
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"{name}")

    safe_username = str(username)
    user_file = f"historical_calls_{safe_username}.parquet"

    # Load saved historical data if it exists
    if os.path.exists(user_file):
        st.session_state["historical_df"] = pd.read_parquet(user_file)
    else:
        st.session_state["historical_df"] = pd.DataFrame(columns=["call_time", "facility"])


    st.title("NextCall Analytics")

st.header("Historical Call Volume Dashboard")

hist_df = st.session_state["historical_df"]

if hist_df.empty:
    st.info("No historical data available.")
else:
    daily = hist_df.groupby(hist_df["call_time"].dt.date).size().reset_index(name="calls")
    daily.rename(columns={daily.columns[0]: "date"}, inplace=True)

    st.subheader("Daily Call Volume (Historical)")
    st.line_chart(daily.set_index("date"))

    st.subheader("Raw Historical Data")
    st.dataframe(hist_df)



    # 1. Allow multiple file uploads
    uploaded_files = st.file_uploader(
    "Upload one or more CSV or Excel files", 
    type=['csv', 'xlsx'],
    accept_multiple_files=True
)


    # We will store all the dataframes in a list
    all_dfs = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_hash = get_file_hash(uploaded_file)
            if file_hash in st.session_state["uploaded_hashes"]:
                st.warning(f"{uploaded_file.name} has already been uploaded. Skipping.")
                continue
            else:
                st.session_state["uploaded_hashes"].add(file_hash)

            # If file ends with .xlsx, read Excel 
            if uploaded_file.name.lower().endswith('.xlsx') or uploaded_file.name.lower().endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)



            # Try to automatically detect a date/time column
            possible_names = ['call_time', 'timestamp', 'date', 'datetime', 'time', 'activation date']
            possible_facility_columns = ['origin name', 'facility', 'location name']
            found = None
            found2 = None

            for col in df.columns:
                if col.lower().strip() in possible_names:
                    found = col
                    break

            for col in df.columns:
                if col.lower().strip() in possible_facility_columns:
                    found2 = col
                    break

            if not found:
                st.error(f"File {uploaded_file.name}: Could not find a valid date column. Please rename your date column to 'call_time'.")
                continue  # Skip this file

            # Convert and clean date column
            df['call_time'] = pd.to_datetime(df[found], errors='coerce')
            df = df.dropna(subset=['call_time'])

            # Clean facility/location if found
            if found2:
                df['facility'] = df[found2].astype(str).str.strip()
            else:
                df['facility'] = None  # placeholder if no facility column

            all_dfs.append(df[['call_time', 'facility']])

            clean_df = df[['call_time', 'facility']]
            combined = pd.concat([st.session_state["historical_df"], clean_df], ignore_index=True)
            combined.drop_duplicates(subset=["call_time", "facility"], inplace=True)
            combined.sort_values("call_time", inplace=True)
            st.session_state["historical_df"] = combined

            st.session_state["historical_df"].to_parquet(user_file)



      

            # 2. Aggregate all uploaded data
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # 3. Group by date (daily calls)
            daily_calls = combined_df.groupby(combined_df['call_time'].dt.date).size().reset_index(name='y')
            daily_calls.rename(columns={'call_time': 'ds'}, inplace=True)
            daily_calls.rename(columns={daily_calls.columns[0]: 'ds'}, inplace=True)
            daily_calls['ds'] = pd.to_datetime(daily_calls['ds'])

            st.subheader("Aggregated Daily EMS Calls")
            st.write(daily_calls)

            model = Prophet()
            model.fit(daily_calls)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            cutoff_date = daily_calls['ds'].max()
            future_only = forecast[forecast['ds'] > cutoff_date]



            st.subheader("Houston Wastewater Signals")




            with st.spinner("Downloading latest wastewater data from CDC..."):
                texas_covid = get_texas_data(DATA_URL2)
                texas_flu = get_texas_data(DATA_URL1)
                texas_rsv = get_texas_data(DATA_URL3)

            with st.spinner("Loading rankings for prediction..."):
                covid_score = texas_latest_score(texas_covid)
                flu_score = texas_latest_score(texas_flu)
                rsv_score = texas_latest_score(texas_rsv)


            st.success("Wastewater data updated")

            st.write("COVID-19 Wastewater Data (Houston)")
            st.dataframe(texas_covid.head())

            st.write("Flu Wastewater Data (Houston)")
            st.dataframe(texas_flu.head())

            st.write("RSV Wastewater Data (Houston)")
            st.dataframe(texas_rsv.head())


            


            combined_score = round((covid_score + flu_score + rsv_score) / 3)
            multiplier = multiplier_map[combined_score]
            future_only["adjusted_yhat"] = future_only["yhat"] * multiplier



            plt.figure(figsize=(10, 6))
            plt.plot(future_only['ds'], future_only['adjusted_yhat'], label='Wastewater-Adjusted Forecast')
            plt.fill_between(future_only['ds'], future_only['yhat_lower'], future_only['yhat_upper'], alpha=0.3)
            plt.title("Forecasted EMS Calls (Next 30 Days)")
            plt.xlabel("Date")
            plt.ylabel("Predicted Calls")
            plt.legend()
            st.pyplot(plt)


            start = future_only.iloc[0]['yhat']
            end = future_only.iloc[-1]['yhat']
            growth_rate = ((end - start) / start) * 100
            trend = "increase" if growth_rate > 0 else "decrease"
            st.write(f"Call volume is expected to {trend} by **{abs(growth_rate):.2f}%** over the next 30 days.")

            avg_calls = future_only['yhat'].mean()
            st.write(f"Estimated Average Call Volume: {avg_calls:.2f}")

            max_volume = future_only['yhat'].max()
            threshold = max_volume * 0.9
            peak_days_range = future_only[future_only['yhat'] >= threshold]
            start_date = peak_days_range['ds'].min().date()
            end_date = peak_days_range['ds'].max().date()

            st.write(
                f"Peak call volume is expected between **{start_date}** and **{end_date}**, "
                f"with call volumes near the maximum of approximately **{max_volume:.0f} calls** per day."
            )


            if combined_df['facility'].notnull().all():
                st.subheader("Likelihood of Calls by Facility (Next 30 Days)")

                facility_daily = combined_df.groupby(
                    [combined_df['call_time'].dt.date, 'facility']
                ).size().reset_index(name='y')
                facility_daily.rename(columns={'call_time': 'ds'}, inplace=True)
                facility_daily.rename(columns={facility_daily.columns[0]: 'ds'}, inplace=True)
                facility_daily['ds'] = pd.to_datetime(facility_daily['ds'])

                facility_totals = {}
                facilities = facility_daily['facility'].unique()

                for fac in facilities:
                    fac_df = facility_daily[facility_daily['facility'] == fac][['ds', 'y']]
                    if len(fac_df) < 5:
                        continue  # skip if too little data
                    fac_model = Prophet()
                    fac_model.fit(fac_df)
                    fac_future = fac_model.make_future_dataframe(periods=30)
                    fac_forecast = fac_model.predict(fac_future)
                    facility_totals[fac] = fac_forecast.tail(30)['yhat'].sum()

                total = sum(facility_totals.values())
                if total > 0:
                    facility_probs = {k: (v / total) * 100 for k, v in facility_totals.items()}
                    prob_df = pd.DataFrame(facility_probs.items(), columns=['Facility', 'Likelihood (%)'])
                    prob_df = prob_df.sort_values(by='Likelihood (%)', ascending=False)
                    st.write(prob_df)
                    st.bar_chart(prob_df.set_index('Facility'))
                else:
                    st.warning("Not enough facility data for forecasting.")
            else:
                st.warning("No facility/location data available for facility-based forecasting.")

  
            st.subheader("5-Day Weather-Adjusted Forecast (Houston)")


            lat, lon = 29.7604, -95.3698  
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&daily=precipitation_sum,snowfall_sum,temperature_2m_min,temperature_2m_max"
                f"&timezone=America/Chicago"
            )

            try:
                weather_data = requests.get(url).json()
                weather_df = pd.DataFrame(weather_data["daily"])
                weather_df["time"] = pd.to_datetime(weather_df["time"])
                weather_df.rename(columns={"time": "ds"}, inplace=True)
            except Exception as e:
                st.error(f"Error fetching weather data: {e}")
                weather_df = pd.DataFrame(columns=["ds", "temperature_2m_min", "precipitation_sum", "snowfall_sum"])


            if "ds" not in forecast.columns:
                st.error("Forecast data missing 'ds' column. Cannot perform weather-adjusted merge.")
            else:
                five_day_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5).copy()

                if "ds" not in weather_df.columns:
                    st.error("Weather data missing 'ds' column. Cannot perform merge.")
                else:
                    merged = pd.merge(five_day_forecast, weather_df, on="ds", how="left")


                    def adjust_call_volume(row):
                        temp_min = row.get("temperature_2m_min", np.nan)
                        rain = row.get("precipitation_sum", np.nan)
                        snow = row.get("snowfall_sum", np.nan)

                        if pd.isna(temp_min) or pd.isna(rain) or pd.isna(snow):
                            return row["yhat"]


                        if snow > 0:
                            return row["yhat"] * 0.3
                        # Extreme cold (< 2°C / 35°F)
                        elif temp_min < 2:
                            return row["yhat"] * 0.8
                        # Heavy rain (> 10 mm/day)
                        elif rain > 10:
                            return row["yhat"] * 0.7
                        # Light rain (1–10 mm/day)
                        elif rain > 0:
                            return row["yhat"] * 0.9
                        # Normal or hot
                        else:
                            return row["yhat"]

                    merged["adjusted_yhat"] = merged.apply(adjust_call_volume, axis=1)


                    plt.figure(figsize=(10, 6))
                    plt.plot(merged["ds"], merged["yhat"], label="Baseline Forecast")
                    plt.plot(merged["ds"], merged["adjusted_yhat"], label="Weather-Adjusted Forecast", linestyle="--")
                    plt.title("Next 5 Days - Weather Adjusted EMS Forecast")
                    plt.xlabel("Date")
                    plt.ylabel("Predicted Calls")
                    plt.legend()
                    st.pyplot(plt)

                    st.write(
                        merged[
                            ["ds", "yhat", "adjusted_yhat", "temperature_2m_min", "precipitation_sum", "snowfall_sum"]
                        ]
                    )

           

            
                    

        


