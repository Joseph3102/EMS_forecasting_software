import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
import numpy as np
import streamlit_authenticator as stauth




names = ["Joseph", "Admin"]
usernames = ["joseph", "admin"]

# Passwords should be hashed using bcrypt
passwords = ["$2b$12$dFwPQuX2Oe8n5o4ba2ZydOVyx5WdYcGjgNnwlAaLYQXL/kfG3lESi", "$2b$12$dFwPQuX2Oe8n5o4ba2ZydOVyx5WdYcGjgNnwlAaLYQXL/kfG3lESi"]
# I will show how to generate these below.

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




    st.title("NextCall Analytics")

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

        if len(all_dfs) == 0:
            st.warning("No valid data to process after checking uploaded files.")
        else:
            # 2. Aggregate all uploaded data
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # 3. Group by date (daily calls)
            daily_calls = combined_df.groupby(combined_df['call_time'].dt.date).size().reset_index(name='y')
            daily_calls.rename(columns={'call_time': 'ds'}, inplace=True)
            daily_calls.rename(columns={daily_calls.columns[0]: 'ds'}, inplace=True)
            daily_calls['ds'] = pd.to_datetime(daily_calls['ds'])

            st.subheader("Aggregated Daily EMS Calls")
            st.write(daily_calls)

            # 4. Train Prophet model on all combined data
            model = Prophet()
            model.fit(daily_calls)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            cutoff_date = daily_calls['ds'].max()
            future_only = forecast[forecast['ds'] > cutoff_date]

            # Plot 30-day baseline forecast
            plt.figure(figsize=(10, 6))
            plt.plot(future_only['ds'], future_only['yhat'], label='Forecast')
            plt.fill_between(future_only['ds'], future_only['yhat_lower'], future_only['yhat_upper'], alpha=0.3)
            plt.title("Forecasted EMS Calls (Next 30 Days)")
            plt.xlabel("Date")
            plt.ylabel("Predicted Calls")
            plt.legend()
            st.pyplot(plt)

            # Trend info
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

            # 5. Facility-based forecasting if available
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

            # 6. Weather-adjusted short-term forecast (Houston)
            st.subheader("5-Day Weather-Adjusted Forecast (Houston)")

            # Get Houston daily weather forecast (Open-Meteo)
            lat, lon = 29.7604, -95.3698  # Houston
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

            # Ensure forecast has ds
            if "ds" not in forecast.columns:
                st.error("Forecast data missing 'ds' column. Cannot perform weather-adjusted merge.")
            else:
                five_day_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5).copy()

                if "ds" not in weather_df.columns:
                    st.error("Weather data missing 'ds' column. Cannot perform merge.")
                else:
                    merged = pd.merge(five_day_forecast, weather_df, on="ds", how="left")

                    # Apply weather adjustment rules
                    def adjust_call_volume(row):
                        temp_min = row.get("temperature_2m_min", np.nan)
                        rain = row.get("precipitation_sum", np.nan)
                        snow = row.get("snowfall_sum", np.nan)

                        if pd.isna(temp_min) or pd.isna(rain) or pd.isna(snow):
                            return row["yhat"]

                        # Any snow = significantly fewer calls
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

                    # Plot comparison between normal and weather-adjusted forecasts
                    plt.figure(figsize=(10, 6))
                    plt.plot(merged["ds"], merged["yhat"], label="Baseline Forecast")
                    plt.plot(merged["ds"], merged["adjusted_yhat"], label="Weather-Adjusted Forecast", linestyle="--")
                    plt.title("Next 5 Days - Weather Adjusted EMS Forecast")
                    plt.xlabel("Date")
                    plt.ylabel("Predicted Calls")
                    plt.legend()
                    st.pyplot(plt)

                    # Show numeric values in a table
                    st.write(
                        merged[
                            ["ds", "yhat", "adjusted_yhat", "temperature_2m_min", "precipitation_sum", "snowfall_sum"]
                        ]
                    )

        


