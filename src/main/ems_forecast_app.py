import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("NextCall Analytics")

# 1. Allow multiple file uploads
uploaded_files = st.file_uploader(
    "Upload one or more CSVs with date/time of EMS calls", type='csv', accept_multiple_files=True
)

# We will store all the dataframes in a list
all_dfs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
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

        plt.figure(figsize=(10, 6))
        plt.plot(future_only['ds'], future_only['yhat'], label='Forecast')
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

