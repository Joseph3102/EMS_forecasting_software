import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("EMS Call Forecasting App")

uploaded_file = st.file_uploader("Upload a CSV with a column containing date/time of EMS calls", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Step 1: Try to automatically detect a date/time column
    possible_names = ['call_time', 'timestamp', 'date', 'datetime', 'time', 'activation date']
    possible_facility_columns = ['origin name','facility','location name']
    found = None
    found2 = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in possible_names:
            found = col
            break
        
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in possible_facility_columns:
            found2 = col
            break

    if not found:
        st.error("Could not find a valid date column. Please rename your date column to 'call_time'.")
    else:
        df['call_time'] = pd.to_datetime(df[found], errors='coerce')  # convert to datetime
        df = df.dropna(subset=[found])  # drop any invalid timestamps

        daily_calls = df.groupby(df['call_time'].dt.date).size().reset_index(name='y')
        daily_calls.columns = ['ds', 'y']
        
        st.subheader("Daily EMS Calls")
        st.write(daily_calls)
        
        model = Prophet()
        model.fit(daily_calls)

        future = model.make_future_dataframe(periods=120)
        forecast = model.predict(future)

        st.subheader("Forecasted EMS Calls (Next 120 Days)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

    
    if not found2:
        st.warning("Could not find a valid facility/location column. Some features like facility call prediction will be skipped.")
    else:
        # Clean the location data
        df = df.dropna(subset=[found2])
        df['facility'] = df[found2].astype(str).str.strip()

        st.subheader("Likelihood of Calls by Facility (Next 120 Days)")

    # Step 1: Count daily calls by facility
        facility_daily = df.groupby([df['call_time'].dt.date, 'facility']).size().reset_index(name='y')
        facility_daily.columns = ['ds', 'facility', 'y']

        # Step 2: Forecast future calls for each facility
        facility_totals = {}
        facilities = facility_daily['facility'].unique()

        for fac in facilities:
            fac_df = facility_daily[facility_daily['facility'] == fac][['ds', 'y']]
            if len(fac_df) < 5:  # skip facilities with too little data
                continue
            model = Prophet()
            model.fit(fac_df)
            future = model.make_future_dataframe(periods=120)
            forecast = model.predict(future)
            forecast_sum = forecast.tail(120)['yhat'].sum()
            facility_totals[fac] = forecast_sum

        # Step 3: Calculate probabilities
        total = sum(facility_totals.values())
        facility_probs = {k: (v / total) * 100 for k, v in facility_totals.items()}

        # Step 4: Display result
        prob_df = pd.DataFrame(list(facility_probs.items()), columns=['Facility', 'Likelihood (%)'])
        prob_df = prob_df.sort_values(by='Likelihood (%)', ascending=False)

        st.write(prob_df)
        st.bar_chart(prob_df.set_index('Facility'))

        