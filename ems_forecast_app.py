import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("EMS Call Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV with a 'call_time' column", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure the column exists
    if 'call_time' not in df.columns:
        st.error("CSV must contain a 'call_time' column.")
    else:
        df['call_time'] = pd.to_datetime(df['call_time'])
        daily_calls = df.groupby(df['call_time'].dt.date).size().reset_index(name='y')
        daily_calls.columns = ['ds', 'y']

        # Show data
        st.subheader("Daily EMS Calls")
        st.write(daily_calls)

        # Forecast
        model = Prophet()
        model.fit(daily_calls)

        future = model.make_future_dataframe(periods=120)
        forecast = model.predict(future)

        st.subheader("Forecasted EMS Calls (120 Days Ahead)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
