import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def detect_date_column(df):
    possible_names = ['call_time', 'timestamp', 'date', 'datetime', 'time', 'activation date']
    for col in df.columns:
        if col.lower().strip() in possible_names:
            return col
    return None

def detect_facility_column(df):
    possible_names = ['origin name', 'facility', 'location name']
    for col in df.columns:
        if col.lower().strip() in possible_names:
            return col
    return None

def forecast_citywide(df, date_col, periods=30):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    daily_calls = df.groupby(df[date_col].dt.date).size().reset_index(name='y')
    daily_calls.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(daily_calls)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return {
        "daily_calls": daily_calls,
        "forecast": forecast,
        "cutoff": pd.to_datetime(daily_calls['ds'].max())
    }

def forecast_by_facility(df, date_col, facility_col, periods=120):
    df = df.dropna(subset=[facility_col])
    df['facility'] = df[facility_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    facility_daily = df.groupby([df[date_col].dt.date, 'facility']).size().reset_index(name='y')
    facility_daily.columns = ['ds', 'facility', 'y']
    
    facility_totals = {}
    facilities = facility_daily['facility'].unique()

    for fac in facilities:
        fac_df = facility_daily[facility_daily['facility'] == fac][['ds', 'y']]
        if len(fac_df) < 5:
            continue
        model = Prophet()
        model.fit(fac_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        facility_totals[fac] = forecast.tail(periods)['yhat'].sum()

    total = sum(facility_totals.values())
    facility_probs = {k: (v / total) * 100 for k, v in facility_totals.items()}
    
    return facility_probs
