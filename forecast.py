import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Joseph\Desktop\Projects\EMS-ForecastingModel")

print("Current working directory:", os.getcwd())

# Load your call data (replace with your file later)
df = pd.read_csv('sample_calls.csv')  # Format: call_time column
df['call_time'] = pd.to_datetime(df['call_time'])
daily_calls = df.groupby(df['call_time'].dt.date).size().reset_index()
daily_calls.columns = ['ds', 'y']  # Prophet needs columns named ds (date) and y (value)

# Build and fit the model
model = Prophet()
model.fit(daily_calls)

# Forecast 30 days into the future
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.show()
