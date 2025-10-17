import requests
import pandas as pd

# Houston, TX coordinates
lat, lon = 29.7604, -95.3698  

# Request daily weather totals
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={lat}&longitude={lon}"
    f"&daily=precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min"
    f"&timezone=America/Chicago"
)

response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data["daily"])

# Convert the time column to datetime
df["time"] = pd.to_datetime(df["time"])

print(df.head())
