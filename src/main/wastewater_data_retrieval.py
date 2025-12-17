import os
import requests
from datetime import datetime
import pandas as pd



# !!! Incomplete & out of format



# Folder where downloaded CSVs will be stored
SAVE_DIR = "wastewater_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

# CDC Wastewater data URL (replace with real link once you decide)
DATA_URL1 = "https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/FluA/nwssfluastateactivitylevelDL.csv"
DATA_URL2 = "https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/SC2/nwsssc2stateactivitylevelDL.csv"
DATA_URL3 = "https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/rsv/nwssrsvstateactivitylevel.csv"



def download_file(url, filename=None):
    """Download wastewater CSV file from CDC."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wastewater_{timestamp}.csv"

    path = os.path.join(SAVE_DIR, filename)

    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    return path





def load_csv(path):
    """Load a CSV file."""
    return pd.read_csv(path)


def filter_texas(df):
    
    possible_columns = ["county", "location", "site", "sewershed_name", "state/territory"]

    loc_col = None
    for col in df.columns:
        if col.lower().strip() in possible_columns:
            loc_col = col
            break

    if loc_col is None:
        raise ValueError("No location column found for filtering Texas/Houston.")

    texas_df = df[
        df[loc_col].str.contains("Texas", case=False, na=False) |
        df[loc_col].str.contains("TX", case=False, na=False)
    ]

    return texas_df


def get_texas_data(url):


    path = download_file(url)
    df = load_csv(path)
    texas_df = filter_texas(df)
    return texas_df
