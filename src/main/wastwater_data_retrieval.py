import os
import requests
from datetime import datetime



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


def auto_update(url=DATA_URL):
    """
    Only download once per day.
    If today's file exists, reuse it.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    files_today = [f for f in os.listdir(SAVE_DIR) if today in f]

    # If today's file exists → return it
    if files_today:
        return os.path.join(SAVE_DIR, files_today[0])

    # Otherwise download a fresh one
    filename = f"wastewater_{today}.csv"
    return download_file(url, filename)


def load_csv(path):
    """Load a CSV file."""
    return pd.read_csv(path)


def filter_houston(df):
    """
    Extract Houston/Harris County-specific data.
    We will update column names once we see the real dataset fields.
    """
    possible_columns = ["county", "location", "site", "sewershed_name"]

    loc_col = None
    for col in df.columns:
        if col.lower().strip() in possible_columns:
            loc_col = col
            break

    if loc_col is None:
        raise ValueError("No location column found for filtering Houston.")

    houston_df = df[
        df[loc_col].str.contains("Houston", case=False, na=False) |
        df[loc_col].str.contains("Harris", case=False, na=False)
    ]

    return houston_df


def get_houston_data():
    """
    Main function that:
    - Downloads or loads the freshest dataset
    - Reads it
    - Returns only Houston rows
    """

    path = auto_update()
    df = load_csv(path)
    houston_df = filter_houston(df)
    return houston_df
