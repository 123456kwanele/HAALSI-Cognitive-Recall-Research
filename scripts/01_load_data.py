import os
import pandas as pd

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DTA_FILE_PATH = os.path.join(DATA_DIR, "Longitudinal Public Release Dataset V5.dta")

def load_haalsi_dataset():
    print("Loading HAALSI Stata dataset (this might take a moment as the file is large)...")
    try:
        # read_stata natively loads .dta files. 
        # convert_categoricals=False keeps the raw numerical codes, which are easier to filter.
        df = pd.read_stata(DTA_FILE_PATH, convert_categoricals=False)
        print(f"Dataset successfully loaded! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find the file at {DTA_FILE_PATH}.")
        print("Please ensure your .dta file is named correctly and placed in the data/ folder.")
        return None

if __name__ == "__main__":
    df = load_haalsi_dataset()
    if df is not None:
        # Print out the first few columns to verify the structure
        print("\n--- Quick Column Check (First 10 columns) ---")
        print(df.columns[:10].tolist())