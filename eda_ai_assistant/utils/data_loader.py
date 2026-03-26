import pandas as pd

def load_csv(file):
    try:
        # Load CSV file safely and return dataframe. Raises exception if file is invalid.
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

def validate_dataframe(df):
    # Perform basic validation on dataframe. Returns a dictionary of checks.
    checks = {
        "is_empty": df.empty,
        "has_missing": df.isnull().sum().sum() > 0,
        "duplicate_rows": df.duplicated().sum(),
        "rows": df.shape[0],
        "columns": df.shape[1],
    }
    return checks
