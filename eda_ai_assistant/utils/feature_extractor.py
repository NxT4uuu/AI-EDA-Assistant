import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def detect_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    else:
        return "categorical"
    
def extract_dataset_features(df):
    dataset_features = {}

    dataset_features["num_rows"] = df.shape[0]
    dataset_features["num_columns"] = df.shape[1]

    total_cells = df.size
    missing_cells = df.isnull().sum().sum()

    dataset_features["missing_percentage"] = (missing_cells / total_cells) * 100

    dataset_features["duplicate_rows"] = df.duplicated().sum()

    numeric_cols = df.select_dtypes(include= np.number).columns
    categorical_cols = df.select_dtypes(exclude= np.number).columns

    dataset_features["numeric_columns"] = len(numeric_cols)
    dataset_features["categorical_columns"] = len(categorical_cols)

    return dataset_features

def extract_column_features(df):
    column_features = []

    for col in df.columns:
        series = df[col]
        col_data = {}

        col_data["column_name"] = col
        col_data["type"] = detect_column_type(series)
        col_data["missing_percentage"] = (series.isnull().sum() / len(series)) * 100
        col_data["unique_values"] = series.nunique()

        if col_data["type"] == "numeric":
            clean_series = series.dropna()

            col_data["mean"] = clean_series.mean()
            col_data["median"] = clean_series.median()
            col_data["std"] = clean_series.std()
            col_data["min"] = clean_series.min()
            col_data["max"] = clean_series.max()
            col_data["variance"] = clean_series.var()

            if len(clean_series) > 0:
                col_data["skewness"] = skew(clean_series)
                col_data["kurtosis"] = kurtosis(clean_series)
            else:
                col_data["skewness"] = None
                col_data["kurtosis"] = None

        column_features.append(col_data)
    return column_features
