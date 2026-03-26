import pandas as pd
import numpy as np

def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr 
    upper = q3 + 1.5 * iqr 
    outliers = series[(series < lower) | (series > upper)] 
    return len(outliers), lower, upper

def detect_outliers_zscore(series, threshold = 3):
    mean = series.mean()
    std = series.std()

    if std == 0:
        return 0
    
    z_scores = (series - mean) / std
    outliers = series[np.abs(z_scores) > threshold]

    return len(outliers)

def analyze_outliers(df):
    numeric_cols = df.select_dtypes(include = np.number).columns
    result = []

    for col in numeric_cols:
        clean = df[col].dropna()

        if len(clean) == 0:
            continue

        iqr_outliers, lower, upper = detect_outliers_iqr(clean)
        z_outliers = detect_outliers_zscore(clean)

        result.append({
            "column": col,
            "iqr_outliers": iqr_outliers,
            "zscore_outliers": z_outliers,
            "lower_bound": lower,
            "upper_bound": upper
        })
    return result
