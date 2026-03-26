import pandas as pd
import numpy as np

def detect_outliers_iqr(series):

    q1 = series.quantile(0.25) 
    q3 = series.quantile(0.75) # 1/3rd

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]
    
    return len(outliers)

def evaluate_data_quality(df):
    score = 100
    issues = []

    total_cells = df.size
    missing_cells = df.isnull().sum().sum()

    missing_ratio = missing_cells / total_cells

    if missing_ratio > 0.2 :
        score -= 25
        issues.append("High percentage of missing values in dataset")
    
    elif missing_ratio > 0.1:
        score -= 10
        issues.append("Moderate missing values detected")

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        score -= min(duplicate_rows * 2, 15)
        issues.append(f"{duplicate_rows} duplicate rows detected")
    
    numeric_cols = df.select_dtypes(include = np.number).columns

    for col in numeric_cols:
        outlier_count = detect_outliers_iqr(df[col].dropna())

        if outlier_count > 0:
            score -= 2
            issues.append(f"Outliers detected in column: {col}")

    categorical_cols = df.select_dtypes(exclude = np.number).columns

    for col in categorical_cols:
        counts = df[col].value_counts(normalize = True)
        if len(counts) > 0 and counts.iloc[0] > 0.9:
            score -= 5
            issues.append(f"Column {col} is highly imbalanced")
    
    score = max(score, 0)

    return score, issues