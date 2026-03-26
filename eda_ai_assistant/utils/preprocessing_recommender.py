import numpy as np
import pandas as pd
from scipy.stats import skew

def recommend_processing(df):
    
    recommendationds = []

    for col in df.columns:
        series = df[col]

        col_recs = []

        # handling missing values
        missing_ratio = series.isnull().sum() / len(series)


        if missing_ratio > 0.4:
            col_recs.append("High misssing values, consider droping this column")
        
        elif missing_ratio > 0:
            if pd.api.types.is_numeric_dtype(series):
                col_recs.append("Missing values -> use mean/median imputation")
            else:
                col_recs.append("Missing values -> use mode imputation")

        # Numeric features
        if pd.api.types.is_numeric_dtype(series):
             clean = series.dropna()
             if len(clean) > 0:
                 s = skew(clean)

                 if abs(s) > 1:
                     col_recs.append("Highly skewed distribution -> apply log transformation")
                 elif abs(s) > 0.5:
                     col_recs.append("Moderately skewed -> consider Box-Cox transformation")
                 
                #  check variance
                 variance = clean.var()
                 if variance > 1000:
                     col_recs.append("High variance -> apply feature scaling (standardization)")
        
        # Categorical features
        else:
            unique_vals = series.nunique()
            if unique_vals <= 10:
                col_recs.append("Low cardinality categorical -> use one-hot encoding")
            elif unique_vals <= 50:
                col_recs.append("Moderate cardinality -> use label encoding")
            else:
                col_recs.append("High cardinality -> consider target encoding")
        
        recommendationds.append({
            "column": col,
            "recommendations": " | ".join(col_recs)
        })
    return recommendationds