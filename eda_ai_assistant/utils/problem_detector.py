import pandas as pd

def detect_problem_type(df, target_column):
    target = df[target_column]
    unique_values = target.nunique()

    # for numeric
    if pd.api.types.is_numeric_dtype(target):
        if unique_values <= 10:
            return "Classification", "Numeric target with few unique values"
        else:
            return "Regression", "Numeric target with many unique values"
    
    # for categorical
    else:
        if unique_values == 2:
            return "Binary Classification", "Target has two categories"
        elif unique_values > 2:
            return "Multi-class Classification", "Target has multiple categories"
    
    # will use agents here to determine the problem
    return "Unknown", "Unable to determine the problem type"