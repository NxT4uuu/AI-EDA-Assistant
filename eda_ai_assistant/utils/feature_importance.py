import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def compute_feature_importance(df, target_column, problem_type):

    df_clean = df.dropna()

    X = df_clean.drop(columns = [target_column])
    y = df_clean[target_column]

    # Encode Categorical columns
    X = pd.get_dummies(X)

    # Encode target if Categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # choose model
    if "Regression" in problem_type:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)

    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    return importance_df