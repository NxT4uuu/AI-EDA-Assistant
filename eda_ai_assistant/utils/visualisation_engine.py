import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig

def plt_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    ax.set_title("Correlation Heatmap")
    return fig

def plot_catergorical_distribution(df, column):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Category Frequency: {column}")
    return fig

def plot_feature_vs_target(df, feature, target): 
    fig, ax = plt.subplots() 
    if df[target].dtype == "object": 
        sns.boxplot(x=df[target], y=df[feature], ax=ax)
    else: 
        sns.scatterplot(x=df[feature], y=df[target], ax=ax)
    
    ax.set_title(f"{feature} vs {target}") 
    return fig