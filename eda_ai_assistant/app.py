import streamlit as st
import pandas as pd
from utils.data_loader import load_csv, validate_dataframe
from utils.feature_extractor import extract_dataset_features, extract_column_features
from utils.data_quality import evaluate_data_quality
import seaborn as sns
import matplotlib.pyplot as plt
from utils.outlier_detection import analyze_outliers
from utils.preprocessing_recommender import recommend_processing
from utils.problem_detector import detect_problem_type
from utils.feature_importance import compute_feature_importance
from utils.visualisation_engine import plot_histogram, plot_feature_vs_target, plot_catergorical_distribution, plt_correlation_heatmap
from models.insight_genrator import EDAInsightGenerator
from utils.feature_to_text import convert_feature_to_text


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import load_csv, validate_dataframe
from utils.feature_extractor import extract_dataset_features, extract_column_features
from utils.data_quality import evaluate_data_quality
from utils.outlier_detection import analyze_outliers
from utils.preprocessing_recommender import recommend_processing
from utils.problem_detector import detect_problem_type
from utils.feature_importance import compute_feature_importance
from utils.visualisation_engine import (
    plot_histogram,
    plot_feature_vs_target,
    plot_catergorical_distribution,
    plt_correlation_heatmap
)

from models.insight_genrator import EDAInsightGenerator
from utils.feature_to_text import convert_feature_to_text


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-Powered EDA Assistant",
    layout="wide"
)

st.title("AI-Powered Exploratory Data Analysis (EDA) Assistant")
st.write("Upload a CSV dataset to begin automatic analysis.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Dataset Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

df = None
target_column = None

if uploaded_file is not None:
    df = load_csv(uploaded_file)

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        df.columns
    )

    viz_mode = st.sidebar.selectbox(
        "Visualization Mode",
        ["Basic", "Advanced"]
    )


# ---------------- MAIN DASHBOARD ----------------
if df is not None:

    try:

        checks = validate_dataframe(df)

        # ---------------- DATASET OVERVIEW ----------------
        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", checks["rows"])
        col2.metric("Columns", checks["columns"])
        col3.metric("Duplicate Rows", checks["duplicate_rows"])

        st.subheader("Dataset Preview")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info_df)

        # ---------------- FEATURE EXTRACTION ----------------
        st.header("Dataset Feature Extraction")

        dataset_features = extract_dataset_features(df)
        column_features = extract_column_features(df)

        column_df = pd.DataFrame(column_features)

        st.dataframe(column_df)

        # ---------------- CREATE TABS ----------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Quality",
            "Outliers",
            "Preprocessing",
            "Feature Importance",
            "Visualizations"
        ])

        # ---------------- DATA QUALITY ----------------
        with tab1:

            st.subheader("Dataset Quality Assessment")

            quality_score, issues = evaluate_data_quality(df)

            st.metric("Dataset Quality Score", f"{quality_score}/100")

            if len(issues) > 0:

                st.write("### Detected Issues")

                for issue in issues:
                    st.warning(issue)

            else:
                st.success("No major data quality issues detected")

        # ---------------- OUTLIER DETECTION ----------------
        with tab2:

            st.subheader("Outlier Detection")

            outlier_results = analyze_outliers(df)

            if len(outlier_results) == 0:
                st.info("No numeric columns available for outlier analysis")

            else:

                outlier_df = pd.DataFrame(outlier_results)

                st.write("### Outlier Summary")
                st.dataframe(outlier_df)

                numeric_cols = df.select_dtypes(include="number").columns

                selected_cols = st.selectbox(
                    "Select column for boxplot",
                    numeric_cols
                )

                fig, ax = plt.subplots()

                sns.boxplot(x=df[selected_cols], ax=ax)

                ax.set_title(f"Boxplot for {selected_cols}")

                st.pyplot(fig)

        # ---------------- PREPROCESSING ----------------
        with tab3:

            st.subheader("Automatic Preprocessing Recommendations")

            recs = recommend_processing(df)

            rec_df = pd.DataFrame(recs)

            st.dataframe(rec_df)

        # ---------------- FEATURE IMPORTANCE ----------------
        with tab4:

            st.subheader("Machine Learning Problem Detection")

            problem_type, reason = detect_problem_type(df, target_column)

            st.write("### Detected Problem Type")
            st.info(problem_type)

            st.write("Reason:", reason)

            st.subheader("Feature Importance Analysis")

            importance_df = compute_feature_importance(
                df,
                target_column,
                problem_type
            )

            top_features = importance_df.head(10)

            st.write("### Top Important Features")

            st.dataframe(top_features)

            fig, ax = plt.subplots()

            ax.barh(top_features["feature"], top_features["importance"])

            ax.invert_yaxis()

            ax.set_title("Feature Importance")

            st.pyplot(fig)

        # ---------------- VISUALIZATION ----------------
        with tab5:

            st.subheader("Data Visualization")

            numeric_cols = df.select_dtypes(include="number").columns

            categorical_cols = df.select_dtypes(exclude="number").columns

            st.write("### Numeric Distribution")

            selected_numeric = st.selectbox(
                "Select numeric column",
                numeric_cols
            )

            fig = plot_histogram(df, selected_numeric)

            st.pyplot(fig)

            st.write("### Correlation Analysis")

            heatmap_fig = plt_correlation_heatmap(df)

            st.pyplot(heatmap_fig)

            if len(categorical_cols) > 0:

                st.write("### Categorical Distribution")

                selected_cat = st.selectbox(
                    "Select categorical column",
                    categorical_cols
                )

                fig = plot_catergorical_distribution(
                    df,
                    selected_cat
                )

                st.pyplot(fig)

            st.write("### Feature vs Target Relationship")

            feature_col = st.selectbox(
                "Select feature column",
                df.columns
            )

            fig = plot_feature_vs_target(
                df,
                feature_col,
                target_column
            )

            st.pyplot(fig)

        # ---------------- AI INSIGHTS ----------------
        st.header("AI Analyst Insights")

        insight_model = EDAInsightGenerator()

        for col_feature in column_features:

            text_features = convert_feature_to_text(col_feature)

            insight = insight_model.generate_insight(text_features)

            st.markdown(f"### Column: {col_feature['column_name']}")

            st.write(insight)

    except Exception as e:

        st.error(f"Failed to process dataset: {e}")