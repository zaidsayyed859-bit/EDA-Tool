import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# PAGE CONFIGURATION
st.set_page_config(
    page_title="🧹 Data Cleaning & Analysis Tool",
    page_icon="🧹",
    layout="wide"
)

# CONSTANTS

INVALID_VALUES = [
    "ERROR",
    "UNKNOWN",
    "N/A",
    "NA",
    "NULL",
    "NONE",
    "INVALID",
    "-"
]

CATEGORY_MAP = {
    "CC": "Credit Card",
    "Card": "Credit Card",
    "CREDIT": "Credit Card"
}


# FILE LOADING

def load_file(uploaded_file):

    try:

        filename = uploaded_file.name.lower()

        # ---------------- CSV ----------------

        if filename.endswith(".csv"):

            try:
                uploaded_file.seek(0)

                return pd.read_csv(
                    uploaded_file,
                    encoding="utf-8"
                )

            except UnicodeDecodeError:

                uploaded_file.seek(0)

                return pd.read_csv(
                    uploaded_file,
                    encoding="latin1"
                )


        # ---------------- XLSX ----------------

        elif filename.endswith(".xlsx"):

            uploaded_file.seek(0)

            return pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )


        # ---------------- Unsupported ----------------

        else:

            st.error(
                "❌ Unsupported file format."
            )

            return None


    except Exception as e:

        st.error(
            f"❌ Failed to read file: {e}"
        )

        return None


# NUMERIC COLUMN DETECTION
def detect_numeric_columns(df, threshold=0.60):

    numeric_columns = []

    for col in df.columns:

        cleaned = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(
                r"[^\d\.\-]",
                "",
                regex=True
            )
        )

        converted = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= threshold:

            numeric_columns.append(col)

    return numeric_columns


# DATA TYPE DETECTION

def detect_best_dtype(series):

    # Already numeric
    if pd.api.types.is_numeric_dtype(series):

        return "numeric"


    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):

        return "datetime"


    non_null = series.dropna()

    if len(non_null) == 0:

        return "unknown"


    # Numeric detection

    numeric_ratio = pd.to_numeric(
        non_null,
        errors="coerce"
    ).notna().mean()

    if numeric_ratio >= 0.90:

        return "numeric"


    # Date detection

    date_ratio = pd.to_datetime(
        non_null,
        errors="coerce"
    ).notna().mean()

    if date_ratio >= 0.90:

        return "datetime"


    # Category detection

    category_ratio = (
        non_null.nunique() /
        len(non_null)
    )

    if category_ratio < 0.05:

        return "category"


    return "text"


# QUALITY REPORT

def quality_report(df):

    return pd.DataFrame({

        "Column":
            df.columns,

        "Data Type":
            df.dtypes.astype(str).values,

        "Missing":
            df.isna().sum().values,

        "Missing %":
            (
                df.isna().mean() * 100
            ).round(2).values,

        "Unique":
            df.nunique(
                dropna=True
            ).values
    })


# INVALID VALUE REPORT
def invalid_value_report(df):

    results = []

    for col in df.columns:

        if (
            df[col].dtype == "object"
            or pd.api.types.is_string_dtype(
                df[col]
            )
        ):

            values = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            for invalid in INVALID_VALUES:

                count = (
                    values == invalid.upper()
                ).sum()

                if count > 0:

                    results.append({

                        "Column":
                            col,

                        "Invalid Value":
                            invalid,

                        "Count":
                            int(count)
                    })


    return pd.DataFrame(results)

# REPLACE INVALID VALUES

def replace_invalid_values(df):

    df = df.copy()

    for col in df.columns:

        if (
            df[col].dtype == "object"
            or pd.api.types.is_string_dtype(
                df[col]
            )
        ):

            normalized = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            mask = normalized.isin(
                [
                    value.upper()
                    for value in INVALID_VALUES
                ]
            )

            df.loc[mask, col] = np.nan

    return df


# OUTLIER REPORT

def outlier_report(df):

    results = []

    numeric_columns = detect_numeric_columns(df)

    for col in numeric_columns:

        series = pd.to_numeric(

            df[col]
            .astype(str)
            .str.replace(
                r"[^\d\.\-]",
                "",
                regex=True
            ),

            errors="coerce"

        ).dropna()


        if len(series) < 4:

            continue


        q1 = series.quantile(0.25)

        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr


        outliers = (
            (series < lower) |
            (series > upper)
        )


        results.append({

            "Column":
                col,

            "Outliers":
                int(outliers.sum()),

            "Lower Bound":
                round(lower, 2),

            "Upper Bound":
                round(upper, 2)
        })


    return pd.DataFrame(results)


# DATA QUALITY SCORE
def data_quality_score(df):

    total_cells = df.size

    if total_cells == 0:

        return 0


    missing = (
        df.isna()
        .sum()
        .sum()
    )


    duplicates = (
        df.duplicated()
        .sum()
    )


    invalid_report = (
        invalid_value_report(df)
    )


    if invalid_report.empty:

        invalid_count = 0

    else:

        invalid_count = int(
            invalid_report["Count"].sum()
        )


    missing_penalty = (
        missing /
        total_cells
    ) * 40


    duplicate_penalty = (

        duplicates /
        len(df)

    ) * 20 if len(df) > 0 else 0


    invalid_penalty = (

        invalid_count /
        total_cells

    ) * 20


    score = (
        100
        - missing_penalty
        - duplicate_penalty
        - invalid_penalty
    )


    return max(
        0,
        round(score, 2)
    )


# QUALITY SCORE BREAKDOWN
def quality_score_breakdown(df):

    total_cells = df.size

    if total_cells == 0:

        return {

            "Missing Penalty": 0,

            "Duplicates Penalty": 0,

            "Invalid Values Penalty": 0
        }


    invalid_report = (
        invalid_value_report(df)
    )


    if invalid_report.empty:

        invalid_count = 0

    else:

        invalid_count = int(
            invalid_report["Count"].sum()
        )


    missing_penalty = (

        df.isna()
        .sum()
        .sum()
        /
        total_cells

    ) * 40


    duplicate_penalty = (

        df.duplicated().sum()
        /
        len(df)

    ) * 20 if len(df) > 0 else 0


    invalid_penalty = (

        invalid_count /
        total_cells

    ) * 20


    return {

        "Missing Penalty":
            round(
                missing_penalty,
                2
            ),

        "Duplicates Penalty":
            round(
                duplicate_penalty,
                2
            ),

        "Invalid Values Penalty":
            round(
                invalid_penalty,
                2
            )
    }


# OUTLIER ACTION
def apply_outlier_action(
    df,
    col,
    action
):

    df = df.copy()


    series = pd.to_numeric(

        df[col]
        .astype(str)
        .str.replace(
            r"[^\d\.\-]",
            "",
            regex=True
        ),

        errors="coerce"
    )


    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr


    mask = (

        (series < lower) |
        (series > upper)

    )


    # Remove rows

    if action == "Remove rows":

        return (
            df.loc[~mask]
            .reset_index(drop=True)
        )


    # Cap values

    elif action == "Cap values":

        df.loc[
            series < lower,
            col
        ] = lower

        df.loc[
            series > upper,
            col
        ] = upper


    # Replace with median

    elif action == "Replace with median":

        df.loc[
            mask,
            col
        ] = series.median()


    return df


# CATEGORY MAPPING
def smart_category_mapping(
    df,
    col
):

    df = df.copy()

    df[col] = (
        df[col]
        .replace(CATEGORY_MAP)
    )

    return df


# SESSION STATE
if "original_df" not in st.session_state:

    st.session_state.original_df = None


if "cleaned_df" not in st.session_state:

    st.session_state.cleaned_df = None


if "cleaning_log" not in st.session_state:

    st.session_state.cleaning_log = []


if "file_id" not in st.session_state:

    st.session_state.file_id = None


# HEADER
st.title(
    "🧹 Data Cleaning & Analysis Tool"
)

st.caption(
    "Upload your dataset, analyze its quality, "
    "clean it and download the cleaned version."
)


# FILE UPLOAD
uploaded = st.file_uploader(

    "📁 Upload CSV or Excel",

    type=[
        "csv",
        "xlsx"
    ]
)


if uploaded is not None:

    current_file_id = (
        f"{uploaded.name}_"
        f"{uploaded.size}"
    )


    # Only load when a NEW file is uploaded

    if (
        st.session_state.file_id
        != current_file_id
    ):

        df = load_file(uploaded)


        if df is not None:

            st.session_state.original_df = (
                df.copy()
            )

            st.session_state.cleaned_df = (
                df.copy()
            )

            st.session_state.cleaning_log = []

            st.session_state.file_id = (
                current_file_id
            )

            st.success(
                f"✅ Successfully loaded "
                f"{uploaded.name}"
            )

        else:

            st.error(
                "❌ Failed to load the dataset."
            )

            st.stop()


# STOP IF NO DATASET
if (
    st.session_state.original_df
    is None
):

    st.info(
        "👆 Upload a CSV or Excel file "
        "to start."
    )

    st.stop()


# DATA REFERENCES
original_df = (
    st.session_state.original_df
)

cleaned_df = (
    st.session_state.cleaned_df
)


# TABS
overview, quality, viz, cleaning = st.tabs([

    "📊 Overview",

    "🔍 Quality Report",

    "📈 Visualizations",

    "🧹 Cleaning"

])


# OVERVIEW TAB
with overview:

    st.header(
        "📊 Dataset Overview"
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Rows",
        len(original_df)
    )


    c2.metric(
        "Columns",
        len(original_df.columns)
    )


    c3.metric(
        "Missing Values",
        int(
            original_df
            .isna()
            .sum()
            .sum()
        )
    )


    c4.metric(
        "Duplicate Rows",
        int(
            original_df
            .duplicated()
            .sum()
        )
    )


    c5.metric(
        "Data Quality Score",
        f"{data_quality_score(original_df)}%"
    )


    st.divider()


    st.subheader(
        "⚖️ Before vs After"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.write(
            "### Original Dataset"
        )

        st.dataframe(
            original_df.head(20),
            use_container_width=True
        )


    with c2:

        st.write(
            "### Cleaned Dataset"
        )

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True
        )


# QUALITY REPORT TAB
with quality:

    st.header(
        "🔍 Data Quality Report"
    )


    # ---------------- Quality Table ----------------

    st.subheader(
        "📋 Column Quality"
    )


    st.dataframe(

        quality_report(
            original_df
        ),

        use_container_width=True
    )


    # ---------------- Data Types ----------------

    st.subheader(
        "🔠 Data Type Detection"
    )


    dtype_df = pd.DataFrame({

        "Column":
            original_df.columns,

        "Detected Type": [

            detect_best_dtype(
                original_df[col]
            )

            for col
            in original_df.columns

        ]
    })


    st.dataframe(
        dtype_df,
        use_container_width=True
    )


    # ---------------- Duplicates ----------------

    st.subheader(
        "🔁 Duplicate Report"
    )


    duplicates = original_df[
        original_df.duplicated(
            keep=False
        )
    ]


    if len(duplicates) > 0:

        st.warning(
            f"⚠️ {len(duplicates)} "
            f"rows belong to duplicate groups."
        )

        st.dataframe(
            duplicates,
            use_container_width=True
        )

    else:

        st.success(
            "✅ No duplicate rows found."
        )


    # ---------------- Outliers ----------------

    st.subheader(
        "📌 Outlier Report"
    )


    outliers = outlier_report(
        original_df
    )


    if not outliers.empty:

        st.dataframe(
            outliers,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns available "
            "for outlier detection."
        )


    # ---------------- Invalid Values ----------------

    st.subheader(
        "⚠️ Invalid Value Report"
    )


    invalids = (
        invalid_value_report(
            original_df
        )
    )


    if not invalids.empty:

        st.dataframe(
            invalids,
            use_container_width=True
        )

    else:

        st.success(
            "✅ No invalid values detected."
        )


    # ---------------- Score Breakdown ----------------

    st.subheader(
        "💯 Data Quality Score Breakdown"
    )


    breakdown = (
        quality_score_breakdown(
            original_df
        )
    )


    breakdown_df = pd.DataFrame({

        "Quality Issue":
            breakdown.keys(),

        "Penalty":
            breakdown.values()

    })


    st.dataframe(
        breakdown_df,
        use_container_width=True
    )


# VISUALIZATION TAB
with viz:

    st.header(
        "📈 Data Visualizations"
    )


    # MISSING VALUES
    st.subheader(
        "1️⃣ Missing Values"
    )


    missing = (

        original_df
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )

    )


    if missing.sum() > 0:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.bar(

            missing.index.astype(str),

            missing.values

        )


        ax.set_xlabel(
            "Columns"
        )

        ax.set_ylabel(
            "Missing Values"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.success(
            "✅ No missing values found."
        )


    # NUMERIC COLUMNS
    st.subheader(
        "2️⃣ Numeric Distribution"
    )


    num_cols = (
        detect_numeric_columns(
            cleaned_df
        )
    )


    if num_cols:

        selected_numeric = st.selectbox(

            "Select numeric column",

            num_cols,

            key="histogram_column"

        )


        series = pd.to_numeric(

            cleaned_df[
                selected_numeric
            ]
            .astype(str)
            .str.replace(
                r"[^\d\.\-]",
                "",
                regex=True
            ),

            errors="coerce"

        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        ax.hist(
            series.dropna(),
            bins=30
        )


        ax.set_title(
            f"Distribution of {selected_numeric}"
        )

        ax.set_xlabel(
            selected_numeric
        )

        ax.set_ylabel(
            "Frequency"
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No numeric columns detected."
        )


    # BOXPLOT
    st.subheader(
        "3️⃣ Boxplot / Outlier Visualization"
    )


    if num_cols:

        selected_box = st.selectbox(

            "Select numeric column",

            num_cols,

            key="boxplot_column"

        )


        box_data = pd.to_numeric(

            cleaned_df[
                selected_box
            ]
            .astype(str)
            .str.replace(
                r"[^\d\.\-]",
                "",
                regex=True
            ),

            errors="coerce"

        )


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        sns.boxplot(
            y=box_data,
            ax=ax
        )


        ax.set_ylabel(
            selected_box
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # CATEGORY DISTRIBUTION
    st.subheader(
        "4️⃣ Category Distribution"
    )


    cat_cols = (

        cleaned_df
        .select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()

    )


    if cat_cols:

        selected_category = st.selectbox(

            "Select category column",

            cat_cols,

            key="category_column"

        )


        counts = (

            cleaned_df[
                selected_category
            ]
            .value_counts()
            .head(15)

        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.bar(

            counts.index.astype(str),

            counts.values

        )


        ax.set_xlabel(
            selected_category
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    else:

        st.info(
            "No categorical columns detected."
        )


    # PIE CHART
    st.subheader(
        "5️⃣ Pie Chart"
    )


    if cat_cols:

        selected_pie = st.selectbox(

            "Select category column",

            cat_cols,

            key="pie_column"

        )


        pie_counts = (

            cleaned_df[
                selected_pie
            ]
            .value_counts()
            .head(10)

        )


        fig, ax = plt.subplots(
            figsize=(7, 7)
        )


        ax.pie(

            pie_counts.values,

            labels=pie_counts.index.astype(str),

            autopct="%1.1f%%"

        )


        ax.set_title(
            f"{selected_pie} Distribution"
        )


        st.pyplot(fig)

        plt.close(fig)


    # CORRELATION
    st.subheader(
        "6️⃣ Correlation Matrix"
    )


    if len(num_cols) >= 2:

        numeric_df = (

            cleaned_df[
                num_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )

        )


        corr = numeric_df.corr()


        fig, ax = plt.subplots(
            figsize=(9, 7)
        )


        sns.heatmap(

            corr,

            annot=True,

            cmap="coolwarm",

            fmt=".2f",

            ax=ax

        )


        ax.set_title(
            "Correlation Matrix"
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "At least two numeric columns "
            "are required for correlation analysis."
        )


    # SUMMARY STATISTICS
    st.subheader(
        "7️⃣ Summary Statistics"
    )


    if num_cols:

        numeric_summary = (

            cleaned_df[
                num_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )

        )


        summary = (

            numeric_summary
            .describe()
            .T[
                [
                    "mean",
                    "50%",
                    "std",
                    "min",
                    "max"
                ]
            ]
            .rename(
                columns={
                    "50%": "median"
                }
            )

        )


        st.dataframe(
            summary,
            use_container_width=True
        )


    # CATEGORY FREQUENCY
    st.subheader(
        "8️⃣ Category Frequency Analysis"
    )


    if cat_cols:

        for col in cat_cols:

            st.write(
                f"### {col}"
            )


            counts = (
                cleaned_df[col]
                .value_counts()
            )


            st.write(
                "Top categories:"
            )


            st.dataframe(
                counts.head(5)
                .rename("Count")
                .to_frame()
            )


            rare_threshold = (
                0.01 *
                len(cleaned_df)
            )


            rare = counts[
                counts < rare_threshold
            ]


            if len(rare) > 0:

                st.warning(

                    f"Rare categories detected "
                    f"in `{col}`: "
                    f"{rare.index.tolist()}"

                )


# CLEANING TAB
with cleaning:

    st.header(
        "🧹 Cleaning Actions"
    )


    # ========================================================
    # INVALID VALUES
    # ========================================================

    st.subheader(
        "1️⃣ Replace Invalid Values"
    )


    st.write(
        "Detected invalid values:"
    )


    current_invalids = (
        invalid_value_report(
            cleaned_df
        )
    )


    if not current_invalids.empty:

        st.dataframe(
            current_invalids,
            use_container_width=True
        )

    else:

        st.success(
            "No invalid values detected."
        )


    if st.button(
        "🧹 Replace Invalid Values",
        key="replace_invalid"
    ):

        before = (
            invalid_value_report(
                st.session_state.cleaned_df
            )
        )


        st.session_state.cleaned_df = (
            replace_invalid_values(
                st.session_state.cleaned_df
            )
        )


        if not before.empty:

            total_replaced = int(
                before["Count"].sum()
            )

        else:

            total_replaced = 0


        st.session_state.cleaning_log.append({

            "Action":
                "Replace Invalid Values",

            "Details":
                f"{total_replaced} values replaced with NaN",

            "Status":
                "Completed"

        })


        st.success(
            f"✅ {total_replaced} "
            f"invalid values replaced with NaN."
        )


        st.rerun()


    # ========================================================
    # DUPLICATES
    # ========================================================

    st.subheader(
        "2️⃣ Remove Duplicate Rows"
    )


    duplicate_count = int(
        st.session_state
        .cleaned_df
        .duplicated()
        .sum()
    )


    st.write(
        f"Duplicate rows detected: "
        f"**{duplicate_count}**"
    )


    if st.button(
        "🗑️ Remove Duplicate Rows",
        key="remove_duplicates"
    ):

        before_rows = len(
            st.session_state.cleaned_df
        )


        st.session_state.cleaned_df = (

            st.session_state
            .cleaned_df
            .drop_duplicates()
            .reset_index(drop=True)

        )


        after_rows = len(
            st.session_state.cleaned_df
        )


        removed = (
            before_rows -
            after_rows
        )


        st.session_state.cleaning_log.append({

            "Action":
                "Remove Duplicate Rows",

            "Details":
                f"{removed} duplicate rows removed",

            "Status":
                "Completed"

        })


        st.success(
            f"✅ {removed} duplicate rows removed."
        )


        st.rerun()


    # ========================================================
    # OUTLIER CLEANING
    # ========================================================

    st.subheader(
        "3️⃣ Outlier Treatment"
    )


    clean_num_cols = (
        detect_numeric_columns(
            st.session_state.cleaned_df
        )
    )


    if clean_num_cols:

        selected_outlier_col = st.selectbox(

            "Select numeric column",

            clean_num_cols,

            key="outlier_column"

        )


        outlier_action = st.selectbox(

            "Choose action",

            [
                "Remove rows",
                "Cap values",
                "Replace with median"
            ],

            key="outlier_action"

        )


        if st.button(
            "⚙️ Apply Outlier Action",
            key="apply_outlier"
        ):

            before_rows = len(
                st.session_state.cleaned_df
            )


            st.session_state.cleaned_df = (

                apply_outlier_action(

                    st.session_state.cleaned_df,

                    selected_outlier_col,

                    outlier_action

                )

            )


            after_rows = len(
                st.session_state.cleaned_df
            )


            if outlier_action == "Remove rows":

                details = (
                    f"{before_rows - after_rows} "
                    f"rows removed from "
                    f"{selected_outlier_col}"
                )

            else:

                details = (
                    f"{outlier_action} applied "
                    f"to {selected_outlier_col}"
                )


            st.session_state.cleaning_log.append({

                "Action":
                    "Outlier Treatment",

                "Details":
                    details,

                "Status":
                    "Completed"

            })


            st.success(
                f"✅ {details}"
            )


            st.rerun()


    else:

        st.info(
            "No numeric columns available "
            "for outlier treatment."
        )


    # ========================================================
    # CATEGORY MAPPING
    # ========================================================

    st.subheader(
        "4️⃣ Category Standardization"
    )


    mapping_cols = [

        col

        for col in
        st.session_state
        .cleaned_df.columns

        if (
            st.session_state
            .cleaned_df[col]
            .dtype == "object"
        )

    ]


    if mapping_cols:

        selected_mapping_col = st.selectbox(

            "Select category column",

            mapping_cols,

            key="mapping_column"

        )


        st.write(
            "Available mappings:"
        )


        st.code(
            str(CATEGORY_MAP)
        )


        if st.button(
            "🔄 Apply Category Mapping",
            key="apply_mapping"
        ):

            st.session_state.cleaned_df = (

                smart_category_mapping(

                    st.session_state.cleaned_df,

                    selected_mapping_col

                )

            )


            st.session_state.cleaning_log.append({

                "Action":
                    "Category Standardization",

                "Details":
                    f"Applied category mapping "
                    f"to {selected_mapping_col}",

                "Status":
                    "Completed"

            })


            st.success(
                "✅ Category mapping applied."
            )


            st.rerun()


    # ========================================================
    # CLEANING LOG
    # ========================================================

    st.divider()


    st.subheader(
        "📋 Cleaning Summary Log"
    )


    if st.session_state.cleaning_log:

        log_df = pd.DataFrame(
            st.session_state.cleaning_log
        )


        st.dataframe(
            log_df,
            use_container_width=True
        )

    else:

        st.info(
            "No cleaning actions performed yet."
        )


    # ========================================================
    # RESET
    # ========================================================

    st.divider()


    if st.button(
        "↩️ Reset All Cleaning",
        key="reset_cleaning"
    ):

        st.session_state.cleaned_df = (

            st.session_state
            .original_df
            .copy()

        )


        st.session_state.cleaning_log = []


        st.success(
            "✅ Dataset reset to original."
        )


        st.rerun()


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()


    st.subheader(
        "⬇️ Download"
    )


    cleaned_csv = (
        st.session_state
        .cleaned_df
        .to_csv(
            index=False
        )
    )


    st.download_button(

        label="⬇️ Download Cleaned Dataset",

        data=cleaned_csv,

        file_name="cleaned_dataset.csv",

        mime="text/csv",

        key="download_cleaned"

    )