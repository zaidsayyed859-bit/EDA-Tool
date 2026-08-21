from fpdf import FPDF
from google import genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Enterprise AI Data & Executive Suite",
    page_icon="💼",
    layout="wide",
)

# --- INITIALIZE GEMINI CLIENT ---
API_KEY = "AIzaSyBomT8FI8QhJ4Jv6KwBaJ2jY8TN05bvRkE"
try:
  client = genai.Client(api_key=API_KEY)
except Exception:
  client = None


# --- FULL ENTERPRISE ENGINE ---
class EnterpriseEngine:

  def __init__(self, file_path):
    self.file_path = file_path
    self.df = None
    self.clean_df = None
    self.audit_report = {}
    self.anomalies = pd.DataFrame()
    self.audit_log = []

  def load_file(self):
    if self.file_path.endswith(".csv"):
      self.df = pd.read_csv(self.file_path)
    elif self.file_path.endswith((".xls", ".xlsx")):
      self.df = pd.read_excel(self.file_path)
    else:
      raise ValueError("Unsupported format.")
    return self.df

  def run_enterprise_pipeline(self):
    if self.df is None:
      return
    self.clean_df = self.df.copy()
    initial_rows = len(self.clean_df)
    total_cells = self.clean_df.size
    missing_initial = self.clean_df.isnull().sum().sum()

    # 1. Duplicate Removal & Logging
    dups = self.clean_df.duplicated().sum()
    if dups > 0:
      self.clean_df.drop_duplicates(inplace=True)
      self.audit_log.append(f"Removed {dups} exact duplicate rows.")

    # 2. String Cleaning & Typo Standardization
    string_cols = self.clean_df.select_dtypes(include=["object"]).columns
    for col in string_cols:
      self.clean_df[col] = self.clean_df[col].astype(str).str.strip()
      mask = self.clean_df[col].isin(
          ["ERROR", "Error", "error", "nan", "NaN", ""]
      )
      if mask.sum() > 0:
        self.audit_log.append(
            f"Normalized {mask.sum()} placeholder/error strings in column '{col}'."
        )
        self.clean_df.loc[mask, col] = np.nan

    # 3. Missing Value Imputation & Logging
    missing_filled = 0
    for col in self.clean_df.columns:
      nulls = self.clean_df[col].isnull().sum()
      if nulls > 0:
        missing_filled += nulls
        if self.clean_df[col].dtype in ["float64", "int64"]:
          med = self.clean_df[col].median()
          self.clean_df[col].fillna(med, inplace=True)
          self.audit_log.append(
              f"Filled {nulls} missing numeric cells in '{col}' with median"
              f" ({med})."
          )
        else:
          self.clean_df[col].fillna("Unknown", inplace=True)
          self.audit_log.append(
              f"Filled {nulls} missing text cells in '{col}' with 'Unknown'."
          )

    # 4. Anomaly & Outlier Detection (Z-Score method)
    numeric_df = self.clean_df.select_dtypes(include=["float64", "int64"])
    if not numeric_df.empty:
      z_scores = np.abs(stats.zscore(numeric_df))
      anomaly_mask = (z_scores > 3).any(axis=1)
      self.anomalies = self.clean_df[anomaly_mask]
      if not self.anomalies.empty:
        self.audit_log.append(
            f"Flagged {len(self.anomalies)} statistical anomaly rows (Z-score >"
            " 3)."
        )

    # 5. Health Score Calculation
    penalty = (
        (missing_initial / max(total_cells, 1)) * 40
        + (dups / max(initial_rows, 1)) * 40
        + (len(self.anomalies) / max(initial_rows, 1)) * 20
    )
    health_score = max(round(100 - penalty, 2), 10.0)

    # 6. Financial KPIs if available
    kpis = {}
    if "Total Spent" in self.clean_df.columns:
      self.clean_df["Total Spent"] = pd.to_numeric(
          self.clean_df["Total Spent"], errors="coerce"
      )
      kpis["total_revenue"] = round(self.clean_df["Total Spent"].sum(), 2)
      kpis["avg_transaction"] = round(self.clean_df["Total Spent"].mean(), 2)
    else:
      kpis["total_revenue"] = 0
      kpis["avg_transaction"] = 0

    self.audit_report = {
        "initial_rows": initial_rows,
        "final_rows": len(self.clean_df),
        "duplicates_removed": int(dups),
        "missing_fixed": int(missing_filled),
        "anomalies_found": len(self.anomalies),
        "health_score": health_score,
        "kpis": kpis,
    }


# --- PDF REPORT GENERATOR ---
def generate_pdf_report(audit):
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font("Arial", "B", 16)
  pdf.cell(0, 10, "Executive Data Quality & Audit Report", ln=True, align="center")
  pdf.ln(10)

  pdf.set_font("Arial", "", 12)
  pdf.cell(
      0, 10, f"Data Health Score: {audit['health_score']}%", ln=True
  )
  pdf.cell(
      0, 10, f"Initial Records Processed: {audit['initial_rows']}", ln=True
  )
  pdf.cell(0, 10, f"Final Clean Records: {audit['final_rows']}", ln=True)
  pdf.cell(0, 10, f"Duplicates Removed: {audit['duplicates_removed']}", ln=True)
  pdf.cell(
      0, 10, f"Missing Values Sanitized: {audit['missing_fixed']}", ln=True
  )
  pdf.cell(
      0, 10, f"Statistical Outliers Flagged: {audit['anomalies_found']}", ln=True
  )

  output = pdf.output(dest="S")
  if isinstance(output, str):
    return output.encode("latin1")
  elif isinstance(output, bytearray):
    return bytes(output)
  return output


# --- SESSION STATE INITIALIZATION ---
if "pipeline_run" not in st.session_state:
  st.session_state.pipeline_run = False
if "engine" not in st.session_state:
  st.session_state.engine = None


# --- UI DASHBOARD ---
st.title("💼 Enterprise AI Data Cleaning & Executive Suite")
st.write(
    "Upload raw datasets to execute sanitization pipelines, review anomalies,"
    " chat with Gemini AI, and export professional reports."
)

uploaded_file = st.file_uploader(
    "Upload Company Dataset (CSV or Excel)", type=["csv", "xlsx"]
)

if uploaded_file is not None:
  temp_path = "temp_dataset.csv"
  with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  # Initialize engine if new file uploaded
  if (
      st.session_state.engine is None
      or st.session_state.engine.file_path != temp_path
  ):
    st.session_state.engine = EnterpriseEngine(temp_path)
    st.session_state.engine.load_file()
    st.session_state.pipeline_run = False

  if st.button("🚀 Run Full Enterprise Processing Pipeline", type="primary"):
    with st.spinner("Executing sanitization & anomaly scan..."):
      st.session_state.engine.run_enterprise_pipeline()
      st.session_state.pipeline_run = True
      st.success("Pipeline executed seamlessly!")

  # Persist dashboard if pipeline has been run successfully
  if st.session_state.pipeline_run:
    engine = st.session_state.engine
    audit = engine.audit_report
    kpis = audit["kpis"]

    # --- KPI DASHBOARD ---
    st.markdown("### 🏢 Executive Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Health Grade", f"{audit['health_score']}%")
    c2.metric("Total Revenue / Value", f"${kpis['total_revenue']:,.2f}")
    c3.metric("Avg Transaction", f"${kpis['avg_transaction']:,.2f}")
    c4.metric("Anomalies Flagged", audit["anomalies_found"])

    # --- TABS FOR MODULES ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Cleaned Data",
        "🚨 Outliers & Anomalies",
        "📋 Audit Log",
        "✨ Gemini AI Assistant",
        "📥 Executive Exports",
    ])

    with tab1:
      st.subheader("Sanitized Master Table")
      st.dataframe(engine.clean_df, height=350, use_container_width=True)

    with tab2:
      st.subheader("Suspicious Records & Statistical Outliers")
      if not engine.anomalies.empty:
        st.warning(
            f"Found {len(engine.anomalies)} records breaking standard"
            " statistical patterns."
        )
        st.dataframe(engine.anomalies, height=300, use_container_width=True)
      else:
        st.info(
            "No major statistical anomalies detected based on Z-score filters."
        )

    with tab3:
      st.subheader("Step-by-Step Data Quality Log")
      for log_entry in engine.audit_log:
        st.text(f"✔ {log_entry}")

    with tab4:
      st.subheader("✨ Gemini Enterprise Data Intelligence")
      if not client:
        st.error("Gemini client failed to initialize.")
      else:
        st.write(
            "Ask questions, generate business summaries, or uncover hidden"
            " patterns directly with Gemini AI based on your dataset."
        )

        df_summary = engine.clean_df.describe().to_string()
        df_columns = list(engine.clean_df.columns)

        # Using a form or button prevents accidental resets when typing
        with st.form(key="gemini_form"):
          user_prompt = st.text_area(
              "What would you like to ask Gemini about your data?",
              placeholder=(
                  "e.g., Give me a 3-bullet executive summary or explain any"
                  " trends."
              ),
          )
          submit_button = st.form_submit_button(label="Ask Gemini")

        if submit_button:
          if user_prompt:
            with st.spinner("Gemini is analyzing your dataset..."):
              system_instruction = (
                  "You are an expert Chief Data Officer and business"
                  f" analyst. Dataset columns: {df_columns}. Dataset"
                  f" statistical summary:\n{df_summary}"
              )
              response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                )
              
              st.markdown("### 🤖 Gemini Insights:")
              st.write(response.text)
          else:
            st.warning("Please enter a question or prompt.")

    with tab5:
      st.subheader("Download Executive Deliverables")
      col_dl1, col_dl2 = st.columns(2)

      with col_dl1:
        csv_data = engine.clean_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned CSV",
            csv_data,
            "cleaned_master_dataset.csv",
            "text/csv",
        )

      with col_dl2:
        pdf_bytes = generate_pdf_report(audit)
        st.download_button(
            "📄 Download Executive PDF Report",
            pdf_bytes,
            "Executive_Audit_Report.pdf",
            "application/pdf",
        )