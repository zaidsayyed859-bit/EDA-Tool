import numpy as np
import pandas as pd


class DataCleaningEngine:

  def __init__(self, file_path):
    self.file_path = file_path
    self.df = None
    self.audit_report = {}

  def load_data(self):
    """Loads CSV or Excel files safely into a Pandas DataFrame."""
    if self.file_path.endswith('.csv'):
      self.df = pd.read_csv(self.file_path)
    elif self.file_path.endswith(('.xls', '.xlsx')):
      self.df = pd.read_excel(self.file_path)
    else:
      raise ValueError('Unsupported file format. Please use CSV or Excel.')
    return self.df

  def clean_data(self):
    """Performs deep cleaning, tracks error metrics, and computes health scores."""
    if self.df is None:
      return None

    total_cells_initial = self.df.size
    missing_initial = self.df.isnull().sum().sum()

    # 1. Track and remove duplicates
    initial_rows = len(self.df)
    self.df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(self.df)

    # 2. String cleaning and typo/case standardization
    string_cols = self.df.select_dtypes(include=['object']).columns
    for col in string_cols:
      self.df[col] = self.df[col].astype(str).str.strip()
      # Standardize placeholder variations like 'UNKNOWN', 'Error', 'NaN'
      self.df[col] = self.df[col].replace(
          ['ERROR', 'Error', 'error', 'nan', 'NaN', ''], np.nan
      )

    # 3. Handle missing values contextually
    missing_filled_count = 0
    for col in self.df.columns:
      missing_filled_count += self.df[col].isnull().sum()
      if self.df[col].dtype in ['float64', 'int64']:
        median_val = self.df[col].median()
        self.df[col].fillna(median_val, inplace=True)
      else:
        self.df[col].fillna('Unknown', inplace=True)

    # 4. Calculate Data Health Score (0 to 100%)
    penalty = (
        (missing_initial / max(total_cells_initial, 1)) * 50
        + (duplicates_removed / max(initial_rows, 1)) * 50
    )
    health_score = max(round(100 - penalty, 2), 10.0)

    # Compile Audit Report Summary
    self.audit_report = {
        'initial_rows': initial_rows,
        'final_rows': len(self.df),
        'duplicates_removed': duplicates_removed,
        'missing_values_fixed': int(missing_filled_count),
        'health_score': health_score,
    }

    return {
        'cleaned_dataframe': self.df,
        'audit': self.audit_report,
    }

  def export_data(self, output_path='cleaned_output.csv'):
    """Exports the cleaned dataset to disk."""
    if self.df is not None:
      self.df.to_csv(output_path, index=False)
      return output_path
    return None