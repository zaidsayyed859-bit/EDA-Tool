import pandas as pd

# Load both files
dirty = pd.read_csv("dirty_cafe_sales.csv")
cleaned = pd.read_csv("cleaned_dataset.csv")

# Quick overview
print("Dirty dataset shape:", dirty.shape)
print("Cleaned dataset shape:", cleaned.shape)

# Check for missing values
print("Missing values before:\n", dirty.isna().sum())
print("Missing values after:\n", cleaned.isna().sum())

# Compare column names
print("Columns before:", dirty.columns.tolist())
print("Columns after:", cleaned.columns.tolist())

# Spot differences in unique values
for col in cleaned.columns:
    if col in dirty.columns:
        print(f"{col}: before={dirty[col].nunique()}, after={cleaned[col].nunique()}")
