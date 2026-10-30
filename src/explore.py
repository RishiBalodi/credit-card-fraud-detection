import pandas as pd
import numpy as np

csv_path = r"C:\Users\hp\.gemini\antigravity\scratch\credit_card_fraud_detection\data\credit_card_transactions.csv"
df = pd.read_csv(csv_path)

print("=== AMT Stats Overall ===")
print(df['amt'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n=== AMT Stats by is_fraud ===")
print(df.groupby('is_fraud')['amt'].describe())

# Check IQR outliers
q1 = df['amt'].quantile(0.25)
q3 = df['amt'].quantile(0.75)
iqr = q3 - q1
upper_15 = q3 + 1.5 * iqr
upper_30 = q3 + 3.0 * iqr
p99 = df['amt'].quantile(0.99)
p995 = df['amt'].quantile(0.995)

print(f"\nQ1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"Upper 1.5*IQR={upper_15:.2f}, count above={sum(df['amt'] > upper_15)}, frauds above={sum((df['amt'] > upper_15) & (df['is_fraud']==1))}")
print(f"Upper 3.0*IQR={upper_30:.2f}, count above={sum(df['amt'] > upper_30)}, frauds above={sum((df['amt'] > upper_30) & (df['is_fraud']==1))}")
print(f"99th percentile={p99:.2f}, count above={sum(df['amt'] > p99)}, frauds above={sum((df['amt'] > p99) & (df['is_fraud']==1))}")
print(f"99.5th percentile={p995:.2f}, count above={sum(df['amt'] > p995)}, frauds above={sum((df['amt'] > p995) & (df['is_fraud']==1))}")

print("\n=== Category fraud distribution ===")
cat_summary = df.groupby('category')['is_fraud'].agg(Total='count', Fraud_Count='sum', Fraud_Rate='mean').sort_values(by='Fraud_Count', ascending=False)
print(cat_summary)
