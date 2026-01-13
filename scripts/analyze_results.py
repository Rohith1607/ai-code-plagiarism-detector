import pandas as pd

CSV_PATH = "data/results/evaluation_results.csv"

df = pd.read_csv(r"C:\Users\DELL\Rohith\ai-code-plagiarism-detector\data\results\evaluation_results.csv")

print("\n=== SAMPLE COUNTS ===")
print(df["label"].value_counts())

print("\n=== PLAGIARISM % ===")
print(df.groupby("label")["plagiarism_percentage"].describe())

print("\n=== AI PROBABILITY ===")
print(df.groupby("label")["ai_probability"].describe())
