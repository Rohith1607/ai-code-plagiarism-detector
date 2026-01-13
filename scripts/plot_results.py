import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/results/evaluation_results.csv"

df = pd.read_csv(CSV_PATH)

human = df[df["label"] == "HUMAN"]
ai = df[df["label"] == "AI"]

# -------------------------
# 1. BOX PLOT – PLAGIARISM %
# -------------------------
plt.figure()
plt.boxplot(
    [human["plagiarism_percentage"], ai["plagiarism_percentage"]],
    labels=["HUMAN", "AI"]
)
plt.title("Plagiarism Percentage Distribution")
plt.ylabel("Plagiarism %")
plt.show()

# -------------------------
# 2. BOX PLOT – AI PROBABILITY
# -------------------------
plt.figure()
plt.boxplot(
    [human["ai_probability"], ai["ai_probability"]],
    labels=["HUMAN", "AI"]
)
plt.title("AI Probability Distribution")
plt.ylabel("AI Probability")
plt.show()

# -------------------------
# 3. HISTOGRAM – PLAGIARISM %
# -------------------------
plt.figure()
plt.hist(human["plagiarism_percentage"], bins=20, alpha=0.7, label="HUMAN")
plt.hist(ai["plagiarism_percentage"], bins=20, alpha=0.7, label="AI")
plt.title("Plagiarism Percentage Histogram")
plt.xlabel("Plagiarism %")
plt.ylabel("Frequency")
plt.legend()
plt.show()
