import os
import csv
import sys

from transformers import pipeline
sys.path.append(os.path.abspath("."))
from src.pipeline.orchestrator import AnalysisPipeline


HUMAN_DIR = "data/raw/human"
AI_DIR = "data/raw/ai"
OUTPUT_FILE = "data/results/evaluation_results.csv"


def read_code(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def evaluate_folder(pipeline, root_dir, label, writer):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)
            code = read_code(path)

            if not code or len(code.strip()) == 0:
                continue

            result = pipeline.run(code, language="python")



            writer.writerow({
                "file": path,
                "label": label,
                "plagiarism_percentage": result["plagiarism_percentage"],
                "ai_probability": result["ai_probability"]
            })


def main():
    os.makedirs("data/results", exist_ok=True)

    pipeline = AnalysisPipeline()

    # 🔥 Disable DB writes ONLY for evaluation
    pipeline.repo.save_result = lambda *args, **kwargs: None

    #pipeline.repo = None   # disable DB writes

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "label",
                "plagiarism_percentage",
                "ai_probability"
            ]
        )
        writer.writeheader()

        print("▶ Evaluating HUMAN code...")
        evaluate_folder(pipeline, HUMAN_DIR, "HUMAN", writer)

        print("▶ Evaluating AI code...")
        evaluate_folder(pipeline, AI_DIR, "AI", writer)

    print(f"✅ Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
