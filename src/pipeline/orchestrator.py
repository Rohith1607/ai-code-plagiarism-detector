from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity


class AnalysisPipeline:
    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.ast_analyzer = ASTAnalyzer()
        self.token_similarity = TokenSimilarity()

    def run(self, code: str, language: str | None = None) -> dict:

        # 1️⃣ Normalization
        normalized_code, norm_meta = self.normalizer.normalize(code)

        # 2️⃣ AST Analysis (Python only)
        ast_features = {}
        if language == "python" or language is None:
            ast_features = self.ast_analyzer.analyze(normalized_code)

        # 3️⃣ Token Similarity (TEMP self-compare)
        token_sim = self.token_similarity.jaccard_similarity(
            normalized_code,
            normalized_code
        )


        # Dummy response for now
        return {
            "plagiarism_percentage": 0.0,
            "ai_probability": 0.0,
            "confidence": "low"
        }
