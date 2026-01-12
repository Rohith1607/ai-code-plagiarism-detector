import hashlib
import json

from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.scorer import ScoreAggregator
from src.storage.faiss_index import FaissIndex
from src.storage.repository import AnalysisRepository


class AnalysisPipeline:

    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.ast_analyzer = ASTAnalyzer()
        self.token_similarity = TokenSimilarity()
        self.embedding_generator = EmbeddingGenerator()
        self.scorer = ScoreAggregator()
        self.repo = AnalysisRepository()
        self.faiss_index = FaissIndex(vector_dim=768)

    def _ast_similarity(self, a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        keys = set(a.keys()) | set(b.keys())
        diff = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
        return 1 / (1 + diff)

    def run(self, code: str, language: str | None = None) -> dict:

        normalized_code, _ = self.normalizer.normalize(code)

        ast_features = {}
        if language in (None, "python"):
            ast_features = self.ast_analyzer.analyze(normalized_code)

        embedding = self.embedding_generator.generate(normalized_code)
        vector = embedding.detach().numpy()

        # First submission baseline
        if self.faiss_index.size() == 0:
            self.faiss_index.add(vector)
            self.repo.save_result(
                hashlib.sha256(code.encode()).hexdigest(),
                0.0,
                0.0,
                normalized_code,
                ast_features
            )
            return {
                "plagiarism_percentage": 0.0,
                "ai_probability": 0.0,
                "confidence": "low",
                "explanation": {
                    "reasoning": "First submission baseline"
                }
            }

        distances, _ = self.faiss_index.search(vector, k=1)
        semantic_sim = 1 / (1 + distances[0]) if distances else 0.0

        token_sim = 0.0
        structure_sim = 0.0

        for norm_db, ast_json in self.repo.fetch_all_for_similarity():
            if norm_db:
                token_sim = max(
                    token_sim,
                    self.token_similarity.jaccard_similarity(
                        normalized_code, norm_db
                    )
                )
            if ast_json:
                structure_sim = max(
                    structure_sim,
                    self._ast_similarity(ast_features, json.loads(ast_json))
                )

        plagiarism_score = self.scorer.compute_plagiarism_score(
            token_sim, semantic_sim, structure_sim
        )

        ai_probability = self.scorer.compute_ai_probability(
            semantic_sim, structure_sim
        )

        line_count = len([l for l in normalized_code.splitlines() if l.strip()])
        size_penalty = False

        if line_count <= 4:
            plagiarism_score = min(plagiarism_score, 35.0)
            size_penalty = True
        elif line_count <= 8:
            plagiarism_score = min(plagiarism_score, 60.0)
            size_penalty = True

        self.faiss_index.add(vector)
        self.repo.save_result(
            hashlib.sha256(code.encode()).hexdigest(),
            plagiarism_score,
            ai_probability,
            normalized_code,
            ast_features
        )

        reasoning = (
            "Small code sample; score capped to avoid overestimation."
            if size_penalty else
            "Similarity based on semantic, token, and structural overlap."
        )

        return {
            "plagiarism_percentage": round(plagiarism_score, 2),
            "ai_probability": round(ai_probability, 2),
            "confidence": "medium",
            "explanation": {
                "token_similarity": round(token_sim, 3),
                "semantic_similarity": round(semantic_sim, 3),
                "structure_similarity": round(structure_sim, 3),
                "code_lines": line_count,
                "size_penalty_applied": size_penalty,
                "reasoning": reasoning
            }
        }
