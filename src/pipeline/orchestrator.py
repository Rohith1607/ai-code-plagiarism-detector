import hashlib
from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity
from src.pipeline.embedding import EmbeddingGenerator
from src.storage.faiss_index import FaissIndex
from src.pipeline.scorer import ScoreAggregator
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

    def run(self, code: str, language: str | None = None) -> dict:

        # 1️⃣ Normalize
        normalized_code, _ = self.normalizer.normalize(code)

        # 2️⃣ AST
        if language == "python" or language is None:
            _ = self.ast_analyzer.analyze(normalized_code)

        # 3️⃣ Token similarity (self baseline for now)
        token_sim = self.token_similarity.jaccard_similarity(
            normalized_code, normalized_code
        )

        # 4️⃣ Embedding
        embedding = self.embedding_generator.generate(normalized_code)
        vector = embedding.detach().numpy()

        # 5️⃣ FAISS SEARCH (REAL PART)
        if self.faiss_index.size() > 0:
            distances, _ = self.faiss_index.search(vector, k=1)
            best_distance = distances[0]
            semantic_sim = 1 / (1 + best_distance)
        else:
            semantic_sim = 0.0  # first ever submission

        # 6️⃣ Structure similarity (still baseline)
        structure_sim = 1.0

        # 7️⃣ Score aggregation
        plagiarism_score = self.scorer.compute_plagiarism_score(
            token_similarity=token_sim,
            semantic_similarity=semantic_sim,
            structure_similarity=structure_sim
        )

        ai_probability = self.scorer.compute_ai_probability(
            semantic_similarity=semantic_sim,
            structure_similarity=structure_sim
        )

        # 8️⃣ STORE AFTER COMPARISON
        self.faiss_index.add(vector)

        code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.repo.save_result(
            code_hash=code_hash,
            plagiarism_score=plagiarism_score,
            ai_probability=ai_probability
        )

        return {
            "plagiarism_percentage": plagiarism_score,
            "ai_probability": ai_probability,
            "confidence": "medium"
        }
