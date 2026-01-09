from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity
from src.pipeline.embedding import EmbeddingGenerator
from src.storage.faiss_index import FaissIndex
from src.pipeline.scorer import ScoreAggregator


class AnalysisPipeline:
    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.ast_analyzer = ASTAnalyzer()
        self.token_similarity = TokenSimilarity()
        self.embedding_generator = EmbeddingGenerator()
        self.scorer = ScoreAggregator()

        # FAISS index (CodeBERT = 768 dims)
        self.faiss_index = FaissIndex(vector_dim=768)

    def run(self, code: str, language: str | None = None) -> dict:

        # 1️⃣ Normalize
        normalized_code, _ = self.normalizer.normalize(code)

        # 2️⃣ AST analysis
        ast_features = {}
        if language == "python" or language is None:
            ast_features = self.ast_analyzer.analyze(normalized_code)

        # 3️⃣ Token similarity (self baseline)
        token_sim = self.token_similarity.jaccard_similarity(
            normalized_code,
            normalized_code
        )

        # 4️⃣ Embedding
        embedding = self.embedding_generator.generate(normalized_code)

        # 5️⃣ FAISS add & search (self baseline)
        self.faiss_index.add(embedding.detach().numpy())
        distances, _ = self.faiss_index.search(
            embedding.detach().numpy(), k=1
        )

        # Convert distance → similarity
        semantic_sim = 1 / (1 + distances[0])

        # 6️⃣ Structure similarity (temporary baseline)
        structure_sim = 1.0

        # 7️⃣ Final scoring
        plagiarism_score = self.scorer.compute_plagiarism_score(
            token_similarity=token_sim,
            semantic_similarity=semantic_sim,
            structure_similarity=structure_sim
        )

        ai_probability = self.scorer.compute_ai_probability(
            semantic_similarity=semantic_sim,
            structure_similarity=structure_sim
        )

        return {
            "plagiarism_percentage": plagiarism_score,
            "ai_probability": ai_probability,
            "confidence": "medium"
        }
