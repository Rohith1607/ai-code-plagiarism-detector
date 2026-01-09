import faiss
import numpy as np


class FaissIndex:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)

    def add(self, vector: np.ndarray):
        vector = np.array(vector).astype("float32").reshape(1, -1)
        self.index.add(vector)

    def search(self, vector: np.ndarray, k: int = 3):
        if self.index.ntotal == 0:
            return [], []

        vector = np.array(vector).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(vector, k)
        return distances[0], indices[0]

    def size(self):
        return self.index.ntotal
