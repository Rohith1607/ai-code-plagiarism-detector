# src/storage/faiss_index.py

import faiss
import numpy as np


class FaissIndex:
    """
    Manages FAISS vector index.
    """

    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)

    def add(self, vector: np.ndarray):
        """
        Add a vector to the index.
        """
        vector = np.array(vector).astype("float32").reshape(1, -1)
        self.index.add(vector)

    def search(self, vector: np.ndarray, k: int = 3):
        """
        Search for top-k similar vectors.
        """
        vector = np.array(vector).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(vector, k)
        return distances[0], indices[0]
