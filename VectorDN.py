import numpy as np
from typing import Dict, List, Tuple

class VectorDB:
    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}  # Dictionary to store vectors
    
    def add_vector(self, vec_id: str, vector: np.ndarray) -> None:
        """Add a vector to the database."""
        if vec_id in self.vectors:
            raise ValueError(f"Vector with id {vec_id} already exists.")
        self.vectors[vec_id] = vector
    
    def get_vector(self, vec_id: str) -> np.ndarray:
        """Retrieve a vector by ID."""
        return self.vectors.get(vec_id, None)
    
    def delete_vector(self, vec_id: str) -> None:
        """Delete a vector by ID."""
        if vec_id in self.vectors:
            del self.vectors[vec_id]
        else:
            raise KeyError(f"Vector with id {vec_id} not found.")
from numpy.linalg import norm

class VectorDB(VectorDB):
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the top_k most similar vectors to the query vector."""
        similarities = []
        
        for vec_id, vector in self.vectors.items():
            sim = self.cosine_similarity(query_vector, vector)
            similarities.append((vec_id, sim))
        
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class SimpleVectorDB(VectorDB):
    def update_vector(self, vec_id: str, vector: np.ndarray) -> None:
        """Update an existing vector."""
        if vec_id not in self.vectors:
            raise KeyError(f"Vector with id {vec_id} not found.")
        self.vectors[vec_id] = vector
