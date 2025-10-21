"""
KD-Tree-based Candidate Retrieval Layer
Purpose: Accelerate inference by retrieving only the most similar statute embeddings.
"""
import numpy as np
from sklearn.neighbors import KDTree
import torch
from typing import Union, List

class KDTreeRetriever:
    def __init__(self, k_neighbors: int = 10, leaf_size: int = 30):
        self.k_neighbors = k_neighbors
        self.leaf_size = leaf_size
        self.kdtree = None
        self.built = False
    
    def build(self, section_embeddings: Union[np.ndarray, torch.Tensor]):
        """Build KD-Tree from section embeddings (Structural Embeddings h_s^(s))"""
        if isinstance(section_embeddings, torch.Tensor):
            section_embeddings = section_embeddings.cpu().detach().numpy()
        
        if section_embeddings.ndim != 2:
             raise ValueError("Section embeddings must be 2D array (num_sections, embedding_dim).")

        self.kdtree = KDTree(section_embeddings, leaf_size=self.leaf_size)
        self.built = True
        print(f"✓ KD-Tree built with {len(section_embeddings)} sections")
    
    def query(self, fact_embedding: Union[np.ndarray, torch.Tensor], k: int = None) -> np.ndarray:
        """
        Query KD-Tree for nearest statute indices (for a single fact).
        Returns: indices: array of indices of k nearest neighbors, shape (k,)
        """
        if not self.built:
            raise RuntimeError("KD-Tree not built. Call build() first.")
        
        k = k or self.k_neighbors
        
        if isinstance(fact_embedding, torch.Tensor):
            fact_embedding = fact_embedding.cpu().detach().numpy()
        
        if fact_embedding.ndim == 1:
            fact_embedding = fact_embedding.reshape(1, -1)
        
        distances, indices = self.kdtree.query(fact_embedding, k=k)
        
        return indices[0]

    def batch_query(self, fact_embeddings: Union[np.ndarray, torch.Tensor], k: int = None) -> np.ndarray:
        """
        Query KD-Tree for multiple fact embeddings (for a batch).
        Returns: indices: array of indices of k nearest neighbors, shape (batch_size, k)
        """
        if not self.built:
            raise RuntimeError("KD-Tree not built. Call build() first.")
        
        k = k or self.k_neighbors
        
        if isinstance(fact_embeddings, torch.Tensor):
            fact_embeddings = fact_embeddings.cpu().detach().numpy()
        
        distances, indices = self.kdtree.query(fact_embeddings, k=k)
        
        return indices