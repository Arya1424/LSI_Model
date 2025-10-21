"""
KD-Tree-based Candidate Retrieval Layer
Purpose: Accelerate inference by retrieving only the most similar statute embeddings
"""
import numpy as np
import pickle
from sklearn.neighbors import KDTree

class KDTreeRetriever:
    def __init__(self, k_neighbors=10, leaf_size=30):
        """
        Initialize KD-Tree retriever
        Args:
            k_neighbors: number of nearest neighbors to retrieve
            leaf_size: leaf size for KDTree
        """
        self.k_neighbors = k_neighbors
        self.leaf_size = leaf_size
        self.kdtree = None
        self.section_embeddings = None
        self.built = False
    
    def build(self, section_embeddings):
        """
        Build KD-Tree from section embeddings
        Args:
            section_embeddings: numpy array or torch tensor of shape (num_sections, embedding_dim)
        """
        # Convert to numpy if needed
        if hasattr(section_embeddings, 'cpu'):
            section_embeddings = section_embeddings.cpu().detach().numpy()
        
        self.section_embeddings = section_embeddings
        self.kdtree = KDTree(section_embeddings, leaf_size=self.leaf_size)
        self.built = True
        print(f"✓ KD-Tree built with {len(section_embeddings)} sections")
    
    def query(self, fact_embedding, k=None):
        """
        Query KD-Tree for nearest statute embeddings
        Args:
            fact_embedding: numpy array or torch tensor of shape (embedding_dim,) or (1, embedding_dim)
            k: number of neighbors (defaults to self.k_neighbors)
        Returns:
            distances: array of distances to k nearest neighbors
            indices: array of indices of k nearest neighbors
        """
        if not self.built:
            raise RuntimeError("KD-Tree not built. Call build() first.")
        
        k = k or self.k_neighbors
        
        # Convert to numpy if needed
        if hasattr(fact_embedding, 'cpu'):
            fact_embedding = fact_embedding.cpu().detach().numpy()
        
        # Ensure shape is (1, embedding_dim)
        if fact_embedding.ndim == 1:
            fact_embedding = fact_embedding.reshape(1, -1)
        
        # Query KD-Tree
        distances, indices = self.kdtree.query(fact_embedding, k=k)
        
        return distances[0], indices[0]
    
    def batch_query(self, fact_embeddings, k=None):
        """
        Query KD-Tree for multiple fact embeddings
        Args:
            fact_embeddings: numpy array or torch tensor of shape (batch_size, embedding_dim)
            k: number of neighbors (defaults to self.k_neighbors)
        Returns:
            distances: array of shape (batch_size, k)
            indices: array of shape (batch_size, k)
        """
        if not self.built:
            raise RuntimeError("KD-Tree not built. Call build() first.")
        
        k = k or self.k_neighbors
        
        # Convert to numpy if needed
        if hasattr(fact_embeddings, 'cpu'):
            fact_embeddings = fact_embeddings.cpu().detach().numpy()
        
        # Query KD-Tree
        distances, indices = self.kdtree.query(fact_embeddings, k=k)
        
        return distances, indices
    
    def save(self, filepath):
        """Save KD-Tree to disk"""
        if not self.built:
            raise RuntimeError("KD-Tree not built. Call build() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kdtree': self.kdtree,
                'section_embeddings': self.section_embeddings,
                'k_neighbors': self.k_neighbors,
                'leaf_size': self.leaf_size
            }, f)
        print(f"✓ KD-Tree saved to {filepath}")
    
    def load(self, filepath):
        """Load KD-Tree from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.kdtree = data['kdtree']
        self.section_embeddings = data['section_embeddings']
        self.k_neighbors = data['k_neighbors']
        self.leaf_size = data['leaf_size']
        self.built = True
        print(f"✓ KD-Tree loaded from {filepath}")
    
    def get_candidate_mask(self, fact_embedding, num_sections, k=None):
        """
        Get a binary mask for candidate sections
        Args:
            fact_embedding: embedding for a single fact
            num_sections: total number of sections
            k: number of candidates
        Returns:
            mask: binary array of shape (num_sections,) with 1 for candidates
        """
        _, indices = self.query(fact_embedding, k=k)
        mask = np.zeros(num_sections, dtype=bool)
        mask[indices] = True
        return mask