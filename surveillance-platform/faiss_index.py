import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import threading
import os

class FaissIndex:
    def __init__(self, dimension: int = 128, index_type: str = 'FlatL2'):
        """
        Initialize Faiss index for face embeddings

        Args:
            dimension: Dimension of face embeddings (128 for FaceNet)
            index_type: Type of Faiss index ('FlatL2', 'FlatIP', 'IVFFlat')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.id_map = {}  # Map from Faiss index to person_id
        self.reverse_id_map = {}  # Map from person_id to Faiss indices
        # Guards all reads/writes of the index and id maps. The face worker
        # thread searches the index every frame while request threads may
        # rebuild it (after registering/deleting persons or uploading faces).
        # faiss add/search are not safe to run concurrently, and rebuild_index
        # swaps self.index/id_map wholesale, so every access is serialized.
        # Reentrant so rebuild_index -> add_embeddings_batch doesn't deadlock.
        self._lock = threading.RLock()
        
    def _create_index(self) -> faiss.Index:
        """Create Faiss index based on specified type"""
        if self.index_type == 'FlatL2':
            # Exact search using L2 distance
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'FlatIP':
            # Exact search using inner product (cosine similarity)
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == 'IVFFlat':
            # Approximate search for larger datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            return index
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
            
    def add_embedding(self, embedding: np.ndarray, person_id: int):
        """
        Add a single embedding to the index
        
        Args:
            embedding: Face embedding vector
            person_id: ID of the person in the database
        """
        # Ensure embedding is the right shape
        embedding = embedding.reshape(1, -1).astype('float32')

        with self._lock:
            # Add to index
            idx = self.index.ntotal
            self.index.add(embedding)

            # Update ID mappings
            self.id_map[idx] = person_id
            if person_id not in self.reverse_id_map:
                self.reverse_id_map[person_id] = []
            self.reverse_id_map[person_id].append(idx)

    def add_embeddings_batch(self, embeddings: List[np.ndarray], person_ids: List[int]):
        """
        Add multiple embeddings to the index
        
        Args:
            embeddings: List of face embedding vectors
            person_ids: List of person IDs corresponding to embeddings
        """
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')

        with self._lock:
            # Get starting index
            start_idx = self.index.ntotal

            # Add to index
            self.index.add(embeddings_array)

            # Update ID mappings
            for i, person_id in enumerate(person_ids):
                idx = start_idx + i
                self.id_map[idx] = person_id
                if person_id not in self.reverse_id_map:
                    self.reverse_id_map[person_id] = []
                self.reverse_id_map[person_id].append(idx)
            
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               threshold: float = 0.6) -> List[Tuple[int, float]]:
        """
        Search for similar faces in the index
        
        Args:
            query_embedding: Query face embedding
            k: Number of nearest neighbors to return
            threshold: Distance threshold for valid matches
            
        Returns:
            List of (person_id, distance) tuples
        """
        # Ensure embedding is the right shape
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        with self._lock:
            # Search index
            distances, indices = self.index.search(query_embedding, k)

            # Filter results by threshold and map to person IDs
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                distance = distances[0][i]

                if idx >= 0 and distance < threshold:  # Valid match
                    person_id = self.id_map.get(idx)
                    if person_id is not None:
                        results.append((person_id, distance))

        return results
        
    def remove_person(self, person_id: int):
        """
        Remove all embeddings for a person from the index
        
        Args:
            person_id: ID of the person to remove
        """
        with self._lock:
            if person_id not in self.reverse_id_map:
                return

            # Get indices to remove
            indices_to_remove = self.reverse_id_map[person_id]

            # Note: Faiss doesn't support efficient removal
            # In production, you might need to rebuild the index
            # or use a different strategy

            # Remove from mappings
            for idx in indices_to_remove:
                del self.id_map[idx]
            del self.reverse_id_map[person_id]

        print(f"Note: Faiss doesn't support efficient removal. Consider rebuilding index.")
        
    def save_index(self, filepath: str):
        """Save index and mappings to disk"""
        # Save Faiss index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save ID mappings
        mappings = {
            'id_map': self.id_map,
            'reverse_id_map': self.reverse_id_map
        }
        with open(f"{filepath}.mappings", 'wb') as f:
            pickle.dump(mappings, f)
            
    def load_index(self, filepath: str):
        """Load index and mappings from disk"""
        # Load ID mappings first, then swap the index in under the lock so a
        # concurrent search never sees a new index with stale mappings.
        with open(f"{filepath}.mappings", 'rb') as f:
            mappings = pickle.load(f)
        loaded_index = faiss.read_index(f"{filepath}.index")
        with self._lock:
            self.index = loaded_index
            self.id_map = mappings['id_map']
            self.reverse_id_map = mappings['reverse_id_map']

    def get_statistics(self) -> Dict:
        """Get statistics about the index"""
        with self._lock:
            return {
                'total_embeddings': self.index.ntotal,
                'unique_persons': len(self.reverse_id_map),
                'index_type': self.index_type,
                'dimension': self.dimension
            }
        
    def rebuild_index(self, embeddings_data: List[Dict]):
        """
        Rebuild index from scratch with new data
        
        Args:
            embeddings_data: List of dicts with 'embedding' and 'person_id' keys
        """
        with self._lock:
            # Create new index
            self.index = self._create_index()
            self.id_map = {}
            self.reverse_id_map = {}

            # Add all embeddings
            if embeddings_data:
                embeddings = [data['embedding'] for data in embeddings_data]
                person_ids = [data['person_id'] for data in embeddings_data]
                self.add_embeddings_batch(embeddings, person_ids)