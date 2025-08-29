import faiss
import numpy as np
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
from src.utils import embed_semantic, embed_emotion, semantic_embedder
from src.config import config
from src.logging import logger # Import logger

class VectorStore:
    def __init__(self, dimension: int, faiss_dir=config.get("general.faiss_dir")):
        self.dimension = dimension
        self.faiss_dir = faiss_dir
        os.makedirs(self.faiss_dir, exist_ok=True)

        self.semantic_index_path = os.path.join(self.faiss_dir, "semantic.faiss")
        self.emotion_index_path = os.path.join(self.faiss_dir, "emotion.faiss")
        self.metadata_path = os.path.join(self.faiss_dir, "metadata.json")

        self.semantic_index = self._load_or_create_index(self.semantic_index_path)
        self.emotion_index = self._load_or_create_index(self.emotion_index_path)
        self.metadata = self._load_or_create_metadata(self.metadata_path)
        self.doc_id_to_idx = {meta["doc_id"]: i for i, meta in enumerate(self.metadata)}

    def _load_or_create_index(self, path: str):
        if os.path.exists(path):
            return faiss.read_index(path)
        else:
            # Using IndexFlatL2 for simplicity, can be changed to IVFFlat for performance
            return faiss.IndexFlatL2(self.dimension)

    def _load_or_create_metadata(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    def _save_indices(self):
        faiss.write_index(self.semantic_index, self.semantic_index_path)
        faiss.write_index(self.emotion_index, self.emotion_index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Adds a list of KB chunks to the vector store."""
        semantic_embeddings = []
        emotion_embeddings = []
        new_metadata = []

        for chunk in chunks:
            doc_id = chunk.get("doc_id", str(uuid.uuid4()))
            text = chunk["text"]
            meta = chunk.get("meta", {})

            # Generate embeddings
            sem_embed = embed_semantic(text)
            # For emotion embedding of a chunk, we might need a pre-classified emotion or a way to infer it.
            # For now, let's assume a neutral emotion for KB chunks unless specified.
            # A more advanced approach would pre-classify emotions for KB chunks.
            # Here, we'll use a dummy emotion embedding for chunks.
            # This needs to be consistent with how embed_emotion is called for user input.
            # Let's use a zero vector for now, or a specific "neutral" emotion embedding.
            # For consistency with `embed_emotion` in `utils.py`, we'll create a dummy emotion dict.
            dummy_emotion_data = {"primary": "neutral", "scores": {"neutral": 1.0}}
            em_embed = embed_emotion(dummy_emotion_data)

            semantic_embeddings.append(sem_embed)
            emotion_embeddings.append(em_embed)
            new_metadata.append({
                "doc_id": doc_id,
                "source": chunk.get("source"),
                "text": text,
                "meta": meta,
                "updated_at": chunk.get("updated_at", datetime.now().isoformat())
            })

        if semantic_embeddings:
            self.semantic_index.add(np.array(semantic_embeddings).astype('float32'))
            self.emotion_index.add(np.array(emotion_embeddings).astype('float32'))
            # Update metadata and doc_id_to_idx mapping
            start_idx = len(self.metadata)
            self.metadata.extend(new_metadata)
            for i, meta in enumerate(new_metadata):
                self.doc_id_to_idx[meta["doc_id"]] = start_idx + i
            self._save_indices()

    def clear_vector_store(self):
        """Clears all documents and indices from the vector store."""
        self.semantic_index = faiss.IndexFlatL2(self.dimension)
        self.emotion_index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.doc_id_to_idx = {}
        self._save_indices()
        logger.info("Vector store cleared.") # Changed from print to logger.info
        
    def vector_search(self, query_semantic_embed: List[float], query_emotion_embed: List[float], filters: Dict[str, Any] = None, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Performs a vector search across semantic and emotion indices.
        Filters are applied post-retrieval for simplicity.
        """
        # Search semantic index
        D_sem, I_sem = self.semantic_index.search(np.array([query_semantic_embed]).astype('float32'), top_k * 2) # Retrieve more to filter
        
        # Search emotion index
        D_em, I_em = self.emotion_index.search(np.array([query_emotion_embed]).astype('float32'), top_k * 2)

        # Combine results and re-rank (simplified for now, actual re-ranking happens in orchestration)
        # For now, we'll just get unique indices from both searches
        combined_indices = np.unique(np.concatenate((I_sem.flatten(), I_em.flatten())))
        
        results = []
        for idx in combined_indices:
            if idx == -1: # FAISS can return -1 for empty slots
                continue
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                # Apply hard filters
                if filters:
                    match = True
                    for key, allowed_values in filters.items():
                        # Handle nested keys like "meta.therapy"
                        current_value = chunk
                        for sub_key in key.split('.'):
                            current_value = current_value.get(sub_key) if isinstance(current_value, dict) else None
                            if current_value is None:
                                match = False
                                break
                        if match and current_value not in allowed_values:
                            match = False
                            break
                    if not match:
                        continue
                results.append(chunk)
        
        # In a real scenario, you'd re-score and sort these results based on combined similarity
        # For now, we'll just return the filtered results up to top_k
        return results[:top_k]

# Initialize the vector store with the dimension of the embeddings
# Assuming semantic_embedder is initialized in utils.py and has get_sentence_embedding_dimension()
vector_store = VectorStore(dimension=semantic_embedder.get_sentence_embedding_dimension())
