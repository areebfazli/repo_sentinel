import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from backend.app.config import settings

class VectorStore:
    def __init__(self):
        """
        Initialize the Qdrant client.
        Uses local file-based storage for dev, and connects to a server for prod.
        """
        if settings.is_dev:
            # File-based database (No Docker required)
            self.client = QdrantClient(path=settings.qdrant_local_path)
        else:
            # Server-based database
            self.client = QdrantClient(
                host=settings.QDRANT_HOST, 
                port=settings.QDRANT_PORT
            )
            
        self.cve_collection = "cve_corpus"
        self.team_collection = "team_history"
        
        # CodeBERT hidden size
        self.vector_size = 768 
        
        self._init_collections()

    def _init_collections(self):
        """Ensure both collections exist, creating them if they don't."""
        existing_collections = [c.name for c in self.client.get_collections().collections]
        
        for collection_name in [self.cve_collection, self.team_collection]:
            if collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )

    def insert_cves(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Insert embedded CVEs into the Ghost Hunter pipeline."""
        self._insert(self.cve_collection, vectors, payloads)
        
    def insert_team_history(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Insert embedded PRs/Commits into the Team Memory pipeline."""
        self._insert(self.team_collection, vectors, payloads)

    def _insert(self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Helper to insert vectors into a specific collection."""
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            for vector, payload in zip(vectors, payloads)
        ]
        
        # Upsert in batches to avoid payload limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i + batch_size]
            )

    def search_cves(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Find CVEs similar to the given code vector."""
        return self._search(self.cve_collection, query_vector, limit)
        
    def search_team_history(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Find team history similar to the given code vector."""
        return self._search(self.team_collection, query_vector, limit)

    def _search(self, collection_name: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Helper to perform ANN search."""
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        results = []
        for hit in hits:
            # Reconstruct dictionary with score
            res = hit.payload.copy() if hit.payload else {}
            res["similarity_score"] = hit.score
            results.append(res)
            
        return results
