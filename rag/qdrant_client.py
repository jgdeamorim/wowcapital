from __future__ import annotations
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore
    PointStruct = None  # type: ignore


class RAGStore:
    def __init__(self, url: str, api_key: Optional[str] = None, collection: str = "wow_docs"):
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key) if QdrantClient else None

    def enabled(self) -> bool:
        return self.client is not None

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        if not self.enabled():
            return
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled():
            return []
        res = self.client.search(collection_name=self.collection, query_vector=vector, limit=limit)
        return [r.dict() for r in res]

