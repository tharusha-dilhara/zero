import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

class QdrantMemory:
    def __init__(self, url, collection_name, embedding_model="all-MiniLM-L6-v2"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create a collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    def _get_embedding(self, text):
        """Generate an embedding for the given text."""
        return self.embedding_model.encode(text)
    
    def add_memory(self, text, metadata=None, user_id=None):
        """Add a memory to the vector database."""
        if metadata is None:
            metadata = {}
        
        if user_id:
            metadata["user_id"] = user_id
            
        embedding = self._get_embedding(text)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=embedding.tolist(),
                    payload={
                        "text": text,
                        **metadata
                    }
                )
            ]
        )
    
    def search_memories(self, query, limit=5, user_id=None):
        """Search for similar memories."""
        query_embedding = self._get_embedding(query)
        
        filter_condition = None
        if user_id:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            query_filter=filter_condition
        )
        
        return [
            {
                "text": hit.payload.get("text"),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score
            } 
            for hit in results
        ]
