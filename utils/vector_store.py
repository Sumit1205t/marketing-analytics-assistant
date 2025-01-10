import chromadb
import numpy as np
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "vector_db"):
        """Initialize vector store with ChromaDB."""
        try:
            self.persist_directory = persist_directory
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)
                
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="document_store",
                metadata={"description": "Document storage for CMO Assistant"}
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to vector store."""
        try:
            for doc in documents:
                # Create unique document ID
                doc_id = f"{doc['filename']}_{datetime.now().timestamp()}"
                
                # Add document to collection
                self.collection.add(
                    documents=[doc['content']],
                    metadatas=[{
                        'filename': doc['filename'],
                        'doc_type': doc['doc_type'],
                        'processed_at': doc['processed_at']
                    }],
                    ids=[doc_id]
                )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents using vector similarity."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return [{
                'content': doc,
                'metadata': meta,
                'distance': dist
            } for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )]
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return [] 