"""Vector store manager using FAISS."""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from ..processors.base_processor import ProcessedChunk
from ..llm.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages FAISS vector store for investment data."""

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        """Initialize the vector store.

        Args:
            persist_directory: Optional directory to persist data (None for in-memory)
        """
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.embedding_generator = EmbeddingGenerator()

        # Initialize FAISS index (will be created after first embedding)
        self.index = None
        self.dimension = None

        # Store metadata separately (FAISS only stores vectors)
        self.documents = []  # List of document texts
        self.metadatas = []  # List of metadata dicts
        self.ids = []  # List of chunk IDs

        # Load existing index if persist directory exists
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._load_index()
            logger.info(f"Initialized FAISS vector store at {persist_directory}")
        else:
            logger.info("Initialized in-memory FAISS vector store")

    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        index_path = self.persist_directory / "faiss.index"
        metadata_path = self.persist_directory / "metadata.pkl"

        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                self.dimension = self.index.d

                # Load metadata
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.metadatas = data["metadatas"]
                    self.ids = data["ids"]

                logger.info(f"Loaded existing index with {len(self.ids)} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Starting fresh.")
                self.index = None
                self.documents = []
                self.metadatas = []
                self.ids = []

    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        if not self.persist_directory or not self.index:
            return

        try:
            index_path = self.persist_directory / "faiss.index"
            metadata_path = self.persist_directory / "metadata.pkl"

            # Save FAISS index
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                    "ids": self.ids
                }, f)

            logger.debug(f"Saved index with {len(self.ids)} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    async def add_chunks(self, chunks: List[ProcessedChunk]) -> None:
        """Add processed chunks to vector store.

        Args:
            chunks: List of processed chunks to add
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        try:
            logger.info(f"Adding {len(chunks)} chunks to vector store")

            # Filter out chunks with empty content
            valid_chunks = [c for c in chunks if c.content and c.content.strip()]
            if len(valid_chunks) < len(chunks):
                skipped = len(chunks) - len(valid_chunks)
                logger.warning(f"Skipping {skipped} chunks with empty content")

            if not valid_chunks:
                logger.warning("No valid chunks to add after filtering")
                return

            # Generate embeddings if not present
            texts = []
            chunks_needing_embeddings = []

            for chunk in valid_chunks:
                if chunk.embedding is None:
                    texts.append(chunk.content)
                    chunks_needing_embeddings.append(chunk)

            if texts:
                logger.debug(f"Generating embeddings for {len(texts)} chunks")
                embeddings = await self.embedding_generator.generate_embeddings(texts)

                # Update chunks with embeddings
                for chunk, embedding in zip(chunks_needing_embeddings, embeddings):
                    chunk.embedding = embedding

            # Prepare embeddings array (now only from valid chunks)
            embeddings_array = np.array([chunk.embedding for chunk in valid_chunks], dtype=np.float32)

            # Initialize index if needed
            if self.index is None:
                self.dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created new FAISS index with dimension {self.dimension}")

            # Add to FAISS index
            self.index.add(embeddings_array)

            # Add metadata (only for valid chunks)
            for chunk in valid_chunks:
                self.ids.append(chunk.chunk_id)
                self.documents.append(chunk.content)
                self.metadatas.append({
                    "company_name": chunk.company_name,
                    "source_file": chunk.source_file,
                    "modality": chunk.modality,
                    **chunk.metadata,
                })

            # Save to disk if persist directory is set
            self._save_index()

            logger.info(f"Successfully added {len(valid_chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    async def query(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query vector store with optional filters.

        Args:
            query_text: Query text
            filters: Optional metadata filters
            n_results: Number of results to return

        Returns:
            List of query results with documents and metadata
        """
        try:
            logger.debug(f"Querying vector store: '{query_text[:50]}...'")

            if self.index is None or len(self.ids) == 0:
                logger.warning("Vector store is empty")
                return []

            # Generate embedding for query
            query_embedding = await self.embedding_generator.generate_embedding(query_text)
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Search in FAISS
            # Get more results than needed to allow for filtering
            search_k = min(len(self.ids), n_results * 5 if filters else n_results)
            distances, indices = self.index.search(query_vector, search_k)

            # Format results and apply filters
            formatted_results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                metadata = self.metadatas[idx]

                # Apply filters if provided
                if filters:
                    match = True
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                formatted_results.append({
                    "id": self.ids[idx],
                    "document": self.documents[idx],
                    "metadata": metadata,
                    "distance": float(distance),
                })

                # Stop once we have enough results
                if len(formatted_results) >= n_results:
                    break

            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise

    async def get_companies(self) -> List[str]:
        """Get list of all companies in the store.

        Returns:
            List of unique company names
        """
        try:
            companies = set()
            for metadata in self.metadatas:
                if "company_name" in metadata:
                    companies.add(metadata["company_name"])

            company_list = sorted(list(companies))
            logger.info(f"Found {len(company_list)} companies in vector store")
            return company_list

        except Exception as e:
            logger.error(f"Error getting companies: {e}")
            return []

    async def get_company_stats(self, company_name: str) -> Dict[str, Any]:
        """Get statistics about stored data for a company.

        Args:
            company_name: Name of the company

        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "company_name": company_name,
                "total_chunks": 0,
                "modalities": {},
                "source_files": set(),
            }

            for metadata in self.metadatas:
                if metadata.get("company_name") == company_name:
                    stats["total_chunks"] += 1

                    # Count by modality
                    modality = metadata.get("modality", "unknown")
                    stats["modalities"][modality] = stats["modalities"].get(modality, 0) + 1

                    # Track source files
                    if "source_file" in metadata:
                        stats["source_files"].add(metadata["source_file"])

            stats["source_files"] = list(stats["source_files"])
            logger.debug(f"Retrieved stats for {company_name}: {stats['total_chunks']} chunks")
            return stats

        except Exception as e:
            logger.error(f"Error getting company stats: {e}")
            return {"company_name": company_name, "error": str(e)}

    def clear(self) -> None:
        """Clear all data from the vector store."""
        try:
            self.index = None
            self.dimension = None
            self.documents = []
            self.metadatas = []
            self.ids = []

            # Delete persisted files if they exist
            if self.persist_directory:
                index_path = self.persist_directory / "faiss.index"
                metadata_path = self.persist_directory / "metadata.pkl"

                if index_path.exists():
                    index_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()

            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise

    def count(self) -> int:
        """Get total count of chunks in the store.

        Returns:
            Total number of chunks
        """
        try:
            return len(self.ids)
        except Exception as e:
            logger.error(f"Error counting chunks: {e}")
            return 0
