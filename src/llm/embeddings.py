"""Embedding generation utilities."""

import logging
from typing import List
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI's embedding models."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """Initialize the embedding generator.

        Args:
            model: The embedding model to use
        """
        self.client = AsyncOpenAI()
        self.model = model
        logger.info(f"Initialized EmbeddingGenerator with model: {model}")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Validate input type
            if text is None or not isinstance(text, str):
                raise ValueError(f"Invalid text input: expected str, got {type(text)}")

            # Remove any null characters that might cause issues
            text = text.replace('\x00', '')

            # Ensure text is not empty after cleaning
            if not text.strip():
                raise ValueError("Text is empty after cleaning")

            # Truncate text if too long (max 8191 tokens for embedding models)
            if len(text) > 8000:
                text = text[:8000]
                logger.debug("Truncated text to 8000 characters for embedding")

            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Validate and clean inputs
            cleaned_texts = []
            skipped_indices = []
            for i, text in enumerate(texts):
                if text is None or not isinstance(text, str):
                    raise ValueError(f"Invalid text at index {i}: expected str, got {type(text)}")

                # Remove null characters
                text = text.replace('\x00', '')

                # Skip empty texts instead of failing
                if not text.strip():
                    logger.warning(f"Skipping empty text at index {i}")
                    skipped_indices.append(i)
                    continue

                # Truncate if too long
                if len(text) > 8000:
                    text = text[:8000]

                cleaned_texts.append(text)

            if not cleaned_texts:
                raise ValueError("All texts are empty after cleaning")

            response = await self.client.embeddings.create(
                model=self.model,
                input=cleaned_texts,
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
