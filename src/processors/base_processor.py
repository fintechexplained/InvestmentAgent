"""Base processor abstract class for all modality processors."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessedChunk(BaseModel):
    """Represents a processed chunk of data.

    Attributes:
        content: The text content of the chunk
        metadata: Additional metadata about the chunk
        chunk_id: Unique identifier for the chunk
        company_name: Name of the company this chunk relates to
        source_file: Path to the source file
        modality: Type of modality (text, audio, image)
        embedding: Optional vector embedding for the chunk
    """

    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    company_name: str
    source_file: str
    modality: str
    embedding: Optional[List[float]] = None

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class ProcessingResult(BaseModel):
    """Result of processing a file.

    Attributes:
        chunks: List of processed chunks
        summary: High-level summary of the content
        extracted_metrics: Key metrics extracted from the content
        processing_metadata: Metadata about the processing operation
    """

    chunks: List[ProcessedChunk]
    summary: str
    extracted_metrics: Dict[str, Any] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class BaseModalityProcessor(ABC):
    """Abstract base class for all modality processors.

    This class defines the interface that all modality processors must implement.
    Each processor handles a specific type of data (text, audio, image, etc.)
    """

    def __init__(self) -> None:
        """Initialize the processor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Get file extensions this processor can handle.

        Returns:
            List of file extensions (e.g., ['.txt', '.md'])
        """
        pass

    @property
    @abstractmethod
    def modality_type(self) -> str:
        """Get the type of modality this processor handles.

        Returns:
            Modality type: 'text', 'audio', 'image', etc.
        """
        pass

    @abstractmethod
    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        """Process a file and return structured results.

        Args:
            file_path: Path to the file to process
            company_name: Name of the company

        Returns:
            ProcessingResult containing chunks and metadata

        Raises:
            ValueError: If file is invalid or cannot be processed
            IOError: If file cannot be read
        """
        pass

    @abstractmethod
    async def validate_file(self, file_path: Path) -> bool:
        """Validate that a file can be processed.

        Args:
            file_path: Path to the file to validate

        Returns:
            True if file is valid and can be processed, False otherwise
        """
        pass
