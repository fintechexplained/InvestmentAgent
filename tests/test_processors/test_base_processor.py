"""Tests for base processor abstract class."""

import pytest
from pathlib import Path
from typing import List
from src.processors.base_processor import (
    BaseModalityProcessor,
    ProcessedChunk,
    ProcessingResult,
)


class MockProcessor(BaseModalityProcessor):
    """Mock processor for testing abstract base class."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".mock", ".test"]

    @property
    def modality_type(self) -> str:
        return "test"

    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        """Mock process implementation."""
        chunk = ProcessedChunk(
            content="Test content",
            metadata={"test": "data"},
            chunk_id="test-123",
            company_name=company_name,
            source_file=str(file_path),
            modality=self.modality_type,
        )
        return ProcessingResult(
            chunks=[chunk],
            summary="Test summary",
            extracted_metrics={"test_metric": 100},
            processing_metadata={"status": "success"},
        )

    async def validate_file(self, file_path: Path) -> bool:
        """Mock validation implementation."""
        return file_path.exists() and file_path.suffix in self.supported_extensions


class TestProcessedChunk:
    """Test ProcessedChunk model."""

    def test_processed_chunk_creation(self):
        """Test creating a ProcessedChunk."""
        chunk = ProcessedChunk(
            content="Test content",
            metadata={"key": "value"},
            chunk_id="chunk-001",
            company_name="TestCompany",
            source_file="test.txt",
            modality="text",
        )

        assert chunk.content == "Test content"
        assert chunk.metadata == {"key": "value"}
        assert chunk.chunk_id == "chunk-001"
        assert chunk.company_name == "TestCompany"
        assert chunk.source_file == "test.txt"
        assert chunk.modality == "text"
        assert chunk.embedding is None

    def test_processed_chunk_with_embedding(self):
        """Test ProcessedChunk with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = ProcessedChunk(
            content="Test",
            metadata={},
            chunk_id="test",
            company_name="Test",
            source_file="test.txt",
            modality="text",
            embedding=embedding,
        )

        assert chunk.embedding == embedding


class TestProcessingResult:
    """Test ProcessingResult model."""

    def test_processing_result_creation(self):
        """Test creating a ProcessingResult."""
        chunk = ProcessedChunk(
            content="Test",
            metadata={},
            chunk_id="test",
            company_name="Test",
            source_file="test.txt",
            modality="text",
        )

        result = ProcessingResult(
            chunks=[chunk],
            summary="Test summary",
            extracted_metrics={"revenue": 1000000},
            processing_metadata={"duration": 1.5},
        )

        assert len(result.chunks) == 1
        assert result.summary == "Test summary"
        assert result.extracted_metrics == {"revenue": 1000000}
        assert result.processing_metadata == {"duration": 1.5}

    def test_processing_result_defaults(self):
        """Test ProcessingResult with default values."""
        chunk = ProcessedChunk(
            content="Test",
            metadata={},
            chunk_id="test",
            company_name="Test",
            source_file="test.txt",
            modality="text",
        )

        result = ProcessingResult(chunks=[chunk], summary="Summary")

        assert result.extracted_metrics == {}
        assert result.processing_metadata == {}


class TestBaseModalityProcessor:
    """Test BaseModalityProcessor abstract class."""

    @pytest.mark.asyncio
    async def test_mock_processor_properties(self):
        """Test processor properties."""
        processor = MockProcessor()

        assert processor.supported_extensions == [".mock", ".test"]
        assert processor.modality_type == "test"

    @pytest.mark.asyncio
    async def test_mock_processor_process(self, tmp_path):
        """Test processor process method."""
        processor = MockProcessor()
        test_file = tmp_path / "test.mock"
        test_file.write_text("test content")

        result = await processor.process(test_file, "TestCompany")

        assert isinstance(result, ProcessingResult)
        assert len(result.chunks) == 1
        assert result.chunks[0].company_name == "TestCompany"
        assert result.summary == "Test summary"
        assert result.extracted_metrics == {"test_metric": 100}

    @pytest.mark.asyncio
    async def test_mock_processor_validate_file(self, tmp_path):
        """Test file validation."""
        processor = MockProcessor()

        # Valid file
        valid_file = tmp_path / "test.mock"
        valid_file.write_text("content")
        assert await processor.validate_file(valid_file) is True

        # Invalid extension
        invalid_file = tmp_path / "test.invalid"
        invalid_file.write_text("content")
        assert await processor.validate_file(invalid_file) is False

        # Non-existent file
        missing_file = tmp_path / "missing.mock"
        assert await processor.validate_file(missing_file) is False

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModalityProcessor()
