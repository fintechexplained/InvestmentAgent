"""Tests for processor registry."""

import pytest
from pathlib import Path
from typing import List
from src.processors.registry import ProcessorRegistry
from src.processors.base_processor import (
    BaseModalityProcessor,
    ProcessedChunk,
    ProcessingResult,
)


class TestProcessorA(BaseModalityProcessor):
    """Mock processor A for testing."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".md"]

    @property
    def modality_type(self) -> str:
        return "text"

    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        chunk = ProcessedChunk(
            content="Test A",
            metadata={},
            chunk_id="a-1",
            company_name=company_name,
            source_file=str(file_path),
            modality=self.modality_type,
        )
        return ProcessingResult(chunks=[chunk], summary="Summary A")

    async def validate_file(self, file_path: Path) -> bool:
        return file_path.exists() and file_path.suffix in self.supported_extensions


class TestProcessorB(BaseModalityProcessor):
    """Mock processor B for testing."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".mp3", ".wav"]

    @property
    def modality_type(self) -> str:
        return "audio"

    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        chunk = ProcessedChunk(
            content="Test B",
            metadata={},
            chunk_id="b-1",
            company_name=company_name,
            source_file=str(file_path),
            modality=self.modality_type,
        )
        return ProcessingResult(chunks=[chunk], summary="Summary B")

    async def validate_file(self, file_path: Path) -> bool:
        return file_path.exists() and file_path.suffix in self.supported_extensions


class TestProcessorRegistry:
    """Test ProcessorRegistry class."""

    def test_registry_initialization(self):
        """Test creating a new registry."""
        registry = ProcessorRegistry()

        assert registry is not None
        assert registry.list_supported_extensions() == []

    def test_register_processor(self):
        """Test registering a processor."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)

        extensions = registry.list_supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions

    def test_register_multiple_processors(self):
        """Test registering multiple processors."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)
        registry.register(TestProcessorB)

        extensions = registry.list_supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".mp3" in extensions
        assert ".wav" in extensions

    def test_get_processor_by_extension(self, tmp_path):
        """Test getting appropriate processor for a file."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)
        registry.register(TestProcessorB)

        # Test text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")
        processor_txt = registry.get_processor(txt_file)
        assert processor_txt is not None
        assert isinstance(processor_txt, TestProcessorA)

        # Test audio file
        mp3_file = tmp_path / "test.mp3"
        mp3_file.write_bytes(b"audio")
        processor_mp3 = registry.get_processor(mp3_file)
        assert processor_mp3 is not None
        assert isinstance(processor_mp3, TestProcessorB)

    def test_get_processor_for_unsupported_extension(self, tmp_path):
        """Test getting processor for unsupported file type."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)

        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")

        processor = registry.get_processor(unsupported_file)
        assert processor is None

    def test_extension_case_insensitivity(self, tmp_path):
        """Test that file extensions are case-insensitive."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)

        # Test uppercase extension
        file_upper = tmp_path / "test.TXT"
        file_upper.write_text("content")
        processor = registry.get_processor(file_upper)
        assert processor is not None
        assert isinstance(processor, TestProcessorA)

    def test_register_same_processor_twice(self):
        """Test registering the same processor twice (should be idempotent)."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)
        registry.register(TestProcessorA)

        extensions = registry.list_supported_extensions()
        # Should still have the extensions, not duplicated
        assert extensions.count(".txt") <= 1
        assert extensions.count(".md") <= 1

    @pytest.mark.asyncio
    async def test_processor_from_registry_works(self, tmp_path):
        """Test that processors retrieved from registry work correctly."""
        registry = ProcessorRegistry()
        registry.register(TestProcessorA)

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test content")

        processor = registry.get_processor(txt_file)
        assert processor is not None

        result = await processor.process(txt_file, "TestCompany")
        assert result.summary == "Summary A"
        assert len(result.chunks) == 1
        assert result.chunks[0].company_name == "TestCompany"
