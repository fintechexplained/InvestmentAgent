"""Tests for IngestionPipeline."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.ingestion.pipeline import IngestionPipeline
from src.processors.registry import ProcessorRegistry
from src.processors.base_processor import (
    BaseModalityProcessor,
    ProcessedChunk,
    ProcessingResult,
)
from src.storage.vector_store import VectorStoreManager


@pytest.fixture
def mock_registry():
    """Create a mock processor registry."""
    return MagicMock(spec=ProcessorRegistry)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock_store = MagicMock(spec=VectorStoreManager)
    mock_store.add_chunks = AsyncMock()
    return mock_store


@pytest.fixture
def mock_processor():
    """Create a mock processor."""
    processor = MagicMock(spec=BaseModalityProcessor)
    processor.validate_file = AsyncMock(return_value=True)
    processor.process = AsyncMock()
    return processor


@pytest.fixture
def sample_processing_result():
    """Create sample processing result."""
    chunks = [
        ProcessedChunk(
            content="Test chunk 1",
            metadata={"section": "intro"},
            chunk_id="chunk1",
            company_name="TestCompany",
            source_file="test.txt",
            modality="text",
        ),
        ProcessedChunk(
            content="Test chunk 2",
            metadata={"section": "body"},
            chunk_id="chunk2",
            company_name="TestCompany",
            source_file="test.txt",
            modality="text",
        ),
    ]
    return ProcessingResult(
        chunks=chunks,
        summary="Test summary",
        extracted_metrics={"revenue": "500M"},
        processing_metadata={"processing_time": 1.5},
    )


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a sample data directory structure."""
    # Create company directories
    company_a = tmp_path / "CompanyA"
    company_a.mkdir()

    company_b = tmp_path / "CompanyB"
    company_b.mkdir()

    # Add files to CompanyA
    (company_a / "transcript.txt").write_text("CompanyA transcript data")
    (company_a / "audio.mp3").write_bytes(b"fake audio")
    (company_a / "chart.png").write_bytes(b"fake image")

    # Add files to CompanyB
    (company_b / "transcript.txt").write_text("CompanyB transcript data")

    return tmp_path


class TestIngestionPipeline:
    """Test IngestionPipeline class."""

    def test_initialization(self, mock_registry, mock_vector_store):
        """Test pipeline initialization."""
        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        assert pipeline.registry == mock_registry
        assert pipeline.vector_store == mock_vector_store

    @pytest.mark.asyncio
    async def test_ingest_company_single_file_success(
        self,
        mock_registry,
        mock_vector_store,
        mock_processor,
        sample_processing_result,
        tmp_path,
    ):
        """Test ingesting a single file successfully."""
        # Setup
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.process.return_value = sample_processing_result

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company("TestCompany", [test_file])

        # Verify
        assert stats["company_name"] == "TestCompany"
        assert stats["total_files"] == 1
        assert stats["processed_files"] == 1
        assert stats["failed_files"] == 0
        assert stats["total_chunks"] == 2
        assert len(stats["errors"]) == 0

        # Verify processor was called
        mock_registry.get_processor.assert_called_once_with(test_file)
        mock_processor.validate_file.assert_called_once_with(test_file)
        mock_processor.process.assert_called_once_with(test_file, "TestCompany")

        # Verify chunks were added to vector store
        mock_vector_store.add_chunks.assert_called_once()
        chunks_added = mock_vector_store.add_chunks.call_args[0][0]
        assert len(chunks_added) == 2

    @pytest.mark.asyncio
    async def test_ingest_company_multiple_files(
        self,
        mock_registry,
        mock_vector_store,
        mock_processor,
        sample_processing_result,
        tmp_path,
    ):
        """Test ingesting multiple files."""
        # Setup
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"

        file1.write_text("content 1")
        file2.write_text("content 2")
        file3.write_text("content 3")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.process.return_value = sample_processing_result

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company(
            "TestCompany", [file1, file2, file3]
        )

        # Verify
        assert stats["total_files"] == 3
        assert stats["processed_files"] == 3
        assert stats["failed_files"] == 0
        assert stats["total_chunks"] == 6  # 2 chunks per file × 3 files

        # Verify processor was called for each file
        assert mock_processor.validate_file.call_count == 3
        assert mock_processor.process.call_count == 3
        assert mock_vector_store.add_chunks.call_count == 3

    @pytest.mark.asyncio
    async def test_ingest_company_no_processor_found(
        self, mock_registry, mock_vector_store, tmp_path
    ):
        """Test ingestion when no processor is found for file."""
        # Setup
        test_file = tmp_path / "unknown.xyz"
        test_file.write_text("unknown file type")

        mock_registry.get_processor.return_value = None

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company("TestCompany", [test_file])

        # Verify
        assert stats["total_files"] == 1
        assert stats["processed_files"] == 0
        assert stats["failed_files"] == 1
        assert stats["total_chunks"] == 0
        assert len(stats["errors"]) == 1
        assert "No processor" in stats["errors"][0]

        # Verify vector store was not called
        mock_vector_store.add_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_company_file_validation_fails(
        self, mock_registry, mock_vector_store, mock_processor, tmp_path
    ):
        """Test ingestion when file validation fails."""
        # Setup
        test_file = tmp_path / "invalid.txt"
        test_file.write_text("content")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.validate_file.return_value = False

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company("TestCompany", [test_file])

        # Verify
        assert stats["processed_files"] == 0
        assert stats["failed_files"] == 1
        assert len(stats["errors"]) == 1
        assert "Validation failed" in stats["errors"][0]

        # Verify process was not called
        mock_processor.process.assert_not_called()
        mock_vector_store.add_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_company_processing_exception(
        self, mock_registry, mock_vector_store, mock_processor, tmp_path
    ):
        """Test ingestion when processing raises exception."""
        # Setup
        test_file = tmp_path / "error.txt"
        test_file.write_text("content")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.process.side_effect = Exception("Processing failed")

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company("TestCompany", [test_file])

        # Verify
        assert stats["processed_files"] == 0
        assert stats["failed_files"] == 1
        assert len(stats["errors"]) == 1
        assert "Processing failed" in stats["errors"][0]
        assert "error.txt" in stats["errors"][0]

        # Verify vector store was not called
        mock_vector_store.add_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_company_mixed_success_failure(
        self,
        mock_registry,
        mock_vector_store,
        mock_processor,
        sample_processing_result,
        tmp_path,
    ):
        """Test ingestion with some files succeeding and some failing."""
        # Setup
        file1 = tmp_path / "success.txt"
        file2 = tmp_path / "fail.txt"

        file1.write_text("content 1")
        file2.write_text("content 2")

        mock_registry.get_processor.return_value = mock_processor

        # First file succeeds, second fails
        mock_processor.process.side_effect = [
            sample_processing_result,
            Exception("Failed"),
        ]

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company("TestCompany", [file1, file2])

        # Verify
        assert stats["total_files"] == 2
        assert stats["processed_files"] == 1
        assert stats["failed_files"] == 1
        assert stats["total_chunks"] == 2
        assert len(stats["errors"]) == 1

        # Verify vector store was called once (for successful file)
        assert mock_vector_store.add_chunks.call_count == 1

    @pytest.mark.asyncio
    async def test_ingest_dataset_success(
        self,
        mock_registry,
        mock_vector_store,
        mock_processor,
        sample_processing_result,
        sample_data_dir,
    ):
        """Test ingesting entire dataset."""
        # Setup
        mock_registry.get_processor.return_value = mock_processor
        mock_processor.process.return_value = sample_processing_result

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_dataset(sample_data_dir)

        # Verify overall stats
        assert stats["data_dir"] == str(sample_data_dir)
        assert stats["total_companies"] == 2
        assert stats["total_files"] == 4  # All successful files
        assert stats["total_chunks"] == 8  # 2 chunks × 4 files

        # Verify company-specific stats
        assert "CompanyA" in stats["companies"]
        assert "CompanyB" in stats["companies"]

        company_a_stats = stats["companies"]["CompanyA"]
        assert company_a_stats["processed_files"] == 3

        company_b_stats = stats["companies"]["CompanyB"]
        assert company_b_stats["processed_files"] == 1

    @pytest.mark.asyncio
    async def test_ingest_dataset_nonexistent_directory(
        self, mock_registry, mock_vector_store, tmp_path
    ):
        """Test ingesting from non-existent directory."""
        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(ValueError) as exc_info:
            await pipeline.ingest_dataset(nonexistent)

        assert "does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ingest_dataset_empty_directory(
        self, mock_registry, mock_vector_store, tmp_path
    ):
        """Test ingesting from empty directory."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_dataset(empty_dir)

        # Verify
        assert stats["total_companies"] == 0
        assert stats["total_files"] == 0
        assert stats["total_chunks"] == 0
        assert len(stats["companies"]) == 0

    @pytest.mark.asyncio
    async def test_ingest_dataset_company_with_no_files(
        self, mock_registry, mock_vector_store, tmp_path
    ):
        """Test ingesting dataset where a company has no files."""
        # Setup
        company_dir = tmp_path / "EmptyCompany"
        company_dir.mkdir()
        # No files in the directory

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_dataset(tmp_path)

        # Verify - company should be skipped
        assert stats["total_companies"] == 0
        assert "EmptyCompany" not in stats["companies"]

    @pytest.mark.asyncio
    async def test_ingest_dataset_skips_files_in_root(
        self,
        mock_registry,
        mock_vector_store,
        mock_processor,
        sample_processing_result,
        tmp_path,
    ):
        """Test that files in root directory are skipped."""
        # Setup
        company_dir = tmp_path / "Company1"
        company_dir.mkdir()
        (company_dir / "file1.txt").write_text("content")

        # Add file to root (should be ignored)
        (tmp_path / "root_file.txt").write_text("root content")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.process.return_value = sample_processing_result

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_dataset(tmp_path)

        # Verify only company directory was processed
        assert stats["total_companies"] == 1
        assert "Company1" in stats["companies"]

        # Root file should not be processed
        assert mock_processor.process.call_count == 1

    @pytest.mark.asyncio
    async def test_ingest_company_empty_file_list(
        self, mock_registry, mock_vector_store
    ):
        """Test ingesting company with empty file list."""
        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        stats = await pipeline.ingest_company("TestCompany", [])

        assert stats["total_files"] == 0
        assert stats["processed_files"] == 0
        assert stats["failed_files"] == 0
        assert stats["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_ingest_company_tracks_all_errors(
        self, mock_registry, mock_vector_store, mock_processor, tmp_path
    ):
        """Test that all errors are tracked in stats."""
        # Setup multiple files with different failure modes
        file1 = tmp_path / "no_processor.xyz"
        file2 = tmp_path / "validation_fail.txt"
        file3 = tmp_path / "process_error.txt"

        file1.write_text("content")
        file2.write_text("content")
        file3.write_text("content")

        # Mock different failure scenarios
        def get_processor_side_effect(file_path):
            if "no_processor" in str(file_path):
                return None
            return mock_processor

        mock_registry.get_processor.side_effect = get_processor_side_effect

        async def validate_side_effect(file_path):
            if "validation_fail" in str(file_path):
                return False
            return True

        mock_processor.validate_file.side_effect = validate_side_effect

        async def process_side_effect(file_path, company_name):
            if "process_error" in str(file_path):
                raise Exception("Process failed")
            return sample_processing_result

        mock_processor.process.side_effect = process_side_effect

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        # Execute
        stats = await pipeline.ingest_company(
            "TestCompany", [file1, file2, file3]
        )

        # Verify all failures tracked
        assert stats["failed_files"] == 3
        assert len(stats["errors"]) == 3
        assert any("No processor" in err for err in stats["errors"])
        assert any("Validation failed" in err for err in stats["errors"])
        assert any("Process failed" in err for err in stats["errors"])

    @pytest.mark.asyncio
    async def test_pipeline_handles_processor_validation_exception(
        self, mock_registry, mock_vector_store, mock_processor, tmp_path
    ):
        """Test pipeline handles exceptions during validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.validate_file.side_effect = Exception("Validation exception")

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        stats = await pipeline.ingest_company("TestCompany", [test_file])

        # Should handle gracefully
        assert stats["failed_files"] == 1
        assert len(stats["errors"]) == 1
        assert "Validation exception" in stats["errors"][0]

    @pytest.mark.asyncio
    async def test_ingest_dataset_accumulates_statistics(
        self,
        mock_registry,
        mock_vector_store,
        mock_processor,
        sample_processing_result,
        tmp_path,
    ):
        """Test that dataset ingestion correctly accumulates statistics."""
        # Setup multiple companies with different file counts
        company1 = tmp_path / "Company1"
        company1.mkdir()
        (company1 / "file1.txt").write_text("content")
        (company1 / "file2.txt").write_text("content")

        company2 = tmp_path / "Company2"
        company2.mkdir()
        (company2 / "file1.txt").write_text("content")

        mock_registry.get_processor.return_value = mock_processor
        mock_processor.process.return_value = sample_processing_result

        pipeline = IngestionPipeline(mock_registry, mock_vector_store)

        stats = await pipeline.ingest_dataset(tmp_path)

        # Verify accumulation
        assert stats["total_companies"] == 2
        assert stats["total_files"] == 3  # 2 + 1
        assert stats["total_chunks"] == 6  # 2 chunks per file × 3 files

        # Verify individual company stats are preserved
        assert stats["companies"]["Company1"]["processed_files"] == 2
        assert stats["companies"]["Company2"]["processed_files"] == 1
