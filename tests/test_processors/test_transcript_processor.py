"""Tests for transcript processor."""

import pytest
from pathlib import Path
from src.processors.transcript_processor import TranscriptProcessor
from src.processors.base_processor import ProcessingResult


@pytest.fixture
def sample_transcript(tmp_path):
    """Create a sample transcript file for testing."""
    transcript = tmp_path / "earnings_call.txt"
    transcript.write_text("""
    Q4 2024 Earnings Call Transcript

    CEO: We delivered strong results with revenue of $500M,
    up 25% year-over-year. Our operating margin expanded to 22%.

    Q&A:
    Analyst: What drove the margin expansion?
    CFO: Improved operational efficiency and scale benefits.
    Our gross margin reached 65% this quarter.
    """)
    return transcript


@pytest.fixture
def long_transcript(tmp_path):
    """Create a longer transcript for chunking tests."""
    transcript = tmp_path / "long_earnings.txt"
    content = "SECTION 1: INTRODUCTION\n" + ("Test content. " * 200) + "\n\n"
    content += "SECTION 2: FINANCIAL RESULTS\n" + ("More content. " * 200) + "\n\n"
    content += "SECTION 3: Q&A\n" + ("Questions and answers. " * 200)
    transcript.write_text(content)
    return transcript


@pytest.fixture
def real_transcript():
    """Get path to real sample transcript fixture."""
    return Path(__file__).parent.parent / "fixtures" / "sample_transcript.txt"


class TestTranscriptProcessor:
    """Test TranscriptProcessor class."""

    def test_processor_properties(self):
        """Test processor properties."""
        processor = TranscriptProcessor()

        assert ".txt" in processor.supported_extensions
        assert ".md" in processor.supported_extensions
        assert ".transcript" in processor.supported_extensions
        assert processor.modality_type == "text"

    @pytest.mark.asyncio
    async def test_validate_file_valid(self, sample_transcript):
        """Test validating a valid file."""
        processor = TranscriptProcessor()
        assert await processor.validate_file(sample_transcript) is True

    @pytest.mark.asyncio
    async def test_validate_file_invalid_extension(self, tmp_path):
        """Test validating file with invalid extension."""
        processor = TranscriptProcessor()
        invalid_file = tmp_path / "test.mp3"
        invalid_file.write_bytes(b"audio")

        assert await processor.validate_file(invalid_file) is False

    @pytest.mark.asyncio
    async def test_validate_file_nonexistent(self, tmp_path):
        """Test validating nonexistent file."""
        processor = TranscriptProcessor()
        missing_file = tmp_path / "missing.txt"

        assert await processor.validate_file(missing_file) is False

    @pytest.mark.asyncio
    async def test_process_basic(self, sample_transcript):
        """Test basic transcript processing."""
        processor = TranscriptProcessor()
        result = await processor.process(sample_transcript, "TestCompany")

        # Check result structure
        assert isinstance(result, ProcessingResult)
        assert len(result.chunks) > 0
        assert result.summary != ""
        assert isinstance(result.extracted_metrics, dict)
        assert isinstance(result.processing_metadata, dict)

    @pytest.mark.asyncio
    async def test_process_chunks_have_required_fields(self, sample_transcript):
        """Test that all chunks have required fields."""
        processor = TranscriptProcessor()
        result = await processor.process(sample_transcript, "TestCompany")

        for chunk in result.chunks:
            assert chunk.content != ""
            assert chunk.chunk_id != ""
            assert chunk.company_name == "TestCompany"
            assert chunk.source_file == str(sample_transcript)
            assert chunk.modality == "text"
            assert isinstance(chunk.metadata, dict)

    @pytest.mark.asyncio
    async def test_process_extracts_metrics(self, sample_transcript):
        """Test that key metrics are extracted from transcript."""
        processor = TranscriptProcessor()
        result = await processor.process(sample_transcript, "TestCompany")

        # Check that some metrics were extracted
        metrics = result.extracted_metrics
        assert "revenue" in str(metrics).lower() or len(metrics) > 0

    @pytest.mark.asyncio
    async def test_process_creates_multiple_chunks(self, long_transcript):
        """Test that long transcripts are split into multiple chunks."""
        processor = TranscriptProcessor()
        result = await processor.process(long_transcript, "TestCompany")

        # Long transcript should be split into multiple chunks
        assert len(result.chunks) > 1

    @pytest.mark.asyncio
    async def test_process_chunk_ids_unique(self, long_transcript):
        """Test that chunk IDs are unique."""
        processor = TranscriptProcessor()
        result = await processor.process(long_transcript, "TestCompany")

        chunk_ids = [chunk.chunk_id for chunk in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    @pytest.mark.asyncio
    async def test_process_with_markdown_file(self, tmp_path):
        """Test processing markdown file."""
        processor = TranscriptProcessor()
        md_file = tmp_path / "transcript.md"
        md_file.write_text("""
        # Q4 2024 Earnings Call

        ## Revenue
        Revenue was $500M, up 25% YoY.

        ## Margin
        Operating margin: 22%
        """)

        result = await processor.process(md_file, "TestCompany")
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_process_real_transcript(self, real_transcript):
        """Test processing the real sample transcript."""
        if not real_transcript.exists():
            pytest.skip("Sample transcript fixture not found")

        processor = TranscriptProcessor()
        result = await processor.process(real_transcript, "RandomCompanyA")

        # Verify structure
        assert len(result.chunks) > 0
        assert result.summary != ""
        assert len(result.extracted_metrics) > 0

        # Verify content
        all_content = " ".join([chunk.content for chunk in result.chunks])
        assert "revenue" in all_content.lower()
        assert "500" in all_content or "$500" in all_content

        # Verify company name is set correctly
        for chunk in result.chunks:
            assert chunk.company_name == "RandomCompanyA"

    @pytest.mark.asyncio
    async def test_process_includes_metadata(self, sample_transcript):
        """Test that processing metadata is included."""
        processor = TranscriptProcessor()
        result = await processor.process(sample_transcript, "TestCompany")

        metadata = result.processing_metadata
        assert "processing_time" in metadata or len(metadata) > 0

    @pytest.mark.asyncio
    async def test_process_empty_file(self, tmp_path):
        """Test processing an empty file."""
        processor = TranscriptProcessor()
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = await processor.process(empty_file, "TestCompany")
        # Should still return a result, even if empty
        assert isinstance(result, ProcessingResult)

    @pytest.mark.asyncio
    async def test_processor_logging(self, sample_transcript, caplog):
        """Test that processor logs appropriately."""
        processor = TranscriptProcessor()

        with caplog.at_level("INFO"):
            await processor.process(sample_transcript, "TestCompany")

        # Check that some logging occurred
        assert len(caplog.records) > 0
