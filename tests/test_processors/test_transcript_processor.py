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


# ---------------------------------------------------------------------------
# Q&A section parsing
# ---------------------------------------------------------------------------


@pytest.fixture
def qa_transcript(tmp_path):
    """Transcript with a realistic multi-speaker Q&A section.

    Layout
    ------
    - Prepared Remarks   – single named speaker (Timothy D. Cook)
    - Q&A Session        – six speaker turns with named analysts, an Operator,
                           and executive responses.  Includes a noise line
                           ("freestar") *before* the first speaker, and one of
                           Timothy's answers spans multiple continuation lines.
    - Closing Remarks    – single named speaker
    """
    transcript = tmp_path / "qa_transcript.txt"
    transcript.write_text("""\
Prepared Remarks:
Timothy D. Cook: Good morning, everyone. We had a fantastic quarter delivering
strong results across all of our product lines.

Q&A Session:
freestar
Suhasini Chandramouli: Thank you, Tim. We ask that you limit yourself to two questions. Operator, may we have the first question, please?

Operator: Certainly.

Amit Daryanani: Thank you. I have two questions. First, can you walk us through
the memory cost dynamics and how they impact gross margin going forward?

Timothy D. Cook: Sure, Amit. Memory had a minimal impact on Q1 gross margin.
We do expect it to be a bit more of an impact on Q2 gross margin, and that
was comprehended in the outlook that we gave earlier.

Amit Daryanani: Thank you. My follow-up is about China. What is driving
the strength there?

Timothy D. Cook: Greater China was up 38% year on year. It was driven by
iPhone, where we set an all-time revenue record.

Closing Remarks:
Timothy D. Cook: Thank you all for joining us today. We appreciate your continued support.
""")
    return transcript


class TestQASectionParsing:
    """Q&A section detection, speaker extraction, and turn splitting."""

    @pytest.mark.asyncio
    async def test_qa_produces_one_chunk_per_speaker_turn(self, qa_transcript):
        """Each speaker turn in the Q&A becomes its own chunk with section == 'Q&A'."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        qa_chunks = [c for c in result.chunks if c.metadata["section"] == "Q&A"]
        # The fixture has exactly six speaker turns in the Q&A section
        assert len(qa_chunks) == 6

    @pytest.mark.asyncio
    async def test_qa_every_chunk_has_speaker(self, qa_transcript):
        """Every Q&A chunk carries a non-empty 'speaker' key in its metadata."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        qa_chunks = [c for c in result.chunks if c.metadata["section"] == "Q&A"]
        for chunk in qa_chunks:
            assert "speaker" in chunk.metadata, (
                f"chunk missing 'speaker': {chunk.content[:60]!r}"
            )
            assert chunk.metadata["speaker"] != ""

    @pytest.mark.asyncio
    async def test_qa_speaker_order_matches_transcript(self, qa_transcript):
        """Speakers appear in the exact order they occur in the source text."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        qa_chunks = [c for c in result.chunks if c.metadata["section"] == "Q&A"]
        speakers = [c.metadata["speaker"] for c in qa_chunks]

        assert speakers == [
            "Suhasini Chandramouli",
            "Operator",
            "Amit Daryanani",
            "Timothy D. Cook",
            "Amit Daryanani",
            "Timothy D. Cook",
        ]

    @pytest.mark.asyncio
    async def test_qa_chunk_text_belongs_to_correct_speaker(self, qa_transcript):
        """Key phrases appear only in the chunk of the speaker who said them."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        qa_chunks = [c for c in result.chunks if c.metadata["section"] == "Q&A"]

        # Collect all text per speaker
        by_speaker: dict[str, list[str]] = {}
        for chunk in qa_chunks:
            by_speaker.setdefault(chunk.metadata["speaker"], []).append(chunk.content)

        amit_text     = " ".join(by_speaker["Amit Daryanani"])
        tim_text      = " ".join(by_speaker["Timothy D. Cook"])
        operator_text = " ".join(by_speaker["Operator"])

        # Amit asked about memory AND about China
        assert "memory" in amit_text.lower()
        assert "china" in amit_text.lower()

        # Tim answered about memory AND about China
        assert "memory" in tim_text.lower()
        assert "china" in tim_text.lower()

        # Operator said only "Certainly"
        assert "certainly" in operator_text.lower()

    @pytest.mark.asyncio
    async def test_qa_multi_line_turn_stays_as_single_chunk(self, qa_transcript):
        """A speaker's answer that wraps across continuation lines is one chunk."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        qa_chunks = [c for c in result.chunks if c.metadata["section"] == "Q&A"]
        tim_chunks = [
            c for c in qa_chunks if c.metadata["speaker"] == "Timothy D. Cook"
        ]

        # The first Timothy chunk is the memory answer; both sentences must be present
        assert "minimal impact" in tim_chunks[0].content
        assert "outlook"        in tim_chunks[0].content

    @pytest.mark.asyncio
    async def test_qa_noise_before_first_speaker_is_skipped(self, qa_transcript):
        """Noise text that appears before the first Q&A speaker is dropped entirely."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        qa_chunks   = [c for c in result.chunks if c.metadata["section"] == "Q&A"]
        all_qa_text = " ".join(c.content for c in qa_chunks)

        assert "freestar" not in all_qa_text

    @pytest.mark.asyncio
    async def test_non_qa_sections_also_populate_speaker(self, qa_transcript):
        """Prepared Remarks and Closing Remarks also carry a speaker in metadata."""
        processor = TranscriptProcessor()
        result = await processor.process(qa_transcript, "Apple")

        prepared = [c for c in result.chunks if c.metadata["section"] == "Prepared Remarks"]
        closing  = [c for c in result.chunks if c.metadata["section"] == "Closing Remarks"]

        assert len(prepared) == 1
        assert prepared[0].metadata.get("speaker") == "Timothy D. Cook"

        assert len(closing) == 1
        assert closing[0].metadata.get("speaker") == "Timothy D. Cook"

    @pytest.mark.asyncio
    async def test_qa_role_based_speakers(self, tmp_path):
        """Q&A parsing also works when speakers are bare roles (CEO:, CFO:, Operator:)."""
        transcript = tmp_path / "role_qa.txt"
        transcript.write_text("""\
Prepared Remarks:
CEO: Good morning, everyone. Strong quarter across the board.

Q&A Session:
Operator: Thank you. We will now take questions.

CEO: Thank you, Operator. Happy to take questions.

CFO: And I am here to handle the financial details.
""")
        processor = TranscriptProcessor()
        result = await processor.process(transcript, "RoleCo")

        qa_chunks = [c for c in result.chunks if c.metadata["section"] == "Q&A"]
        speakers  = [c.metadata["speaker"] for c in qa_chunks]

        assert speakers == ["Operator", "CEO", "CFO"]
        for chunk in qa_chunks:
            assert chunk.content.strip() != ""
