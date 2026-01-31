"""Tests for audio processor."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.processors.audio_processor import AudioProcessor
from src.processors.base_processor import ProcessingResult


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file (empty, for testing)."""
    audio = tmp_path / "earnings_call.mp3"
    audio.write_bytes(b"fake audio data")
    return audio


@pytest.fixture
def mock_whisper_response():
    """Mock Whisper API response."""
    return MagicMock(
        text="""Q4 2024 Earnings Call.
        CEO: We achieved revenue of $500M, up 25% year-over-year.
        Our operating margin expanded to 22%.
        Analyst: What drove the margin expansion?
        CFO: Improved operational efficiency and scale benefits."""
    )


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_processor_properties(self):
        """Test processor properties."""
        processor = AudioProcessor()

        assert ".mp3" in processor.supported_extensions
        assert ".wav" in processor.supported_extensions
        assert ".m4a" in processor.supported_extensions
        assert processor.modality_type == "audio"

    @pytest.mark.asyncio
    async def test_validate_file_valid(self, sample_audio_file):
        """Test validating a valid audio file."""
        processor = AudioProcessor()
        assert await processor.validate_file(sample_audio_file) is True

    @pytest.mark.asyncio
    async def test_validate_file_invalid_extension(self, tmp_path):
        """Test validating file with invalid extension."""
        processor = AudioProcessor()
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("text")

        assert await processor.validate_file(invalid_file) is False

    @pytest.mark.asyncio
    async def test_validate_file_nonexistent(self, tmp_path):
        """Test validating nonexistent file."""
        processor = AudioProcessor()
        missing_file = tmp_path / "missing.mp3"

        assert await processor.validate_file(missing_file) is False

    @pytest.mark.asyncio
    async def test_process_with_mock_whisper(
        self, sample_audio_file, mock_whisper_response
    ):
        """Test processing audio file with mocked Whisper API."""
        processor = AudioProcessor()

        # Mock the OpenAI client
        with patch.object(processor, "whisper_client") as mock_client:
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=mock_whisper_response
            )

            result = await processor.process(sample_audio_file, "TestCompany")

            # Verify result structure
            assert isinstance(result, ProcessingResult)
            assert len(result.chunks) > 0
            assert result.summary != ""
            assert isinstance(result.extracted_metrics, dict)
            assert isinstance(result.processing_metadata, dict)

            # Verify Whisper was called
            mock_client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_chunks_have_required_fields(
        self, sample_audio_file, mock_whisper_response
    ):
        """Test that all chunks have required fields."""
        processor = AudioProcessor()

        with patch.object(processor, "whisper_client") as mock_client:
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=mock_whisper_response
            )

            result = await processor.process(sample_audio_file, "TestCompany")

            for chunk in result.chunks:
                assert chunk.content != ""
                assert chunk.chunk_id != ""
                assert chunk.company_name == "TestCompany"
                assert chunk.source_file == str(sample_audio_file)
                assert chunk.modality == "audio"
                assert isinstance(chunk.metadata, dict)

    @pytest.mark.asyncio
    async def test_process_includes_transcription_metadata(
        self, sample_audio_file, mock_whisper_response
    ):
        """Test that transcription metadata is included."""
        processor = AudioProcessor()

        with patch.object(processor, "whisper_client") as mock_client:
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=mock_whisper_response
            )

            result = await processor.process(sample_audio_file, "TestCompany")

            metadata = result.processing_metadata
            assert "processing_time" in metadata
            assert "transcription_length" in metadata or len(metadata) > 0

    @pytest.mark.asyncio
    async def test_process_handles_api_error(self, sample_audio_file):
        """Test handling of API errors."""
        processor = AudioProcessor()

        with patch.object(processor, "whisper_client") as mock_client:
            mock_client.audio.transcriptions.create = AsyncMock(
                side_effect=Exception("API Error")
            )

            with pytest.raises((Exception, ValueError)):
                await processor.process(sample_audio_file, "TestCompany")

    @pytest.mark.asyncio
    async def test_processor_logging(self, sample_audio_file, mock_whisper_response, caplog):
        """Test that processor logs appropriately."""
        processor = AudioProcessor()

        with patch.object(processor, "whisper_client") as mock_client:
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=mock_whisper_response
            )

            with caplog.at_level("INFO"):
                await processor.process(sample_audio_file, "TestCompany")

            # Check that logging occurred
            assert len(caplog.records) > 0

    @pytest.mark.asyncio
    async def test_process_creates_summary(self, sample_audio_file, mock_whisper_response):
        """Test that a summary is generated from the transcription."""
        processor = AudioProcessor()

        with patch.object(processor, "whisper_client") as mock_client:
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=mock_whisper_response
            )

            result = await processor.process(sample_audio_file, "TestCompany")

            assert result.summary != ""
            assert len(result.summary) > 0
