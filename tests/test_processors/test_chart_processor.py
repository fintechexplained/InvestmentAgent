"""Tests for chart/image processor."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.processors.chart_processor import ChartProcessor
from src.processors.base_processor import ProcessingResult


@pytest.fixture
def sample_chart_file(tmp_path):
    """Create a sample chart image file (empty, for testing)."""
    chart = tmp_path / "stock_chart.png"
    # Create a minimal valid PNG file
    chart.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return chart


@pytest.fixture
def mock_claude_response():
    """Mock Claude Vision API response."""
    return MagicMock(
        content=[
            MagicMock(
                text="""Chart Analysis:

                Chart Type: Stock price chart with volume indicators
                Time Period: Q4 2024 (October - December)

                Key Price Points:
                - Opening Price: $45.20
                - Closing Price: $62.50
                - High: $65.00
                - Low: $43.80

                Trend Analysis: Bullish trend with strong upward momentum.
                The stock price increased by 38% during the quarter.

                Volume Pattern: Increasing volume during price rises,
                indicating strong buyer interest and healthy price action.
                """
            )
        ]
    )


class TestChartProcessor:
    """Test ChartProcessor class."""

    def test_processor_properties(self):
        """Test processor properties."""
        processor = ChartProcessor()

        assert ".png" in processor.supported_extensions
        assert ".jpg" in processor.supported_extensions
        assert ".jpeg" in processor.supported_extensions
        assert ".pdf" in processor.supported_extensions
        assert processor.modality_type == "image"

    @pytest.mark.asyncio
    async def test_validate_file_valid(self, sample_chart_file):
        """Test validating a valid chart file."""
        processor = ChartProcessor()
        assert await processor.validate_file(sample_chart_file) is True

    @pytest.mark.asyncio
    async def test_validate_file_invalid_extension(self, tmp_path):
        """Test validating file with invalid extension."""
        processor = ChartProcessor()
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("text")

        assert await processor.validate_file(invalid_file) is False

    @pytest.mark.asyncio
    async def test_validate_file_nonexistent(self, tmp_path):
        """Test validating nonexistent file."""
        processor = ChartProcessor()
        missing_file = tmp_path / "missing.png"

        assert await processor.validate_file(missing_file) is False

    @pytest.mark.asyncio
    async def test_process_with_mock_claude(
        self, sample_chart_file, mock_claude_response
    ):
        """Test processing chart file with mocked Claude Vision API."""
        processor = ChartProcessor()

        # Mock the Anthropic client
        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            result = await processor.process(sample_chart_file, "TestCompany")

            # Verify result structure
            assert isinstance(result, ProcessingResult)
            assert len(result.chunks) > 0
            assert result.summary != ""
            assert isinstance(result.extracted_metrics, dict)
            assert isinstance(result.processing_metadata, dict)

            # Verify Claude was called
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_chunks_have_required_fields(
        self, sample_chart_file, mock_claude_response
    ):
        """Test that all chunks have required fields."""
        processor = ChartProcessor()

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            result = await processor.process(sample_chart_file, "TestCompany")

            for chunk in result.chunks:
                assert chunk.content != ""
                assert chunk.chunk_id != ""
                assert chunk.company_name == "TestCompany"
                assert chunk.source_file == str(sample_chart_file)
                assert chunk.modality == "image"
                assert isinstance(chunk.metadata, dict)

    @pytest.mark.asyncio
    async def test_process_extracts_metrics(
        self, sample_chart_file, mock_claude_response
    ):
        """Test that metrics are extracted from chart analysis."""
        processor = ChartProcessor()

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            result = await processor.process(sample_chart_file, "TestCompany")

            # Should have extracted some metrics
            assert len(result.extracted_metrics) > 0

    @pytest.mark.asyncio
    async def test_process_includes_vision_metadata(
        self, sample_chart_file, mock_claude_response
    ):
        """Test that vision processing metadata is included."""
        processor = ChartProcessor()

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            result = await processor.process(sample_chart_file, "TestCompany")

            metadata = result.processing_metadata
            assert "processing_time" in metadata
            assert len(metadata) > 0

    @pytest.mark.asyncio
    async def test_process_handles_api_error(self, sample_chart_file):
        """Test handling of API errors."""
        processor = ChartProcessor()

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("API Error")
            )

            with pytest.raises((Exception, ValueError)):
                await processor.process(sample_chart_file, "TestCompany")

    @pytest.mark.asyncio
    async def test_processor_logging(
        self, sample_chart_file, mock_claude_response, caplog
    ):
        """Test that processor logs appropriately."""
        processor = ChartProcessor()

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            with caplog.at_level("INFO"):
                await processor.process(sample_chart_file, "TestCompany")

            # Check that logging occurred
            assert len(caplog.records) > 0

    @pytest.mark.asyncio
    async def test_process_different_image_formats(
        self, tmp_path, mock_claude_response
    ):
        """Test processing different image formats."""
        processor = ChartProcessor()

        # Test PNG
        png_file = tmp_path / "chart.png"
        png_file.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            result = await processor.process(png_file, "TestCompany")
            assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_process_creates_summary(
        self, sample_chart_file, mock_claude_response
    ):
        """Test that a summary is generated from the chart analysis."""
        processor = ChartProcessor()

        with patch.object(processor, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )

            result = await processor.process(sample_chart_file, "TestCompany")

            assert result.summary != ""
            assert "TestCompany" in result.summary or len(result.summary) > 0
