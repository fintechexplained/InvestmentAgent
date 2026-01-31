"""Processor for stock price charts and financial visualizations."""

import logging
import time
import base64
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
from anthropic import AsyncAnthropic
from .base_processor import BaseModalityProcessor, ProcessedChunk, ProcessingResult

logger = logging.getLogger(__name__)


class ChartProcessor(BaseModalityProcessor):
    """Processes stock price charts and financial visualizations.

    This processor:
    1. Loads and encodes images to base64
    2. Uses Claude Vision to extract chart data
    3. Converts visual insights to structured text
    4. Extracts price points and trends
    """

    def __init__(self) -> None:
        """Initialize the chart processor."""
        super().__init__()
        self.claude_client = AsyncAnthropic()

    @property
    def supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf"]

    @property
    def modality_type(self) -> str:
        """Get modality type."""
        return "image"

    async def validate_file(self, file_path: Path) -> bool:
        """Validate that file can be processed.

        Args:
            file_path: Path to the file to validate

        Returns:
            True if file is valid and can be processed
        """
        try:
            is_valid = file_path.exists() and file_path.suffix.lower() in [
                ext.lower() for ext in self.supported_extensions
            ]
            if is_valid:
                logger.debug(f"Chart file {file_path.name} is valid for processing")
            else:
                logger.warning(f"Chart file {file_path.name} failed validation")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating chart file {file_path}: {e}")
            return False

    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        """Process chart/image file.

        Args:
            file_path: Path to the chart file
            company_name: Name of the company

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()
        logger.info(f"Processing chart file {file_path.name} for {company_name}")

        try:
            # Load and encode image
            image_data = self._load_and_encode_image(file_path)

            # Analyze with Claude Vision
            analysis = await self._analyze_chart(image_data, file_path, company_name)

            if not analysis.strip():
                logger.warning(f"Empty analysis for {file_path.name}")
                return ProcessingResult(
                    chunks=[],
                    summary="Empty chart analysis",
                    processing_metadata={
                        "processing_time": time.time() - start_time,
                        "error": "Empty analysis",
                    },
                )

            logger.info(
                f"Analyzed {file_path.name}: {len(analysis)} characters"
            )

            # Extract structured data from analysis
            metrics = self._extract_metrics_from_analysis(analysis)
            summary = self._generate_summary(analysis, company_name)
            chunks = self._create_chunks(analysis, file_path, company_name)

            processing_time = time.time() - start_time
            logger.info(
                f"Processed chart {file_path.name}: {len(chunks)} chunks, "
                f"{len(metrics)} metrics in {processing_time:.2f}s"
            )

            return ProcessingResult(
                chunks=chunks,
                summary=summary,
                extracted_metrics=metrics,
                processing_metadata={
                    "processing_time": processing_time,
                    "analysis_length": len(analysis),
                    "image_file": str(file_path),
                    "vision_model": "claude",
                },
            )

        except Exception as e:
            logger.error(f"Error processing chart file {file_path.name}: {e}")
            raise ValueError(f"Failed to process chart: {e}")

    def _load_and_encode_image(self, file_path: Path) -> str:
        """Load image and encode to base64.

        Args:
            file_path: Path to the image file

        Returns:
            Base64 encoded image data
        """
        try:
            with open(file_path, "rb") as image_file:
                image_data = image_file.read()

            encoded = base64.standard_b64encode(image_data).decode("utf-8")
            logger.debug(f"Encoded image {file_path.name}: {len(encoded)} bytes")
            return encoded

        except Exception as e:
            logger.error(f"Error encoding image {file_path.name}: {e}")
            raise

    async def _analyze_chart(
        self, image_data: str, file_path: Path, company_name: str
    ) -> str:
        """Analyze chart using Claude Vision API.

        Args:
            image_data: Base64 encoded image
            file_path: Original file path
            company_name: Company name

        Returns:
            Analysis text from Claude
        """
        try:
            logger.debug(f"Analyzing chart {file_path.name} with Claude Vision")

            # Determine media type from file extension
            ext = file_path.suffix.lower()
            media_type_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
            }
            media_type = media_type_map.get(ext, "image/png")

            prompt = f"""Analyze this financial chart/graph for {company_name}.
Please provide a detailed analysis including:

1. Chart Type: What kind of chart is this? (e.g., stock price, volume, candlestick, bar chart)
2. Time Period: What time period is shown?
3. Key Price Points: Identify important prices (open, close, high, low, current)
4. Trend Analysis: What is the overall trend? (bullish, bearish, sideways, consolidation)
5. Volume Patterns: If volume is shown, describe the pattern
6. Key Observations: Any notable patterns, support/resistance levels, or significant events
7. Numerical Data: Extract any specific numbers, percentages, or metrics shown

Be specific and quantitative where possible."""

            response = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            # Extract text from response
            analysis = response.content[0].text

            logger.debug(f"Successfully analyzed {file_path.name}")
            return analysis

        except Exception as e:
            logger.error(f"Claude Vision API error for {file_path.name}: {e}")
            raise

    def _extract_metrics_from_analysis(self, analysis: str) -> Dict[str, Any]:
        """Extract structured metrics from chart analysis.

        Args:
            analysis: Analysis text from Claude

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        # Extract price mentions
        price_patterns = [
            r"(?:price|open|close|high|low).*?\$?(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\$(\d+(?:,\d+)*(?:\.\d+)?)",
        ]

        prices = []
        for pattern in price_patterns:
            prices.extend(re.findall(pattern, analysis, re.IGNORECASE))

        if prices:
            metrics["price_mentions"] = list(set(prices))[:10]  # Limit to 10

        # Extract percentages
        percentage_pattern = r"(\d+(?:\.\d+)?)\s*%"
        percentages = re.findall(percentage_pattern, analysis)
        if percentages:
            metrics["percentage_changes"] = list(set(percentages))[:10]

        # Extract trend mentions
        if re.search(r"\bbullish\b", analysis, re.IGNORECASE):
            metrics["trend"] = "bullish"
        elif re.search(r"\bbearish\b", analysis, re.IGNORECASE):
            metrics["trend"] = "bearish"
        elif re.search(r"\bsideways\b", analysis, re.IGNORECASE):
            metrics["trend"] = "sideways"

        # Extract time period
        time_periods = re.findall(
            r"(Q[1-4]\s+\d{4}|"
            r"\d{4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}|"
            r"\d{1,2}/\d{1,2}/\d{2,4})",
            analysis,
            re.IGNORECASE,
        )
        if time_periods:
            metrics["time_period"] = time_periods[0]

        logger.debug(f"Extracted {len(metrics)} metric types from analysis")
        return metrics

    def _generate_summary(self, analysis: str, company_name: str) -> str:
        """Generate a summary of the chart analysis.

        Args:
            analysis: Analysis text
            company_name: Company name

        Returns:
            Summary string
        """
        # Extract first few sentences as summary
        sentences = analysis.split(". ")
        summary_parts = [f"Chart analysis for {company_name}."]

        # Add first 2-3 substantive sentences
        for sentence in sentences[:3]:
            if len(sentence.strip()) > 20:
                summary_parts.append(sentence.strip() + ".")

        summary = " ".join(summary_parts)
        if len(summary) > 300:
            summary = summary[:297] + "..."

        logger.debug(f"Generated summary of {len(summary)} characters")
        return summary

    def _create_chunks(
        self, analysis: str, file_path: Path, company_name: str
    ) -> List[ProcessedChunk]:
        """Create chunks from chart analysis.

        Args:
            analysis: Analysis text
            file_path: Source file path
            company_name: Company name

        Returns:
            List of ProcessedChunk objects
        """
        chunks = []

        # Split analysis into logical sections
        sections = self._split_analysis_into_sections(analysis)

        for i, (section_title, section_content) in enumerate(sections):
            chunk_id = self._generate_chunk_id(
                f"{company_name}_{file_path.name}_{section_title}_{i}"
            )

            chunks.append(
                ProcessedChunk(
                    content=section_content,
                    metadata={
                        "section": section_title,
                        "section_index": i,
                        "image_file": str(file_path),
                        "char_count": len(section_content),
                    },
                    chunk_id=chunk_id,
                    company_name=company_name,
                    source_file=str(file_path),
                    modality=self.modality_type,
                )
            )

        logger.debug(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks

    def _split_analysis_into_sections(
        self, analysis: str
    ) -> List[tuple[str, str]]:
        """Split analysis into sections.

        Args:
            analysis: Analysis text

        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []

        # Try to identify numbered sections or headers
        lines = analysis.split("\n")
        current_section = "Overview"
        current_content = []

        for line in lines:
            # Check if line is a section header (numbered or capitalized)
            if re.match(r"^\d+\.\s+[A-Z]", line) or line.isupper() and len(line) < 50:
                # Save previous section
                if current_content:
                    sections.append((current_section, "\n".join(current_content)))

                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append((current_section, "\n".join(current_content)))

        # If no sections identified, treat entire analysis as one chunk
        if len(sections) == 0:
            sections = [("Full Analysis", analysis)]

        return sections

    def _generate_chunk_id(self, seed: str) -> str:
        """Generate a unique chunk ID.

        Args:
            seed: Seed string for ID generation

        Returns:
            Unique chunk ID
        """
        return hashlib.md5(seed.encode()).hexdigest()[:16]
