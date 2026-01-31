"""Processor for text-based earnings call transcripts."""

import logging
import hashlib
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from .base_processor import BaseModalityProcessor, ProcessedChunk, ProcessingResult

logger = logging.getLogger(__name__)


class TranscriptProcessor(BaseModalityProcessor):
    """Processes text-based earnings call transcripts.

    This processor:
    1. Reads and parses text files
    2. Chunks content into semantic sections
    3. Extracts key financial metrics
    4. Generates summaries
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the transcript processor.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".txt", ".md", ".transcript"]

    @property
    def modality_type(self) -> str:
        """Get modality type."""
        return "text"

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
                logger.debug(f"File {file_path.name} is valid for processing")
            else:
                logger.warning(f"File {file_path.name} failed validation")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        """Process transcript file.

        Args:
            file_path: Path to the transcript file
            company_name: Name of the company

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()
        logger.info(f"Processing transcript {file_path.name} for {company_name}")

        try:
            # Read file content
            content = self._read_file(file_path)

            if not content.strip():
                logger.warning(f"Empty file: {file_path.name}")
                return ProcessingResult(
                    chunks=[],
                    summary="Empty transcript",
                    processing_metadata={
                        "processing_time": time.time() - start_time,
                        "file_size": 0,
                        "error": "Empty file"
                    }
                )

            # Extract metrics from content
            metrics = self._extract_metrics(content)

            # Generate summary
            summary = self._generate_summary(content, company_name)

            # Chunk the content
            chunks = self._chunk_content(content, file_path, company_name)

            processing_time = time.time() - start_time
            logger.info(
                f"Processed {file_path.name}: {len(chunks)} chunks, "
                f"{len(metrics)} metrics in {processing_time:.2f}s"
            )

            return ProcessingResult(
                chunks=chunks,
                summary=summary,
                extracted_metrics=metrics,
                processing_metadata={
                    "processing_time": processing_time,
                    "file_size": len(content),
                    "num_chunks": len(chunks),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }
            )

        except Exception as e:
            logger.error(f"Error processing transcript {file_path.name}: {e}")
            raise ValueError(f"Failed to process transcript: {e}")

    def _read_file(self, file_path: Path) -> str:
        """Read file content.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Read {len(content)} characters from {file_path.name}")
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            logger.warning(f"Used latin-1 encoding for {file_path.name}")
            return content

    def _chunk_content(
        self, content: str, file_path: Path, company_name: str
    ) -> List[ProcessedChunk]:
        """Chunk content into smaller pieces.

        Args:
            content: Text content to chunk
            file_path: Source file path
            company_name: Company name

        Returns:
            List of ProcessedChunk objects
        """
        chunks = []

        # Try to identify sections (basic heuristic)
        sections = self._identify_sections(content)

        if sections:
            # Chunk by sections
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
                            "char_count": len(section_content),
                        },
                        chunk_id=chunk_id,
                        company_name=company_name,
                        source_file=str(file_path),
                        modality=self.modality_type,
                    )
                )
        else:
            # Fallback: chunk by size
            chunks = self._chunk_by_size(content, file_path, company_name)

        logger.debug(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks

    def _identify_sections(self, content: str) -> List[tuple[str, str]]:
        """Identify sections in the transcript.

        Args:
            content: Text content

        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []

        # Common section patterns in earnings transcripts
        patterns = [
            r"(?i)(MANAGEMENT\s+DISCUSSION|PREPARED\s+REMARKS|OPENING\s+REMARKS)",
            r"(?i)(Q&A\s+SESSION|QUESTIONS?\s+AND\s+ANSWERS?)",
            r"(?i)(CLOSING\s+REMARKS|CONCLUSION)",
            r"(?i)(CEO:?|CFO:?|ANALYST\s*\d*:?)",
        ]

        # Try to split by clear section headers
        lines = content.split("\n")
        current_section = "Introduction"
        current_content = []

        for line in lines:
            # Check if line is a section header
            is_header = False
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_content:
                        sections.append((current_section, "\n".join(current_content)))
                    current_section = line.strip()
                    current_content = []
                    is_header = True
                    break

            if not is_header:
                current_content.append(line)

        # Add the last section
        if current_content:
            sections.append((current_section, "\n".join(current_content)))

        # If we only found one section, return empty (will fallback to size-based chunking)
        if len(sections) <= 1 and len(content) > self.chunk_size * 2:
            return []

        return sections

    def _chunk_by_size(
        self, content: str, file_path: Path, company_name: str
    ) -> List[ProcessedChunk]:
        """Chunk content by size with overlap.

        Args:
            content: Text content
            file_path: Source file path
            company_name: Company name

        Returns:
            List of ProcessedChunk objects
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + self.chunk_size

            # Try to end at a sentence or paragraph boundary
            if end < len(content):
                # Look for paragraph break
                paragraph_break = content.rfind("\n\n", start, end)
                if paragraph_break > start:
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = max(
                        content.rfind(". ", start, end),
                        content.rfind(".\n", start, end),
                    )
                    if sentence_break > start:
                        end = sentence_break + 1

            chunk_content = content[start:end].strip()

            if chunk_content:
                chunk_id = self._generate_chunk_id(
                    f"{company_name}_{file_path.name}_chunk_{chunk_index}"
                )
                chunks.append(
                    ProcessedChunk(
                        content=chunk_content,
                        metadata={
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                            "char_count": len(chunk_content),
                        },
                        chunk_id=chunk_id,
                        company_name=company_name,
                        source_file=str(file_path),
                        modality=self.modality_type,
                    )
                )
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Avoid infinite loop
            if start <= (end - self.chunk_overlap):
                start = end

        return chunks

    def _generate_chunk_id(self, seed: str) -> str:
        """Generate a unique chunk ID.

        Args:
            seed: Seed string for ID generation

        Returns:
            Unique chunk ID
        """
        return hashlib.md5(seed.encode()).hexdigest()[:16]

    def _extract_metrics(self, content: str) -> Dict[str, Any]:
        """Extract key financial metrics from content.

        Args:
            content: Text content

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        # Extract revenue mentions
        revenue_patterns = [
            r"revenue\s+(?:of\s+)?\$?(\d+(?:,\d+)*(?:\.\d+)?)\s*([MB]illion)?",
            r"\$(\d+(?:,\d+)*(?:\.\d+)?)\s*([MB]illion)?\s+(?:in\s+)?revenue",
        ]

        for pattern in revenue_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metrics["revenue_mentions"] = matches
                break

        # Extract margin mentions
        margin_patterns = [
            r"(?:operating|gross|net)\s+margin[s]?\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%",
            r"margin[s]?.*?(\d+(?:\.\d+)?)\s*%",
        ]

        margins = []
        for pattern in margin_patterns:
            margins.extend(re.findall(pattern, content, re.IGNORECASE))
        if margins:
            metrics["margin_mentions"] = margins

        # Extract growth percentages
        growth_pattern = r"(\d+(?:\.\d+)?)\s*%\s+(?:year-over-year|YoY|growth)"
        growth = re.findall(growth_pattern, content, re.IGNORECASE)
        if growth:
            metrics["growth_percentages"] = growth

        # Extract earnings per share
        eps_pattern = r"\$(\d+\.\d+)\s+per\s+(?:diluted\s+)?share"
        eps = re.findall(eps_pattern, content, re.IGNORECASE)
        if eps:
            metrics["eps"] = eps

        logger.debug(f"Extracted {len(metrics)} metric types from content")
        return metrics

    def _generate_summary(self, content: str, company_name: str) -> str:
        """Generate a summary of the transcript.

        Args:
            content: Text content
            company_name: Company name

        Returns:
            Summary string
        """
        # Simple extractive summary (first paragraph + metrics)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        summary_parts = [f"Earnings call transcript for {company_name}."]

        # Add first substantive paragraph
        for para in paragraphs[:3]:
            if len(para) > 50 and not para.startswith("#"):
                summary_parts.append(para[:200] + "..." if len(para) > 200 else para)
                break

        # Add key metrics found
        metrics = self._extract_metrics(content)
        if metrics:
            summary_parts.append(f"Key metrics discussed: {', '.join(metrics.keys())}")

        summary = " ".join(summary_parts)
        logger.debug(f"Generated summary of {len(summary)} characters")
        return summary
