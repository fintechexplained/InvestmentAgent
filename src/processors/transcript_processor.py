"""Processor for text-based earnings call transcripts."""

import logging
import hashlib
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
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

    # Matches "First Last: text" including middle initials like "Timothy D. Cook: …"
    _SPEAKER_RE = re.compile(
        r"^([A-Z][a-zA-Z]+(?:\s+(?:[A-Z]\.?\s*)?[A-Z][a-zA-Z]+){0,3})\s*:\s*(.*)"
    )
    # Matches bare-role speakers like "CEO:", "Operator:", etc.
    _ROLE_RE = re.compile(
        r"^(CEO|CFO|COO|CTO|Operator|Moderator|Chairman)\s*:\s*(.*)",
        re.IGNORECASE,
    )
    # Leading words that signal a label/header line rather than a speaker name.
    _NON_SPEAKER_PREFIXES = frozenset({
        "key", "financial", "highlights", "overview", "summary",
        "prepared", "opening", "closing", "management", "discussion",
        "revenue", "note", "important", "disclaimer", "forward",
        "safe", "harbor", "total", "net", "gross", "operating",
    })

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
            for i, (section_title, section_content, speaker) in enumerate(sections):
                chunk_id = self._generate_chunk_id(
                    f"{company_name}_{file_path.name}_{section_title}_{speaker}_{i}"
                )
                metadata: Dict[str, Any] = {
                    "section": section_title,
                    "section_index": i,
                    "char_count": len(section_content),
                }
                if speaker:
                    metadata["speaker"] = speaker
                chunks.append(
                    ProcessedChunk(
                        content=section_content,
                        metadata=metadata,
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

    def _identify_sections(self, content: str) -> List[tuple[str, str, str]]:
        """Identify sections in the transcript, including speaker detection.

        Section headers (e.g. "Q&A Session", "Prepared Remarks") delimit major
        blocks.  Q&A sections are further split into individual speaker turns so
        that each chunk carries its own speaker metadata.  Non-Q&A sections record
        the first detected speaker as the primary speaker for that section.

        Args:
            content: Text content

        Returns:
            List of (section_title, section_content, speaker) tuples.
            Speaker is an empty string when no speaker could be detected.
        """
        # Section-header patterns.  Role labels like "CEO:" / "CFO:" are now
        # handled as speaker turns, not section boundaries.
        header_patterns = [
            r"(?i)(MANAGEMENT\s+DISCUSSION|PREPARED\s+REMARKS|OPENING\s+REMARKS)",
            r"(?i)(Q&A\s+SESSION|QUESTIONS?\s+AND\s+ANSWERS?|Q\s*&\s*A)",
            r"(?i)(CLOSING\s+REMARKS|CONCLUSION)",
            r"(?i)(FINANCIAL\s+HIGHLIGHTS|KEY\s+(?:ACHIEVEMENTS|HIGHLIGHTS))",
        ]

        # --- First pass: split into raw sections by header lines ----------
        lines = content.split("\n")
        current_section = "Introduction"
        current_content: List[str] = []
        raw_sections: List[tuple[str, str]] = []

        for line in lines:
            is_header = False
            for pattern in header_patterns:
                if re.match(pattern, line.strip()):
                    if current_content:
                        raw_sections.append(
                            (current_section, "\n".join(current_content))
                        )
                    current_section = line.strip().rstrip(":")
                    current_content = []
                    is_header = True
                    break

            if not is_header:
                current_content.append(line)

        if current_content:
            raw_sections.append((current_section, "\n".join(current_content)))

        # --- Second pass: attach speakers / expand Q&A --------------------
        sections: List[tuple[str, str, str]] = []
        for title, section_content in raw_sections:
            if self._is_qa_section(title):
                turns = self._parse_speaker_turns(section_content)
                if turns:
                    for speaker, text in turns:
                        sections.append(("Q&A", text, speaker))
                else:
                    # No speaker turns detected – keep as a single section
                    sections.append((title, section_content, ""))
            else:
                speaker = self._detect_primary_speaker(section_content)
                sections.append((title, section_content, speaker))

        # Fall back to size-based chunking when only a single section is found
        if len(sections) <= 1 and len(content) > self.chunk_size * 2:
            return []

        return sections

    # ------------------------------------------------------------------
    # Speaker / Q&A helpers
    # ------------------------------------------------------------------

    def _detect_speaker(self, line: str) -> Optional[tuple[str, str]]:
        """Detect a 'Speaker: text' turn in a single line.

        Matches both full names (e.g. "Timothy D. Cook: …") and bare roles
        (e.g. "CEO: …", "Operator: …").  Lines whose leading word is in
        ``_NON_SPEAKER_PREFIXES`` are rejected to avoid false positives on
        metric or section-header labels.

        Args:
            line: A single line of transcript text.

        Returns:
            ``(speaker_name, remaining_text)`` when a speaker is detected.
            ``remaining_text`` may be empty when the name appears on its own
            line.  Returns ``None`` otherwise.
        """
        stripped = line.strip()

        # Role-only speakers (CEO, CFO, Operator, …)
        role_match = self._ROLE_RE.match(stripped)
        if role_match:
            return role_match.group(1), role_match.group(2).strip()

        # Name-style speakers (e.g. "Timothy D. Cook: …")
        name_match = self._SPEAKER_RE.match(stripped)
        if name_match:
            name = name_match.group(1).strip()
            # Filter out lines that look like section headers or metric labels
            if name.split()[0].lower() in self._NON_SPEAKER_PREFIXES:
                return None
            return name, name_match.group(2).strip()

        return None

    def _is_qa_section(self, title: str) -> bool:
        """Return True when *title* looks like a Q&A section header."""
        return bool(
            re.match(r"(?i)(Q&A|QUESTIONS?\s+AND\s+ANSWERS?|Q\s*&\s*A)", title.strip())
        )

    def _detect_primary_speaker(self, content: str) -> str:
        """Return the first speaker name found in *content*, or empty string."""
        for line in content.split("\n"):
            result = self._detect_speaker(line)
            if result:
                return result[0]
        return ""

    def _parse_speaker_turns(self, content: str) -> List[tuple[str, str]]:
        """Split *content* into ``(speaker, text)`` turns.

        Each new ``Speaker: …`` line starts a fresh turn.  Lines appearing
        before the first detected speaker (e.g. ad-network watermarks) are
        silently skipped.  Blank lines within a single speaker's turn are
        preserved to maintain paragraph boundaries.

        Args:
            content: Raw text of a transcript section.

        Returns:
            Ordered list of ``(speaker_name, turn_text)`` pairs.
        """
        turns: List[tuple[str, str]] = []
        current_speaker: Optional[str] = None
        current_lines: List[str] = []

        for line in content.split("\n"):
            stripped = line.strip()

            if not stripped:
                # Preserve blank lines within a turn for paragraph separation
                if current_lines:
                    current_lines.append("")
                continue

            speaker_result = self._detect_speaker(stripped)
            if speaker_result:
                # Flush the previous speaker's accumulated text
                if current_speaker is not None:
                    text = "\n".join(current_lines).strip()
                    if text:
                        turns.append((current_speaker, text))
                current_speaker, first_text = speaker_result
                current_lines = [first_text] if first_text else []
            elif current_speaker is not None:
                current_lines.append(stripped)
            # Lines before the first speaker are noise – skip them

        # Flush the final turn
        if current_speaker is not None:
            text = "\n".join(current_lines).strip()
            if text:
                turns.append((current_speaker, text))

        return turns

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
                metadata: Dict[str, Any] = {
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "char_count": len(chunk_content),
                }
                speaker = self._detect_primary_speaker(chunk_content)
                if speaker:
                    metadata["speaker"] = speaker
                chunks.append(
                    ProcessedChunk(
                        content=chunk_content,
                        metadata=metadata,
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
