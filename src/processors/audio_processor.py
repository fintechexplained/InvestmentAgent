"""Processor for audio files using Whisper API."""

import logging
import time
from pathlib import Path
from typing import List
from openai import AsyncOpenAI
from .base_processor import BaseModalityProcessor, ProcessingResult
from .transcript_processor import TranscriptProcessor

logger = logging.getLogger(__name__)


class AudioProcessor(BaseModalityProcessor):
    """Processes MP3 audio files using Whisper API.

    This processor:
    1. Transcribes audio using Whisper API
    2. Delegates to TranscriptProcessor for text processing
    3. Adds audio-specific metadata
    """

    def __init__(self) -> None:
        """Initialize the audio processor."""
        super().__init__()
        self.whisper_client = AsyncOpenAI()
        self.transcript_processor = TranscriptProcessor()

    @property
    def supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

    @property
    def modality_type(self) -> str:
        """Get modality type."""
        return "audio"

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
                logger.debug(f"Audio file {file_path.name} is valid for processing")
            else:
                logger.warning(f"Audio file {file_path.name} failed validation")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False

    async def process(self, file_path: Path, company_name: str) -> ProcessingResult:
        """Process audio file.

        Args:
            file_path: Path to the audio file
            company_name: Name of the company

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()
        logger.info(f"Processing audio file {file_path.name} for {company_name}")

        try:
            # Transcribe audio using Whisper
            transcription = await self._transcribe_audio(file_path)

            if not transcription.strip():
                logger.warning(f"Empty transcription for {file_path.name}")
                return ProcessingResult(
                    chunks=[],
                    summary="Empty audio transcription",
                    processing_metadata={
                        "processing_time": time.time() - start_time,
                        "error": "Empty transcription",
                    },
                )

            logger.info(
                f"Transcribed {file_path.name}: {len(transcription)} characters"
            )

            # Process transcription using TranscriptProcessor
            # Create a temporary text representation
            result = await self._process_transcription(
                transcription, file_path, company_name
            )

            # Update processing metadata
            processing_time = time.time() - start_time
            result.processing_metadata.update({
                "processing_time": processing_time,
                "transcription_length": len(transcription),
                "audio_file": str(file_path),
                "transcription_method": "whisper",
            })

            # Update modality for all chunks to 'audio'
            for chunk in result.chunks:
                chunk.modality = "audio"
                chunk.metadata["original_modality"] = "audio"
                chunk.metadata["transcribed"] = True

            logger.info(
                f"Processed audio {file_path.name}: {len(result.chunks)} chunks "
                f"in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing audio file {file_path.name}: {e}")
            raise ValueError(f"Failed to process audio: {e}")

    async def _transcribe_audio(self, file_path: Path) -> str:
        """Transcribe audio file using Whisper API.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text
        """
        try:
            logger.debug(f"Transcribing audio file {file_path.name}")

            with open(file_path, "rb") as audio_file:
                transcription = await self.whisper_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            # The response is the transcribed text directly
            if isinstance(transcription, str):
                text = transcription
            else:
                text = transcription.text if hasattr(transcription, "text") else str(transcription)

            logger.debug(f"Successfully transcribed {file_path.name}")
            return text

        except Exception as e:
            logger.error(f"Whisper API error for {file_path.name}: {e}")
            raise

    async def _process_transcription(
        self, transcription: str, audio_file: Path, company_name: str
    ) -> ProcessingResult:
        """Process transcription text.

        Args:
            transcription: Transcribed text
            audio_file: Original audio file path
            company_name: Company name

        Returns:
            ProcessingResult from transcript processing
        """
        # Use TranscriptProcessor to process the transcription
        # We need to create a temporary path-like object for the transcript processor
        class TranscriptionWrapper:
            def __init__(self, original_path: Path, content: str):
                self.original_path = original_path
                self.content = content
                self.name = original_path.name
                self.suffix = original_path.suffix

            def __str__(self) -> str:
                return str(self.original_path)

        # Process using transcript processor's internal methods
        metrics = self.transcript_processor._extract_metrics(transcription)
        summary = self.transcript_processor._generate_summary(transcription, company_name)
        chunks = self.transcript_processor._chunk_content(
            transcription, audio_file, company_name
        )

        return ProcessingResult(
            chunks=chunks,
            summary=summary,
            extracted_metrics=metrics,
            processing_metadata={
                "source_type": "audio_transcription",
            },
        )
