"""Processor registry for managing modality processors."""

import logging
from typing import Dict, Type, Optional, List
from pathlib import Path
from .base_processor import BaseModalityProcessor

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """Registry for managing modality processors.

    The registry allows processors to be registered and retrieved based on
    file extensions. It provides a pluggable architecture where new processors
    can be added without modifying existing code.
    """

    def __init__(self) -> None:
        """Initialize the processor registry."""
        self._processors: Dict[str, Type[BaseModalityProcessor]] = {}
        self._extension_map: Dict[str, str] = {}  # extension -> processor_name
        logger.info("Initialized ProcessorRegistry")

    def register(self, processor_class: Type[BaseModalityProcessor]) -> None:
        """Register a new processor.

        Args:
            processor_class: The processor class to register (not an instance)

        Example:
            registry.register(TranscriptProcessor)
        """
        try:
            # Create a temporary instance to get properties
            processor = processor_class()
            name = processor.__class__.__name__

            # Store the class (not instance)
            self._processors[name] = processor_class

            # Map extensions to processor name
            for ext in processor.supported_extensions:
                ext_lower = ext.lower()
                if ext_lower in self._extension_map:
                    logger.warning(
                        f"Extension {ext} already registered, overwriting with {name}"
                    )
                self._extension_map[ext_lower] = name

            logger.info(
                f"Registered processor {name} for extensions: {processor.supported_extensions}"
            )

        except Exception as e:
            logger.error(f"Failed to register processor {processor_class.__name__}: {e}")
            raise

    def get_processor(self, file_path: Path) -> Optional[BaseModalityProcessor]:
        """Get appropriate processor for a file.

        Args:
            file_path: Path to the file

        Returns:
            Instance of the appropriate processor, or None if no processor found
        """
        ext = file_path.suffix.lower()

        if ext not in self._extension_map:
            logger.debug(f"No processor found for extension: {ext}")
            return None

        processor_name = self._extension_map[ext]
        processor_class = self._processors[processor_name]

        logger.debug(f"Selected processor {processor_name} for file {file_path.name}")

        # Return a new instance of the processor
        return processor_class()

    def list_supported_extensions(self) -> List[str]:
        """Get all supported file extensions.

        Returns:
            List of file extensions (e.g., ['.txt', '.mp3', '.png'])
        """
        return list(self._extension_map.keys())

    def list_processors(self) -> List[str]:
        """Get names of all registered processors.

        Returns:
            List of processor class names
        """
        return list(self._processors.keys())

    def get_processor_for_modality(self, modality: str) -> Optional[Type[BaseModalityProcessor]]:
        """Get processor class for a specific modality type.

        Args:
            modality: Modality type (e.g., 'text', 'audio', 'image')

        Returns:
            Processor class if found, None otherwise
        """
        for processor_class in self._processors.values():
            processor = processor_class()
            if processor.modality_type == modality:
                return processor_class

        logger.debug(f"No processor found for modality: {modality}")
        return None
