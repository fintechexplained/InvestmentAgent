"""Ingestion pipeline for multi-modal data."""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from ..processors.registry import ProcessorRegistry
from ..storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates ingestion of multi-modal data."""

    def __init__(
        self,
        registry: ProcessorRegistry,
        vector_store: VectorStoreManager,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            registry: Processor registry
            vector_store: Vector store manager
        """
        self.registry = registry
        self.vector_store = vector_store
        logger.info("Initialized IngestionPipeline")

    async def ingest_company(
        self, company_name: str, files: List[Path]
    ) -> Dict[str, Any]:
        """Ingest all files for a company.

        Args:
            company_name: Name of the company
            files: List of file paths to ingest

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion for {company_name} with {len(files)} files")

        stats = {
            "company_name": company_name,
            "total_files": len(files),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "errors": [],
        }

        # Process files
        for file_path in files:
            try:
                logger.info(f"Processing file: {file_path.name}")

                # Get appropriate processor
                processor = self.registry.get_processor(file_path)

                if processor is None:
                    logger.warning(
                        f"No processor found for {file_path.name}, skipping"
                    )
                    stats["failed_files"] += 1
                    stats["errors"].append(
                        f"No processor for {file_path.name}"
                    )
                    continue

                # Validate file
                is_valid = await processor.validate_file(file_path)
                if not is_valid:
                    logger.warning(f"File validation failed for {file_path.name}")
                    stats["failed_files"] += 1
                    stats["errors"].append(
                        f"Validation failed for {file_path.name}"
                    )
                    continue

                # Process file
                result = await processor.process(file_path, company_name)

                # Add chunks to vector store
                await self.vector_store.add_chunks(result.chunks)

                stats["processed_files"] += 1
                stats["total_chunks"] += len(result.chunks)

                logger.info(
                    f"Successfully processed {file_path.name}: "
                    f"{len(result.chunks)} chunks"
                )

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                stats["failed_files"] += 1
                stats["errors"].append(f"{file_path.name}: {str(e)}")

        logger.info(
            f"Completed ingestion for {company_name}: "
            f"{stats['processed_files']}/{stats['total_files']} files, "
            f"{stats['total_chunks']} total chunks"
        )

        return stats

    async def ingest_dataset(self, data_dir: Path) -> Dict[str, Any]:
        """Ingest entire dataset.

        Expected structure:
        data/
          CompanyA/
            transcript.txt
            earnings_call.mp3
            stock_chart.png
          CompanyB/
            transcript.txt
            earnings_call.mp3
            stock_chart.png

        Args:
            data_dir: Root data directory

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting dataset ingestion from {data_dir}")

        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        overall_stats = {
            "data_dir": str(data_dir),
            "total_companies": 0,
            "total_files": 0,
            "total_chunks": 0,
            "companies": {},
        }

        # Find company directories
        company_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

        logger.info(f"Found {len(company_dirs)} company directories")

        for company_dir in company_dirs:
            company_name = company_dir.name

            # Get all files in company directory
            files = [f for f in company_dir.iterdir() if f.is_file()]

            if not files:
                logger.warning(f"No files found for {company_name}")
                continue

            # Ingest company data
            company_stats = await self.ingest_company(company_name, files)

            overall_stats["companies"][company_name] = company_stats
            overall_stats["total_companies"] += 1
            overall_stats["total_files"] += company_stats["processed_files"]
            overall_stats["total_chunks"] += company_stats["total_chunks"]

        logger.info(
            f"Completed dataset ingestion: "
            f"{overall_stats['total_companies']} companies, "
            f"{overall_stats['total_files']} files, "
            f"{overall_stats['total_chunks']} chunks"
        )

        return overall_stats
