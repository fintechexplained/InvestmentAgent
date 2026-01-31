#!/usr/bin/env python
"""Quick test script to verify the installation."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.processors import (
            TranscriptProcessor,
            AudioProcessor,
            ChartProcessor,
            ProcessorRegistry
        )
        from src.storage import VectorStoreManager
        from src.agent import InvestmentAgent
        from src.ingestion import IngestionPipeline
        logger.info("All imports successful!")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

def test_processor_registry():
    """Test processor registry."""
    logger.info("Testing processor registry...")
    
    try:
        from src.processors import ProcessorRegistry, TranscriptProcessor
        
        registry = ProcessorRegistry()
        registry.register(TranscriptProcessor)
        
        extensions = registry.list_supported_extensions()
        logger.info(f"Supported extensions: {extensions}")
        
        assert '.txt' in extensions
        logger.info("Processor registry test passed!")
        return True
    except Exception as e:
        logger.error(f"Registry test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting verification tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Processor Registry", test_processor_registry),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        logger.info(f"\n--- Running {name} Test ---")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n\n=== Results ===")
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("\nAll tests passed! System is ready to use.")
        logger.info("Next steps:")
        logger.info("1. Add API keys to .env file")
        logger.info("2. Run: streamlit run src/ui/app.py")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
