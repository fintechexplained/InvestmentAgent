"""Investment Agent - Multi-modal investment analysis system."""

import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('investment_agent.log'),
        logging.StreamHandler()
    ]
)

__version__ = "0.1.0"
