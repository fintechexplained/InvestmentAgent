# Multi-Modal Investment Agent

A sophisticated AI-powered investment analysis system that processes earnings call transcripts, audio recordings, and stock price charts.

Disclaimer: This code is written for educational purposes only; readers should seek guidance from qualified professional advisors before making any investment decisions.  Do not use it in a production environment.

## Features

- Multi-Modal Processing: Text, audio, and images
- Vector-Based Search using ChromaDB
- AI-Powered Analysis with Claude 3.5 Sonnet
- Interactive Streamlit UI
- Comprehensive Logging

## Quick Start

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Configure API Keys

cp .env.example .env
# Edit .env file and add:
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

### 3. Prepare Data

Create data directory structure:
data/
  CompanyA/
    transcript.txt
    earnings_call.mp3
    stock_chart.png

### 4. Run Application

streamlit run src/ui/app.py

### 5. Use the App

1. Click Ingest Dataset in sidebar
2. Ask questions in the Ask Questions tab

## Running Tests

# All tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_processors/test_transcript_processor.py -v

## Example Questions

- What was CompanyA revenue in Q4?
- Compare CompanyA and CompanyB revenue growth
- How has CompanyA stock trended over time?

## Architecture

- Processors: Handle text, audio, image data
- Vector Store: ChromaDB for semantic search  
- Agent: Claude-powered analysis with RAG
- UI: Streamlit interface

## API Costs

- Whisper: ~0.006 USD per minute of audio
- Claude: ~3-15 USD per million tokens
- Embeddings: ~0.00013 USD per 1K tokens

## Troubleshooting

API Key Issues: Check .env file
Import Errors: Run from project root
Memory Issues: Process smaller datasets

## Built With

- Anthropic Claude
- OpenAI Whisper
- ChromaDB
- Streamlit
