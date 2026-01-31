# Multi-Modal Investment Agent

A sophisticated AI-powered investment analysis system that processes earnings call transcripts, audio recordings, and stock price charts using Retrieval-Augmented Generation (RAG).

**Disclaimer:** This code is written for educational purposes only; readers should seek guidance from qualified professional advisors before making any investment decisions. Do not use it in a production environment.

## Features

- **Multi-Modal Processing**: Text transcripts, audio files, and chart images
- **Vector-Based Search**: FAISS for efficient semantic similarity search
- **AI-Powered Analysis**: Claude 3.5 Sonnet for intelligent Q&A
- **Vision Capabilities**: Claude Haiku for chart analysis
- **Interactive UI**: Streamlit-based web interface
- **Comprehensive Logging**: Detailed logging throughout the pipeline
- **Extensible Architecture**: Plugin-based processor registry

## Requirements

- Python 3.11 or higher
- Anthropic API key (for Claude models)
- OpenAI API key (for Whisper transcription and embeddings)

## Installation

### Option 1: Using pip

```bash
pip install -r requirements.txt
```

### Option 2: Using Poetry

```bash
poetry install
```

## Quick Start

### 1. Configure API Keys

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LOG_LEVEL=INFO
```

### 2. Prepare Data

Create a data directory with company folders containing their respective files:

```
data/
├── CompanyA/
│   ├── transcript.txt
│   ├── earnings_call.mp3
│   └── stock_chart.png
└── CompanyB/
    ├── transcript.txt
    ├── earnings_call.mp3
    └── stock_chart.png
```

**Supported File Types:**
- **Text**: `.txt`, `.md`, `.transcript`
- **Audio**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.pdf`

### 3. Run Application

```bash
streamlit run src/ui/app.py
```

The application will open in your browser at `http://localhost:8501`.

### 4. Use the Application

1. Click **"Ingest Dataset"** in the sidebar to process your data
2. Navigate to the **"Ask Questions"** tab
3. Enter questions about your companies and get AI-powered insights

## Testing

### Verification Tests

Run the verification script to check system setup:

```bash
python run_tests.py
```

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_processors/test_transcript_processor.py -v

# Run specific processor tests
pytest tests/test_processors/ -v
```

## Example Questions

- "What was CompanyA's revenue in Q4 2024?"
- "Compare revenue growth between CompanyA and CompanyB"
- "How has CompanyA's stock price trended over the last quarter?"
- "What were the key topics discussed in CompanyB's earnings call?"
- "Summarize the main financial metrics from the charts"

## Architecture

The system follows a modular architecture:

### Components

- **Processors** ([src/processors/](src/processors/)): Handle different modalities
  - `TranscriptProcessor`: Text document processing
  - `AudioProcessor`: Audio transcription via OpenAI Whisper
  - `ChartProcessor`: Image analysis via Claude Vision
  - `ProcessorRegistry`: Plugin system for extensibility

- **Vector Store** ([src/storage/](src/storage/)): FAISS-based semantic search
  - Efficient vector similarity search
  - Persistent storage support
  - Metadata management

- **Agent** ([src/agent/](src/agent/)): Claude-powered Q&A with RAG
  - Context retrieval from vector store
  - Structured prompts for accurate analysis
  - Citation of sources

- **Ingestion Pipeline** ([src/ingestion/](src/ingestion/)): End-to-end data processing
  - Automatic file type detection
  - Parallel processing support
  - Progress tracking

- **UI** ([src/ui/](src/ui/)): Streamlit web interface
  - Interactive data ingestion
  - Real-time question answering
  - Progress visualization

### Data Flow

```
Raw Data → Processors → Chunks → Embeddings → FAISS Index
                                                    ↓
User Question → Retrieval → Context → Claude → Answer
```

## API Costs (Approximate)

- **Whisper API**: ~$0.006 per minute of audio
- **Claude 3.5 Sonnet**: ~$3-15 per million tokens
- **Claude 3 Haiku**: ~$0.25-1.25 per million tokens
- **OpenAI Embeddings**: ~$0.00013 per 1K tokens

## Project Structure

```
investment_agent/
├── src/
│   ├── agent/              # Investment agent
│   ├── ingestion/          # Data ingestion pipeline
│   ├── llm/                # LLM utilities (embeddings)
│   ├── processors/         # Modality processors
│   ├── storage/            # Vector store
│   └── ui/                 # Streamlit application
├── tests/                  # Unit tests
├── data/                   # Company data (gitignored)
├── requirements.txt        # Pip dependencies
├── pyproject.toml          # Poetry configuration
├── run_tests.py            # Verification script
└── README.md               # This file
```

## Troubleshooting

**API Key Issues**
- Verify keys are correctly set in `.env` file
- Ensure no extra spaces or quotes around keys

**Import Errors**
- Always run commands from the project root directory
- Verify virtual environment is activated

**Memory Issues**
- Process smaller datasets or fewer companies at once
- Reduce chunk sizes in processor configurations

**FAISS Errors**
- Ensure `faiss-cpu` is properly installed
- On some systems, you may need `faiss-gpu` for better performance

## Development

### Code Quality

The project uses:
- **Black**: Code formatting (line length: 88)
- **Ruff**: Fast linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

### Running Linters

```bash
# Format code
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

## Built With

- **[Anthropic Claude](https://www.anthropic.com/)**: Claude 3.5 Sonnet & Claude 3 Haiku
- **[OpenAI](https://openai.com/)**: Whisper API for transcription, embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)**: Efficient vector similarity search
- **[Streamlit](https://streamlit.io/)**: Interactive web UI
- **[Pydantic](https://docs.pydantic.dev/)**: Data validation and settings management

## License

This project is for educational purposes only.

## Contributing

This is an educational project. Feel free to fork and experiment!
