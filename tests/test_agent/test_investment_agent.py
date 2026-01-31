"""Tests for InvestmentAgent."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.investment_agent import InvestmentAgent, CompanyQuery
from src.storage.vector_store import VectorStoreManager


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock_store = MagicMock(spec=VectorStoreManager)
    return mock_store


@pytest.fixture
def sample_query_results():
    """Create sample query results from vector store."""
    return [
        {
            "id": "chunk1",
            "document": "Q4 revenue was $500M, up 25% year-over-year.",
            "metadata": {
                "company_name": "CompanyA",
                "modality": "text",
                "source_file": "transcript.txt",
            },
            "distance": 0.15,
        },
        {
            "id": "chunk2",
            "document": "Operating margin improved to 22% in Q4.",
            "metadata": {
                "company_name": "CompanyA",
                "modality": "text",
                "source_file": "transcript.txt",
            },
            "distance": 0.18,
        },
        {
            "id": "chunk3",
            "document": "Stock price increased from $50 to $75, a 50% gain.",
            "metadata": {
                "company_name": "CompanyA",
                "modality": "image",
                "source_file": "stock_chart.png",
            },
            "distance": 0.22,
        },
    ]


@pytest.fixture
def mock_claude_response():
    """Create a mock Claude API response."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text="According to Source 1, CompanyA achieved Q4 revenue of $500M, "
            "up 25% year-over-year. Source 2 indicates that operating margins "
            "improved to 22%. Source 3 shows the stock price increased 50%."
        )
    ]
    return mock_response


class TestInvestmentAgent:
    """Test InvestmentAgent class."""

    def test_initialization(self, mock_vector_store):
        """Test agent initialization."""
        agent = InvestmentAgent(mock_vector_store)

        assert agent.vector_store == mock_vector_store
        assert agent.system_prompt != ""
        assert "investment analyst" in agent.system_prompt.lower()

    def test_system_prompt_content(self, mock_vector_store):
        """Test that system prompt has required elements."""
        agent = InvestmentAgent(mock_vector_store)

        # Check for key requirements
        assert "data" in agent.system_prompt.lower()
        assert "cite" in agent.system_prompt.lower() or "source" in agent.system_prompt.lower()
        assert "accurate" in agent.system_prompt.lower() or "precise" in agent.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_answer_query_success(
        self, mock_vector_store, sample_query_results, mock_claude_response
    ):
        """Test successful query answering."""
        # Setup mocks
        mock_vector_store.query = AsyncMock(return_value=sample_query_results)

        agent = InvestmentAgent(mock_vector_store)

        with patch.object(agent, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_claude_response)

            result = await agent.answer_query("What was CompanyA's revenue?")

            # Verify vector store was queried
            mock_vector_store.query.assert_called_once()
            call_args = mock_vector_store.query.call_args
            assert call_args[1]["query_text"] == "What was CompanyA's revenue?"
            assert call_args[1]["n_results"] == 10

            # Verify Claude was called
            mock_client.messages.create.assert_called_once()

            # Verify result structure
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Source 1" in result or "source" in result.lower()
            assert "Sources:" in result

    @pytest.mark.asyncio
    async def test_answer_query_no_results(self, mock_vector_store):
        """Test query with no results."""
        mock_vector_store.query = AsyncMock(return_value=[])

        agent = InvestmentAgent(mock_vector_store)
        result = await agent.answer_query("Unknown query")

        assert "don't have enough information" in result.lower()
        mock_vector_store.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_query_handles_exception(
        self, mock_vector_store, sample_query_results
    ):
        """Test query handles exceptions gracefully."""
        mock_vector_store.query = AsyncMock(return_value=sample_query_results)

        agent = InvestmentAgent(mock_vector_store)

        with patch.object(agent, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("API Error")
            )

            result = await agent.answer_query("Test query")

            assert "error" in result.lower()
            assert isinstance(result, str)

    def test_build_context(self, mock_vector_store, sample_query_results):
        """Test context building from results."""
        agent = InvestmentAgent(mock_vector_store)
        context = agent._build_context(sample_query_results)

        # Check that all sources are included
        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "[Source 3]" in context

        # Check that company names are included
        assert "CompanyA" in context

        # Check that modalities are included
        assert "text" in context
        assert "image" in context

        # Check that document content is included
        assert "Q4 revenue was $500M" in context
        assert "Operating margin improved" in context
        assert "Stock price increased" in context

        # Check separators
        assert "---" in context

    def test_build_context_with_missing_metadata(self, mock_vector_store):
        """Test context building with missing metadata."""
        results = [
            {
                "id": "chunk1",
                "document": "Some content",
                "metadata": {},  # Empty metadata
                "distance": 0.1,
            }
        ]

        agent = InvestmentAgent(mock_vector_store)
        context = agent._build_context(results)

        assert "[Source 1]" in context
        assert "Unknown" in context
        assert "unknown" in context.lower()
        assert "Some content" in context

    def test_format_sources(self, mock_vector_store, sample_query_results):
        """Test source formatting."""
        agent = InvestmentAgent(mock_vector_store)
        sources = agent._format_sources(sample_query_results)

        # Check header
        assert "Sources:" in sources

        # Check numbered sources
        assert "[1]" in sources
        assert "[2]" in sources
        assert "[3]" in sources

        # Check company names
        assert "CompanyA" in sources

        # Check modalities
        assert "Text" in sources or "text" in sources
        assert "Image" in sources or "image" in sources

        # Check file names
        assert "transcript.txt" in sources
        assert "stock_chart.png" in sources

    def test_format_sources_with_path_objects(self, mock_vector_store):
        """Test source formatting handles Path objects."""
        results = [
            {
                "id": "chunk1",
                "document": "Content",
                "metadata": {
                    "company_name": "TestCo",
                    "modality": "audio",
                    "source_file": "/path/to/audio_file.mp3",
                },
                "distance": 0.1,
            }
        ]

        agent = InvestmentAgent(mock_vector_store)
        sources = agent._format_sources(results)

        # Should extract just filename, not full path
        assert "audio_file.mp3" in sources
        assert "/path/to/" not in sources or "audio_file.mp3" in sources

    @pytest.mark.asyncio
    async def test_generate_answer_creates_proper_prompt(
        self, mock_vector_store, sample_query_results, mock_claude_response
    ):
        """Test that _generate_answer creates a proper prompt."""
        agent = InvestmentAgent(mock_vector_store)
        context = agent._build_context(sample_query_results)

        with patch.object(agent, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_claude_response)

            await agent._generate_answer(
                "Test question", context, sample_query_results
            )

            # Verify API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args

            # Check model
            assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"

            # Check max_tokens
            assert call_args[1]["max_tokens"] == 2000

            # Check system prompt
            assert call_args[1]["system"] == agent.system_prompt

            # Check user message
            messages = call_args[1]["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "Test question" in messages[0]["content"]
            assert context in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_answer_includes_sources(
        self, mock_vector_store, sample_query_results, mock_claude_response
    ):
        """Test that generated answer includes sources section."""
        agent = InvestmentAgent(mock_vector_store)
        context = agent._build_context(sample_query_results)

        with patch.object(agent, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_claude_response)

            answer = await agent._generate_answer(
                "Test question", context, sample_query_results
            )

            # Verify answer has both the response and sources
            assert "Source 1" in answer
            assert "Sources:" in answer
            assert "CompanyA" in answer

    @pytest.mark.asyncio
    async def test_compare_companies(self, mock_vector_store):
        """Test company comparison functionality."""
        mock_vector_store.query = AsyncMock(
            side_effect=[
                [{"id": "c1", "document": "CompanyA data", "metadata": {}}],
                [{"id": "c2", "document": "CompanyB data", "metadata": {}}],
            ]
        )

        agent = InvestmentAgent(mock_vector_store)
        result = await agent.compare_companies(
            companies=["CompanyA", "CompanyB"], metrics=["revenue", "margin"]
        )

        # Verify structure
        assert "companies" in result
        assert "metrics" in result
        assert "data" in result

        # Verify companies
        assert result["companies"] == ["CompanyA", "CompanyB"]

        # Verify metrics
        assert result["metrics"] == ["revenue", "margin"]

        # Verify data queried for both companies
        assert "CompanyA" in result["data"]
        assert "CompanyB" in result["data"]

        # Verify vector store was called twice (once per company)
        assert mock_vector_store.query.call_count == 2

    @pytest.mark.asyncio
    async def test_compare_companies_queries_with_filters(self, mock_vector_store):
        """Test that compare_companies uses proper filters."""
        mock_vector_store.query = AsyncMock(return_value=[])

        agent = InvestmentAgent(mock_vector_store)
        await agent.compare_companies(
            companies=["TestCompany"], metrics=["revenue"]
        )

        # Check that query was called with company filter
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["filters"] == {"company_name": "TestCompany"}
        assert "TestCompany" in call_args[1]["query_text"]
        assert "revenue" in call_args[1]["query_text"]
        assert call_args[1]["n_results"] == 5

    @pytest.mark.asyncio
    async def test_analyze_trends(self, mock_vector_store):
        """Test trend analysis functionality."""
        mock_results = [
            {"id": "t1", "document": "Q1 data", "metadata": {}},
            {"id": "t2", "document": "Q2 data", "metadata": {}},
        ]
        mock_vector_store.query = AsyncMock(return_value=mock_results)

        agent = InvestmentAgent(mock_vector_store)
        result = await agent.analyze_trends(company="CompanyX", metric="growth")

        # Verify structure
        assert "company" in result
        assert "metric" in result
        assert "data_points" in result
        assert "analysis" in result

        # Verify values
        assert result["company"] == "CompanyX"
        assert result["metric"] == "growth"
        assert result["data_points"] == mock_results

        # Verify query
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert "CompanyX" in call_args[1]["query_text"]
        assert "growth" in call_args[1]["query_text"]
        assert "trend" in call_args[1]["query_text"]
        assert call_args[1]["filters"] == {"company_name": "CompanyX"}

    @pytest.mark.asyncio
    async def test_answer_query_with_multimodal_sources(
        self, mock_vector_store, mock_claude_response
    ):
        """Test query with mixed modality sources."""
        mixed_results = [
            {
                "id": "c1",
                "document": "Text transcript data",
                "metadata": {
                    "company_name": "TestCo",
                    "modality": "text",
                    "source_file": "transcript.txt",
                },
                "distance": 0.1,
            },
            {
                "id": "c2",
                "document": "Audio transcription data",
                "metadata": {
                    "company_name": "TestCo",
                    "modality": "audio",
                    "source_file": "call.mp3",
                },
                "distance": 0.15,
            },
            {
                "id": "c3",
                "document": "Chart analysis data",
                "metadata": {
                    "company_name": "TestCo",
                    "modality": "image",
                    "source_file": "chart.png",
                },
                "distance": 0.2,
            },
        ]

        mock_vector_store.query = AsyncMock(return_value=mixed_results)
        agent = InvestmentAgent(mock_vector_store)

        with patch.object(agent, "claude_client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_claude_response)

            result = await agent.answer_query("Test multimodal query")

            # Verify all modalities are in sources
            assert "text" in result.lower() or "Text" in result
            assert "audio" in result.lower() or "Audio" in result
            assert "image" in result.lower() or "Image" in result


class TestCompanyQuery:
    """Test CompanyQuery model."""

    def test_company_query_creation(self):
        """Test creating a CompanyQuery."""
        query = CompanyQuery(
            query_type="fundamental",
            companies=["CompanyA", "CompanyB"],
            metrics=["revenue", "profit"],
            time_period="Q4 2024",
        )

        assert query.query_type == "fundamental"
        assert len(query.companies) == 2
        assert "CompanyA" in query.companies
        assert len(query.metrics) == 2
        assert query.time_period == "Q4 2024"

    def test_company_query_defaults(self):
        """Test CompanyQuery default values."""
        query = CompanyQuery(query_type="trend", companies=["CompanyX"])

        assert query.metrics == []
        assert query.time_period == "latest"

    def test_company_query_validation(self):
        """Test that CompanyQuery validates required fields."""
        with pytest.raises((ValueError, TypeError)):
            # Missing required fields
            CompanyQuery()
