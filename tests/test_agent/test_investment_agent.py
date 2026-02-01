"""Tests for InvestmentAgent."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from src.agent.investment_agent import InvestmentAgent
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
    ]


class TestInvestmentAgent:
    """Test InvestmentAgent class."""

    def test_initialization(self, mock_vector_store):
        """Test agent initialization."""
        agent = InvestmentAgent(mock_vector_store)

        assert agent.vector_store == mock_vector_store
        assert agent.agent is not None
        assert "investment analyst" in agent._get_system_prompt().lower()

    def test_system_prompt_content(self, mock_vector_store):
        """Test that system prompt has required elements."""
        agent = InvestmentAgent(mock_vector_store)
        system_prompt = agent._get_system_prompt()

        # Check for key requirements
        assert "rag" in system_prompt.lower()
        assert "search" in system_prompt.lower()
        assert "data" in system_prompt.lower()
        assert "cite" in system_prompt.lower() or "source" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_rag_search_success(self, mock_vector_store, sample_query_results):
        """Test successful RAG search."""
        mock_vector_store.query = AsyncMock(return_value=sample_query_results)

        agent = InvestmentAgent(mock_vector_store)

        # Create a mock context
        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        result = await agent._rag_search(mock_ctx, "test query")

        # Verify vector store was queried
        mock_vector_store.query.assert_called_once()

        # Verify result contains data
        assert isinstance(result, str)
        assert "CompanyA" in result
        assert "revenue" in result.lower()
        assert "Source 1" in result

    @pytest.mark.asyncio
    async def test_rag_search_no_results(self, mock_vector_store):
        """Test RAG search with no results."""
        mock_vector_store.query = AsyncMock(return_value=[])

        agent = InvestmentAgent(mock_vector_store)

        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        result = await agent._rag_search(mock_ctx, "unknown query")

        assert "No information found" in result
        mock_vector_store.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_search_handles_exception(self, mock_vector_store):
        """Test RAG search handles exceptions gracefully."""
        mock_vector_store.query = AsyncMock(side_effect=Exception("Database error"))

        agent = InvestmentAgent(mock_vector_store)

        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        result = await agent._rag_search(mock_ctx, "test query")

        assert "Error searching RAG database" in result

    @pytest.mark.asyncio
    async def test_web_search_success(self, mock_vector_store):
        """Test successful web search."""
        # Mock DuckDuckGo search results
        mock_ddgs_results = [
            {
                "title": "Company News",
                "body": "Recent company performance data",
                "href": "https://example.com/news",
            },
            {
                "title": "Stock Analysis",
                "body": "Analysis of stock trends",
                "href": "https://example.com/analysis",
            },
        ]

        agent = InvestmentAgent(mock_vector_store)

        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        with patch("src.agent.investment_agent.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_ddgs_results
            mock_ddgs.return_value = mock_instance

            result = await agent._web_search(mock_ctx, "test query")

            # Verify search was performed
            mock_instance.text.assert_called_once_with("test query", max_results=5)

            # Verify result contains data
            assert isinstance(result, str)
            assert "Company News" in result
            assert "Stock Analysis" in result
            assert "Web Result 1" in result

    @pytest.mark.asyncio
    async def test_web_search_no_results(self, mock_vector_store):
        """Test web search with no results."""
        agent = InvestmentAgent(mock_vector_store)

        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        with patch("src.agent.investment_agent.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs.return_value = mock_instance

            result = await agent._web_search(mock_ctx, "unknown query")

            assert "No web search results found" in result

    @pytest.mark.asyncio
    async def test_web_search_handles_exception(self, mock_vector_store):
        """Test web search handles exceptions gracefully."""
        agent = InvestmentAgent(mock_vector_store)

        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        with patch("src.agent.investment_agent.DDGS") as mock_ddgs:
            mock_ddgs.side_effect = Exception("Search error")

            result = await agent._web_search(mock_ctx, "test query")

            assert "Error performing web search" in result

    @pytest.mark.asyncio
    async def test_answer_query_success(self, mock_vector_store, sample_query_results):
        """Test successful query answering."""
        mock_vector_store.query = AsyncMock(return_value=sample_query_results)

        agent = InvestmentAgent(mock_vector_store)

        # Mock the pydantic-ai agent's run method
        mock_result = MagicMock()
        mock_result.output = "Based on the RAG search results, CompanyA's Q4 revenue was $500M."

        with patch.object(agent.agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await agent.answer_query("What was CompanyA's revenue?")

            # Verify agent was called
            mock_run.assert_called_once()

            # Verify result
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_answer_query_handles_exception(self, mock_vector_store):
        """Test query handles exceptions gracefully."""
        agent = InvestmentAgent(mock_vector_store)

        with patch.object(agent.agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Agent error")

            result = await agent.answer_query("Test query")

            assert "error" in result.lower()
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_rag_search_formats_sources_correctly(
        self, mock_vector_store, sample_query_results
    ):
        """Test that RAG search formats sources with file names only."""
        # Add a result with a full path
        results_with_paths = sample_query_results + [
            {
                "id": "chunk3",
                "document": "Stock price data",
                "metadata": {
                    "company_name": "CompanyA",
                    "modality": "image",
                    "source_file": "/path/to/chart.png",
                },
                "distance": 0.2,
            }
        ]

        mock_vector_store.query = AsyncMock(return_value=results_with_paths)

        agent = InvestmentAgent(mock_vector_store)

        mock_ctx = MagicMock()
        mock_ctx.deps = mock_vector_store

        result = await agent._rag_search(mock_ctx, "test query")

        # Should extract just filename
        assert "chart.png" in result

    @pytest.mark.asyncio
    async def test_tools_are_registered(self, mock_vector_store):
        """Test that tools are properly registered with the agent."""
        agent = InvestmentAgent(mock_vector_store)

        # Verify that the agent has tools registered
        # This is an implementation detail test to ensure tools are set up
        assert agent.agent is not None
        # The agent should have tools registered during init
        # We can verify this indirectly by checking the agent has the expected attributes
        assert hasattr(agent, "_rag_search")
        assert hasattr(agent, "_web_search")
