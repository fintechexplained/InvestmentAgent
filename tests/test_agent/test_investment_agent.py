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


class TestConversationHistory:
    """Test conversation history functionality."""

    def test_initial_history_is_empty(self, mock_vector_store):
        """Test that message history starts empty."""
        agent = InvestmentAgent(mock_vector_store)
        assert agent.get_history_length() == 0
        assert agent.message_history == []

    def test_clear_history(self, mock_vector_store):
        """Test clearing conversation history."""
        agent = InvestmentAgent(mock_vector_store)

        # Add some mock messages
        agent.message_history = [Mock(), Mock(), Mock()]
        assert agent.get_history_length() == 3

        # Clear history
        agent.clear_history()
        assert agent.get_history_length() == 0
        assert agent.message_history == []

    @pytest.mark.asyncio
    async def test_message_history_updates_after_query(self, mock_vector_store):
        """Test that message history is updated after a query."""
        agent = InvestmentAgent(mock_vector_store)

        # Mock the agent.run method to return a result with messages
        mock_result = Mock()
        mock_result.output = "Test answer"
        mock_result.all_messages = Mock(return_value=[Mock(), Mock()])

        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            initial_length = agent.get_history_length()
            assert initial_length == 0

            # Run a query
            answer = await agent.answer_query("What is the revenue?")

            # Check that history was updated
            assert agent.get_history_length() == 2
            assert answer == "Test answer"

    @pytest.mark.asyncio
    async def test_message_history_passed_to_agent(self, mock_vector_store):
        """Test that message history is passed to subsequent queries."""
        agent = InvestmentAgent(mock_vector_store)

        # Mock the agent.run method
        mock_result = Mock()
        mock_result.output = "First answer"
        mock_result.all_messages = Mock(return_value=[Mock()])

        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            # First query
            await agent.answer_query("First question")

            # Verify run was called with empty history first time
            assert mock_run.call_count == 1
            first_call_kwargs = mock_run.call_args[1]
            assert first_call_kwargs['message_history'] == []

            # Second query
            mock_result.output = "Second answer"
            await agent.answer_query("Second question")

            # Verify run was called with message history second time
            assert mock_run.call_count == 2
            second_call_kwargs = mock_run.call_args[1]
            assert len(second_call_kwargs['message_history']) > 0

    @pytest.mark.asyncio
    async def test_history_persists_across_queries(self, mock_vector_store):
        """Test that conversation history persists across multiple queries."""
        agent = InvestmentAgent(mock_vector_store)

        # Create mock results that build up history
        def create_mock_result(output, num_messages):
            result = Mock()
            result.output = output
            result.all_messages = Mock(return_value=[Mock() for _ in range(num_messages)])
            return result

        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            # First query - 2 messages (user + assistant)
            mock_run.return_value = create_mock_result("Answer 1", 2)
            await agent.answer_query("Question 1")
            assert agent.get_history_length() == 2

            # Second query - 4 messages (previous 2 + new user + assistant)
            mock_run.return_value = create_mock_result("Answer 2", 4)
            await agent.answer_query("Question 2")
            assert agent.get_history_length() == 4

            # Third query - 6 messages
            mock_run.return_value = create_mock_result("Answer 3", 6)
            await agent.answer_query("Question 3")
            assert agent.get_history_length() == 6

    @pytest.mark.asyncio
    async def test_clear_resets_conversation_context(self, mock_vector_store):
        """Test that clearing history allows starting a fresh conversation."""
        agent = InvestmentAgent(mock_vector_store)

        mock_result = Mock()
        mock_result.output = "Answer"
        mock_result.all_messages = Mock(return_value=[Mock(), Mock()])

        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            # Build up some history
            await agent.answer_query("First question")
            await agent.answer_query("Second question")
            assert agent.get_history_length() > 0

            # Clear history
            agent.clear_history()
            assert agent.get_history_length() == 0

            # Next query should start fresh
            await agent.answer_query("New question after clear")

            # Verify the last call had empty history
            last_call_kwargs = mock_run.call_args[1]
            assert last_call_kwargs['message_history'] == []

    def test_agent_has_conversation_methods(self, mock_vector_store):
        """Test that agent has all conversation-related methods."""
        agent = InvestmentAgent(mock_vector_store)

        assert hasattr(agent, 'message_history')
        assert hasattr(agent, 'answer_query')
        assert hasattr(agent, 'clear_history')
        assert hasattr(agent, 'get_history_length')
