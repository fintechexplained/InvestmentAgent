"""Main investment agent using Pydantic AI."""

import logging
from typing import List, Dict, Any
from pydantic_ai import Agent, RunContext
from ddgs import DDGS
from ..storage.vector_store import VectorStoreManager
from pathlib import Path

logger = logging.getLogger(__name__)


class InvestmentAgent:
    """Main agent for answering investment questions using RAG and web search."""

    def __init__(self, vector_store: VectorStoreManager) -> None:
        """Initialize the investment agent.

        Args:
            vector_store: Vector store manager
        """
        self.vector_store = vector_store

        # Create pydantic-ai agent
        # Note: Using claude-3-haiku (fastest, most economical model available on this account)
        self.agent = Agent(
            model="anthropic:claude-3-haiku-20240307",
            system_prompt=self._get_system_prompt(),
            deps_type=VectorStoreManager,
            retries=2,
        )

        # Register tools
        self.agent.tool(self._rag_search)
        self.agent.tool(self._web_search)

        logger.info("Initialized InvestmentAgent with pydantic-ai")

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are an expert investment analyst with access to:
1. A RAG database containing earnings call transcripts, audio recordings, and stock price charts
2. Web search capabilities via DuckDuckGo

Your primary source of information should be the RAG database. Use the rag_search tool first to find information about companies in the database.

If you cannot find sufficient information in the RAG database, or if the question is about recent events or companies not in the database, use the web_search tool to find current information.

When answering questions:
1. Always try RAG search first for company information
2. Use web search as a fallback or for recent events
3. Ground responses in the provided data
4. Cite specific sources (transcript sections, chart data, or web sources)
5. Compare metrics accurately
6. Identify trends with supporting evidence
7. Acknowledge limitations in the data
8. Be precise with numbers and percentages
9. Structure your response clearly with sections if needed

Always maintain objectivity and avoid speculation beyond what the data shows."""

    async def _rag_search(
        self, ctx: RunContext[VectorStoreManager], query: str, n_results: int = 10
    ) -> str:
        """Search the RAG database for information about companies.

        Use this tool to find information from earnings call transcripts, audio recordings,
        and stock price charts that have been ingested into the database.

        Args:
            ctx: Run context with vector store
            query: The search query
            n_results: Number of results to return (default: 10)

        Returns:
            Formatted search results with sources
        """
        logger.info(f"RAG search: {query}")

        try:
            results = await ctx.deps.query(query_text=query, n_results=n_results)

            if not results:
                return "No information found in the RAG database for this query."

            # Build context from results
            context_parts = []
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                document = result["document"]

                company = metadata.get("company_name", "Unknown")
                modality = metadata.get("modality", "unknown")
                source = metadata.get("source_file", "unknown")

                context_parts.append(
                    f"[Source {i}] Company: {company} | Type: {modality} | File: {Path(source).name}\n{document}"
                )

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return f"Error searching RAG database: {str(e)}"

    async def _web_search(
        self, ctx: RunContext[VectorStoreManager], query: str, max_results: int = 5
    ) -> str:
        """Search the web using DuckDuckGo for current information.

        Use this tool when:
        - Information is not available in the RAG database
        - You need recent/current information
        - The query is about companies not in the database

        Args:
            ctx: Run context (not used but required by pydantic-ai)
            query: The search query
            max_results: Maximum number of results to return (default: 5)

        Returns:
            Formatted search results from the web
        """
        logger.info(f"Web search: {query}")

        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            if not results:
                return "No web search results found for this query."

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                body = result.get("body", "No description")
                url = result.get("href", "No URL")

                formatted_results.append(
                    f"[Web Result {i}]\nTitle: {title}\nURL: {url}\n{body}"
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error performing web search: {str(e)}"

    async def answer_query(self, user_question: str) -> str:
        """Answer user question using the agent with RAG and web search tools.

        Args:
            user_question: The user's question

        Returns:
            Formatted response with sources
        """
        logger.info(f"Processing query: {user_question}")

        try:
            # Run the agent with the question
            result = await self.agent.run(
                user_question,
                deps=self.vector_store
            )

            answer = result.output
            logger.info("Successfully generated answer")
            return answer

        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
