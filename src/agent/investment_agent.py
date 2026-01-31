"""Main investment agent using Pydantic AI."""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel
from anthropic import AsyncAnthropic
from ..storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class CompanyQuery(BaseModel):
    """Structured query about companies."""

    query_type: str  # 'fundamental', 'comparison', 'trend'
    companies: List[str]
    metrics: List[str] = []
    time_period: str = "latest"


class InvestmentAgent:
    """Main agent for answering investment questions using RAG."""

    def __init__(self, vector_store: VectorStoreManager) -> None:
        """Initialize the investment agent.

        Args:
            vector_store: Vector store manager
        """
        self.vector_store = vector_store
        self.claude_client = AsyncAnthropic()
        self.system_prompt = self._get_system_prompt()
        logger.info("Initialized InvestmentAgent")

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are an expert investment analyst with access to earnings call transcripts,
audio recordings, and stock price charts. You provide accurate, data-driven insights
about company fundamentals, financial metrics, and market trends.

When answering questions:
1. Ground responses in the provided data
2. Cite specific sources (transcript sections, chart data)
3. Compare metrics accurately
4. Identify trends with supporting evidence
5. Acknowledge limitations in the data
6. Be precise with numbers and percentages
7. Structure your response clearly with sections if needed

Always maintain objectivity and avoid speculation beyond what the data shows."""

    async def answer_query(self, user_question: str) -> str:
        """Answer user question using RAG pipeline.

        Args:
            user_question: The user's question

        Returns:
            Formatted response with sources
        """
        logger.info(f"Processing query: {user_question}")

        try:
            # Retrieve relevant chunks from vector store
            results = await self.vector_store.query(
                query_text=user_question, n_results=10
            )

            if not results:
                logger.warning("No results found for query")
                return "I don't have enough information to answer that question. Please ensure data has been ingested for the relevant companies."

            # Build context from results
            context = self._build_context(results)

            # Generate answer using Claude
            answer = await self._generate_answer(user_question, context, results)

            logger.info("Successfully generated answer")
            return answer

        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context from search results.

        Args:
            results: List of search results

        Returns:
            Context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            document = result["document"]

            company = metadata.get("company_name", "Unknown")
            modality = metadata.get("modality", "unknown")
            source = metadata.get("source_file", "unknown")

            context_parts.append(
                f"[Source {i}] Company: {company} | Type: {modality} | File: {source}\n{document}\n"
            )

        context = "\n---\n".join(context_parts)
        logger.debug(f"Built context with {len(results)} sources")
        return context

    async def _generate_answer(
        self, question: str, context: str, results: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using Claude.

        Args:
            question: User's question
            context: Context from search results
            results: Raw search results for source citations

        Returns:
            Generated answer with sources
        """
        try:
            prompt = f"""Based on the following data sources, answer this question: {question}

Available Data:
{context}

Please provide a comprehensive answer that:
1. Directly answers the question
2. Uses specific numbers and metrics from the data
3. Cites which sources you're using (e.g., "According to Source 1...")
4. Acknowledges if any information is missing or unclear

Answer:"""

            response = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.content[0].text

            # Add sources section
            sources = self._format_sources(results)
            full_answer = f"{answer}\n\n{sources}"

            return full_answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _format_sources(self, results: List[Dict[str, Any]]) -> str:
        """Format sources section.

        Args:
            results: Search results

        Returns:
            Formatted sources string
        """
        sources_list = []

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            company = metadata.get("company_name", "Unknown")
            modality = metadata.get("modality", "unknown")
            source_file = metadata.get("source_file", "unknown")

            sources_list.append(
                f"  [{i}] {company} - {modality.title()} data from {Path(source_file).name}"
            )

        sources = "Sources:\n" + "\n".join(sources_list)
        return sources

    async def compare_companies(
        self, companies: List[str], metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare specific metrics across companies.

        Args:
            companies: List of company names
            metrics: List of metrics to compare

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(companies)} companies on {len(metrics)} metrics")

        comparison = {"companies": companies, "metrics": metrics, "data": {}}

        for company in companies:
            # Query for each company's data on specified metrics
            query_text = f"{company} {' '.join(metrics)}"
            results = await self.vector_store.query(
                query_text=query_text,
                filters={"company_name": company},
                n_results=5,
            )

            comparison["data"][company] = results

        return comparison

    async def analyze_trends(self, company: str, metric: str) -> Dict[str, Any]:
        """Analyze trends for a company metric over time.

        Args:
            company: Company name
            metric: Metric to analyze

        Returns:
            Trend analysis
        """
        logger.info(f"Analyzing trends for {company} - {metric}")

        query_text = f"{company} {metric} trend over time historical"
        results = await self.vector_store.query(
            query_text=query_text,
            filters={"company_name": company},
            n_results=10,
        )

        return {
            "company": company,
            "metric": metric,
            "data_points": results,
            "analysis": "Trend analysis based on available data",
        }


# Import Path for source formatting
from pathlib import Path
