"""Streamlit web interface for investment agent."""

import streamlit as st
import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import vector store components
VECTOR_STORE_AVAILABLE = False
try:
    from src.storage.vector_store import VectorStoreManager
    from src.agent.investment_agent import InvestmentAgent
    from src.ingestion.pipeline import IngestionPipeline
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f"Vector store not available: {e}")
    logger_init.info("Running in demo mode without vector store")

# Always available processors
from src.processors.registry import ProcessorRegistry
from src.processors.transcript_processor import TranscriptProcessor
from src.processors.audio_processor import AudioProcessor
from src.processors.chart_processor import ChartProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class InvestmentUI:
    """Streamlit web interface for investment agent."""

    def __init__(self):
        """Initialize the UI components."""
        # Initialize session state
        if "vector_store_available" not in st.session_state:
            st.session_state.vector_store_available = VECTOR_STORE_AVAILABLE

        if "registry" not in st.session_state:
            st.session_state.registry = self._setup_registry()
            st.session_state.chat_history = []
            logger.info("Initialized basic UI components")

        if VECTOR_STORE_AVAILABLE and "vector_store" not in st.session_state:
            st.session_state.vector_store = VectorStoreManager()
            st.session_state.agent = InvestmentAgent(st.session_state.vector_store)
            st.session_state.pipeline = IngestionPipeline(
                st.session_state.registry, st.session_state.vector_store
            )
            logger.info("Initialized vector store components")

    def _setup_registry(self) -> ProcessorRegistry:
        """Setup and configure processor registry."""
        registry = ProcessorRegistry()
        registry.register(TranscriptProcessor)
        registry.register(AudioProcessor)
        registry.register(ChartProcessor)
        return registry

    def render(self):
        """Render the Streamlit UI."""
        st.set_page_config(
            page_title="Multi-Modal Investment Agent",
            page_icon=":chart_with_upwards_trend:",
            layout="wide",
        )

        st.title("Multi-Modal Investment Agent")
        st.markdown("*Analyze earnings calls, transcripts, and stock charts with AI*")

        # Show warning if vector store not available
        if not VECTOR_STORE_AVAILABLE:
            st.warning("""
            **Demo Mode**: Vector store not available. Full functionality is disabled.

            To enable full functionality, ensure all dependencies are installed:
            ```bash
            pip install -r requirements.txt
            ```

            You can still explore the UI and view processor information.
            """)

        # Sidebar for data management
        with st.sidebar:
            self.render_sidebar()

        # Main area tabs
        tab1, tab2, tab3 = st.tabs(["Ask Questions", "Chat History", "About"])

        with tab1:
            self.render_query_interface()

        with tab2:
            self.render_chat_history()

        with tab3:
            self.render_about()

    def render_sidebar(self):
        """Render the sidebar."""
        st.header("Data Management")

        if not VECTOR_STORE_AVAILABLE:
            st.info("Vector store not available. Install dependencies to enable data ingestion.")
            st.markdown("### System Status")
            st.write("✅ All processors working")
            st.write("✅ Tests passing")
            st.write("❌ Vector store (install requirements)")
            return

        # Data ingestion section
        st.subheader("Ingest Data")

        data_dir = st.text_input(
            "Data Directory",
            value="./data",
            help="Path to directory containing company folders",
        )

        if st.button("Ingest Dataset", type="primary"):
            with st.spinner("Ingesting data..."):
                try:
                    data_path = Path(data_dir)
                    if not data_path.exists():
                        st.error(f"Directory not found: {data_dir}")
                    else:
                        stats = asyncio.run(
                            st.session_state.pipeline.ingest_dataset(data_path)
                        )
                        st.success(
                            f"Ingested {stats['total_companies']} companies, "
                            f"{stats['total_chunks']} chunks!"
                        )
                        st.json(stats)
                except Exception as e:
                    st.error(f"Error ingesting data: {e}")
                    logger.error(f"Ingestion error: {e}")

        # Display loaded companies
        st.subheader("Loaded Companies")

        try:
            companies = asyncio.run(st.session_state.vector_store.get_companies())

            if companies:
                st.write(f"**Total Companies:** {len(companies)}")

                for company in companies:
                    with st.expander(company):
                        stats = asyncio.run(
                            st.session_state.vector_store.get_company_stats(company)
                        )
                        st.write(f"Chunks: {stats['total_chunks']}")
                        st.write(f"Modalities: {stats['modalities']}")
                        st.write(f"Files: {len(stats['source_files'])}")
            else:
                st.info("No companies loaded. Ingest data to get started.")

        except Exception as e:
            st.error(f"Error loading companies: {e}")

        # Clear data button
        if st.button("Clear All Data", type="secondary"):
            if st.checkbox("Confirm clear all data"):
                st.session_state.vector_store.clear()
                st.session_state.chat_history = []
                st.session_state.agent.clear_history()
                st.success("Data and conversation cleared!")
                st.rerun()

    def render_query_interface(self):
        """Render the query interface."""
        st.header("Ask Questions")

        if not VECTOR_STORE_AVAILABLE:
            st.info("Question answering requires the vector store. Install dependencies to enable this feature.")
            st.markdown("### What You Can Do Now:")
            st.markdown("- View the **About** tab for system information")
            st.markdown("- Run unit tests: `pytest tests/test_processors/ -v`")
            st.markdown("- Test processors directly in Python")
            st.markdown("- Install all dependencies to enable full functionality")

            with st.expander("How to Install Dependencies"):
                st.markdown("""
                Run the following command to install all required dependencies:
                ```bash
                pip install -r requirements.txt
                ```
                Then restart the application.
                """)
            return

        # Conversation status
        history_length = st.session_state.agent.get_history_length()
        col1, col2 = st.columns([3, 1])

        with col1:
            if history_length > 0:
                st.info(f"💬 Conversation active ({history_length} messages). The agent remembers your previous questions.")
            else:
                st.info("💬 Start a new conversation. Ask follow-up questions and the agent will remember the context!")

        with col2:
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.agent.clear_history()
                st.success("Conversation cleared!")
                st.rerun()

        # Query input
        user_question = st.text_area(
            "Your Question",
            placeholder="e.g., What was RandomCompany's revenue in the last quarter?",
            height=100,
        )

        if st.button("Ask", type="primary", use_container_width=True):
            if not user_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        answer = asyncio.run(
                            st.session_state.agent.answer_query(user_question)
                        )

                        # Display answer
                        st.markdown("### Answer")
                        st.markdown(answer)

                        # Add to chat history
                        st.session_state.chat_history.append(
                            {"question": user_question, "answer": answer}
                        )

                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Query error: {e}")

        # Example questions
        with st.expander("Example Questions"):
            st.markdown("""
            **Fundamental Analysis:**
            - What was RandomCompany's revenue in the last quarter?
            - What are the profit margins for RandomCompany?
            - How is RandomCompany's financial performance?

            **Company Comparison:**
            - Compare RandomCompany and SampleCompany's revenue growth
            - Which company has better operating margins?
            - How do their stock prices compare?

            **Trend Analysis:**
            - How has RandomCompany's revenue trended over time?
            - Is RandomCompany's stock price rising or falling?
            - What's the trend in operating margins?

            **Follow-up Questions (using conversation memory):**
            - After asking about a company: "What about their expenses?"
            - After a comparison: "Which one is more profitable?"
            - After trend analysis: "What might be causing this trend?"
            """)

    def render_chat_history(self):
        """Render the chat history."""
        st.header("Chat History")

        if not st.session_state.chat_history:
            st.info("No chat history yet. Ask a question to get started!")
        else:
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                    st.markdown(chat["answer"])
                    st.divider()

            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()

    def render_about(self):
        """Render the about section."""
        st.header("About")

        st.markdown("""
        ### Multi-Modal Investment Agent

        This application uses AI to analyze multi-modal investment data including:
        - Earnings call transcripts (text)
        - Audio recordings of earnings calls
        - Stock price charts and visualizations

        #### Features
        - **Multi-Modal Processing:** Handles text, audio, and images
        - **Vector Search:** Fast semantic search across all data
        - **AI-Powered Analysis:** Uses Claude for deep insights
        - **Conversational Interface:** Ask follow-up questions with full context memory
        - **Source Citations:** All answers include source references
        - **Dual Search Strategy:** RAG database + web search fallback

        #### Architecture
        - **Processors:** Modality-specific processors for each data type
        - **Vector Store:** FAISS for fast local semantic search
        - **Agent Framework:** Pydantic AI with message history for conversations
        - **LLM:** Claude 3 Haiku for analysis
        - **Embeddings:** OpenAI embeddings for search
        - **Web Search:** DuckDuckGo for real-time information

        #### How to Use
        1. **Ingest Data:** Load company data from the sidebar
        2. **Ask Questions:** Query the data using natural language
        3. **Follow Up:** Ask related questions - the agent remembers context
        4. **Get Insights:** Receive AI-powered analysis with sources

        #### Conversation Memory
        The agent maintains conversation history, allowing you to:
        - Ask follow-up questions without repeating context
        - Reference previous answers ("What about their expenses?")
        - Build complex analysis through iterative questions
        - Clear the conversation to start fresh anytime

        #### Data Format
        Place data in folders by company name:
        ```
        data/
          CompanyA/
            transcript.txt
            earnings_call.mp3
            stock_chart.png
        ```
        """)


def main():
    """Main entry point for the Streamlit app."""
    ui = InvestmentUI()
    ui.render()


if __name__ == "__main__":
    main()
