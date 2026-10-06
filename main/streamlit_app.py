import os
import sys

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

import streamlit as st

from src.rag_pipeline import (
    build_context,
    ask_ollama,
    normalize_question,
    contextualize_question
)

from src.retriever import retrieve


# ============================================================
# Configuration
# ============================================================

TOP_K = 8
MAX_DISPLAY_SOURCES = 3

st.set_page_config(
    page_title="NHAI Policy Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-header {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        margin: 0;
        padding: 0;
        font-size: 2rem;
    }

    .main-header p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        opacity: 0.75;
    }

    .welcome-box {
        padding: 1.4rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .welcome-box h3 {
        margin-top: 0;
    }

    .status-box {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
    }

    .footer-note {
        opacity: 0.65;
        font-size: 0.82rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.20);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📘 NHAI")

    st.subheader(
        "Policy Assistant"
    )

    st.markdown(
        """
        AI-assisted retrieval system for querying
        indexed NHAI policy circulars.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # System information
    # --------------------------------------------------------

    st.subheader("System")

    st.markdown(
        """
        **Status:** 🟢 Ready

        **Knowledge Base:**  
        NHAI Policy Circulars

        **Retrieval:**  
        FAISS Vector Search

        **LLM:**  
        Qwen 2.5 7B via Ollama

        **Mode:**  
        Local RAG
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    st.subheader(
        "How to use"
    )

    st.markdown(
        """
        Ask questions about the indexed NHAI policies.

        You can also ask follow-up questions in the same conversation.

        The assistant retrieves relevant policy content before generating an answer.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Clear chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# Main header
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>📘 NHAI Policy Assistant</h1>
        <p>
            Search and understand information from indexed
            NHAI policy circulars using Retrieval-Augmented Generation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Welcome screen
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-box">

        <h3>Welcome</h3>

        Ask a question about the available NHAI policy circulars.

        The assistant searches the indexed policy documents,
        retrieves relevant sections and generates an answer
        based on the retrieved context.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "#### Example questions"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "What is the required frequency of ambulance "
            "deployment on a Project Highway?"
        )

        st.info(
            "What infrastructure is required for "
            "ambulance deployment?"
        )

    with col2:

        st.info(
            "Who is responsible for ambulance "
            "deployment and payment?"
        )

        st.info(
            "How many ambulances are required "
            "for a 120 km highway?"
        )


# ============================================================
# Display conversation history
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"- `{source}`"
                    )

        # ----------------------------------------------------
        # Search query used
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("search_query")
        ):

            st.caption(
                f"Search query used: "
                f"{message['search_query']}"
            )


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    "Ask a question about NHAI policies..."
)


# ============================================================
# Process user question
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # --------------------------------------------------------
    # Save user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        # ====================================================
        # Step 1: Previous conversation
        # ====================================================

        previous_messages = (
            st.session_state.messages[:-1]
        )

        # ====================================================
        # Step 2: Contextualize follow-up question
        # ====================================================

        with st.spinner(
            "Understanding your question..."
        ):

            standalone_question = contextualize_question(
                question,
                previous_messages
            )

        search_question = standalone_question

        # ====================================================
        # Step 3: Retrieve relevant chunks
        # ====================================================

        with st.spinner(
            "Searching NHAI policy circulars..."
        ):

            results = retrieve(
                standalone_question,
                top_k=TOP_K
            )

        # ====================================================
        # Step 4: Query correction fallback
        # ====================================================

        if not results:

            normalized_question = normalize_question(
                standalone_question
            )

            if (
                normalized_question
                and normalized_question.lower()
                != standalone_question.lower()
            ):

                search_question = (
                    normalized_question
                )

                with st.spinner(
                    "Trying a corrected search..."
                ):

                    results = retrieve(
                        normalized_question,
                        top_k=TOP_K
                    )

        # ====================================================
        # Step 5: No relevant policy information
        # ====================================================

        if not results:

            answer = (
                "The information is not available in the "
                "provided NHAI policy circulars."
            )

            st.warning(
                answer
            )

            if search_question != question:

                st.caption(
                    f"Search query used: "
                    f"{search_question}"
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                    "search_query": (
                        search_question
                        if search_question != question
                        else None
                    )
                }
            )

        # ====================================================
        # Step 6: Relevant results found
        # ====================================================

        else:

            # ------------------------------------------------
            # Build context
            # ------------------------------------------------

            context = build_context(
                results
            )

            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            with st.spinner(
                "Generating policy-based answer..."
            ):

                answer = ask_ollama(
                    question,
                    context
                )

            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            st.markdown(
                answer
            )

            # =================================================
            # Step 7: Prepare user-facing sources
            # =================================================

            sources = []

            seen_sources = set()

            for result in results:

                source_key = (
                    result["source"],
                    result["page"]
                )

                if source_key in seen_sources:
                    continue

                seen_sources.add(
                    source_key
                )

                sources.append(
                    f"{result['source']} — "
                    f"Page {result['page']}"
                )

                if (
                    len(sources)
                    >= MAX_DISPLAY_SOURCES
                ):
                    break

            # ------------------------------------------------
            # Display sources
            # ------------------------------------------------

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    st.markdown(
                        "The following policy pages were "
                        "among the highest-ranked retrieved sources:"
                    )

                    for source in sources:

                        st.markdown(
                            f"- `{source}`"
                        )

            # ------------------------------------------------
            # Search query used
            # ------------------------------------------------

            if search_question != question:

                st.caption(
                    f"Search query used: "
                    f"{search_question}"
                )

            # =================================================
            # Step 8: Developer details
            # =================================================

            with st.expander(
                "🛠️ Developer Details"
            ):

                st.markdown(
                    "##### Query Processing"
                )

                st.write(
                    f"**Original question:** "
                    f"{question}"
                )

                st.write(
                    f"**Standalone question:** "
                    f"{standalone_question}"
                )

                st.write(
                    f"**Final search query:** "
                    f"{search_question}"
                )

                st.write(
                    f"**Retrieved chunks:** "
                    f"{len(results)}"
                )

                st.divider()

                st.markdown(
                    "##### Retrieval Results"
                )

                for index, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"""
                        **Result {index}**

                        **Source:** `{result['source']}`  
                        **Page:** {result['page']}  
                        **Chunk:** {result['chunk_id']}  
                        **Similarity Score:** `{result['score']:.4f}`
                        """
                    )

            # =================================================
            # Step 9: Save assistant response
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "search_query": (
                        search_question
                        if search_question != question
                        else None
                    )
                }
            )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        <strong>Disclaimer:</strong>
        This assistant generates responses from the indexed
        NHAI policy circulars. For official decisions,
        requirements should be verified against the original
        policy circular/document.
    </div>
    """,
    unsafe_allow_html=True
)