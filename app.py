import os
import re
# Suppress the harmless but annoying HuggingFace transformer warnings in the terminal
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import logging
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.synthesizer import RAGSynthesizer
from src.config import Config

# Set up basic logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Bastian Beach Club RAG", layout="wide")

@st.cache_resource
def initialize_pipeline():
    """Initialize backend components once."""
    with st.spinner("Initializing Knowledge Base, Databases, and AI Models..."):
        retriever = HybridRetriever()
        rewriter = QueryRewriter()
        synthesizer = RAGSynthesizer()
        return retriever, rewriter, synthesizer

# Initialize backend
if "retriever" not in st.session_state:
    retriever, rewriter, synthesizer = initialize_pipeline()
    st.session_state.retriever = retriever
    st.session_state.rewriter = rewriter
    st.session_state.synthesizer = synthesizer

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Bastian at the Bottom - AI Assistant")

# Sidebar configuration and RAG Engine Status
with st.sidebar:
    st.header("Pipeline Configuration")
    st.write(f"**Fast LLM (Rewriter):** {Config.FAST_LLM}")
    st.write(f"**Reasoning LLM:** {Config.REASONING_LLM}")
    st.write(f"**Top-K Retrieval:** {Config.TOP_K_RETRIEVAL}")
    
    st.divider()
    
    st.header("Live RAG Engine Status")
    # Vector Search (ChromaDB is our primary base, assumes active if app loads)
    st.success("🟢 Vector Search (ChromaDB)")
    
    # BM25 Keyword Search Check
    if getattr(st.session_state.retriever, 'bm25_index', None) is not None:
        st.success("🟢 Keyword Search (BM25)")
    else:
        st.warning("🟡 Keyword Search (BM25) - Index Empty")

    # Neo4j Graph Search Check
    if getattr(st.session_state.retriever, 'neo4j_active', False):
        st.success("🟢 Graph Search (Neo4j)")
    else:
        st.error("🔴 Graph Search (Neo4j) - Offline")

    # Cross-Encoder Check
    if getattr(st.session_state.retriever, 'reranker_active', False):
        st.success("🟢 Cross-Encoder (Bouncer)")
    else:
        st.error("🔴 Cross-Encoder - Offline")


def render_assistant_message(content: str):
    """Extracts <think> tags and displays them inside a collapsible expander."""
    think_match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
    if think_match:
        thought_process = think_match.group(1).strip()
        clean_answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        with st.expander("🧠 Show thinking..."):
            st.markdown(f"_{thought_process}_")
        
        st.markdown(clean_answer)
    else:
        st.markdown(content)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg["content"])
        else:
            st.markdown(msg["content"])
            
        if "diagnostics" in msg:
            with st.expander("View Retrieval Diagnostics"):
                st.json(msg["diagnostics"])

# Chat input
if prompt := st.chat_input("Ask about Bastian Beach Club policies, menu, or events..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Querying Hybrid Knowledge Base..."):
            
            # PHASE 1: Rewrite the query based on history
            rewritten_prompt = st.session_state.rewriter.rewrite(prompt, st.session_state.messages)
            
            # PHASE 2 & 3: Retrieve Context using the REWRITTEN query across all active engines
            retrieval_result = st.session_state.retriever.retrieve(rewritten_prompt)
            
            # PHASE 4: Synthesize Answer using the ORIGINAL query
            generation_result = st.session_state.synthesizer.generate_answer(
                prompt, 
                retrieval_result, 
                st.session_state.messages
            )
            
            answer = generation_result["answer"]
            engine_used = generation_result["engine_used"]
            
            render_assistant_message(answer)
            
            # Diagnostics to show exact database hits
            diagnostics = {
                "Original Query": prompt,
                "Rewritten Query (Sent to DB)": rewritten_prompt,
                "Engine Triggered": engine_used,
                "Chunks Retrieved": len(retrieval_result.get("context_chunks", [])),
                "Context Preview": [c["text"] for c in retrieval_result.get("context_chunks", [])][:2]
            }
            
            with st.expander("View Retrieval Diagnostics"):
                st.json(diagnostics)
                
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "diagnostics": diagnostics
    })