# Technical Documentation

## 1. Project Summary

This repository implements a hybrid RAG-based assistant for Bastian at the Bottom. It is built as a Streamlit application that allows users to ask natural-language questions and receive grounded answers using local knowledge, semantic retrieval, and an LLM.

The project is a practical example of a domain-specific conversational AI pipeline where the answers are constrained by a known knowledge base rather than by broad general intelligence.

## 2. System Goals

The application is designed to:

- answer guest queries about the venue and experience
- retrieve menu and policy information correctly
- maintain factual consistency through retrieval grounding
- reduce hallucination using strict refusal templates
- provide real-time retrieval diagnostics for debugging

## 3. High-Level Architecture

The runtime stack is composed of:

- Streamlit front end
- configuration and environment loader
- query rewriter module
- hybrid retrieval module
- LLM generator module
- local data store (JSON + ChromaDB)
- optional Neo4j graph database

## 4. Runtime Flow

### Request Lifecycle

When a user submits a message:

1. The app receives the input in `app.py`.
2. It calls the query rewriter with conversation history.
3. The rewritten prompt is passed to the hybrid retriever.
4. The retriever queries multiple sources:
   - vector database
   - BM25 index
   - graph database
5. Results are combined using RRF fusion.
6. A cross-encoder reranks the combined candidates.
7. The top chunks are passed to the synthesizer.
8. The synthesizer constructs a prompt that includes explicit sources.
9. Groq generates the final response.
10. The UI displays the answer and diagnostics.

## 5. Core Modules

### 5.1 `app.py`

This is the application entry point.

Responsibilities:

- configure the page title and layout
- create a cached initialization function using `st.cache_resource`
- initialize retriever, rewriter, and synthesizer once
- read and display the conversation history
- handle user prompts and generate answers
- show retrieval diagnostics via `st.expander`

Important implementation patterns:

- `initialize_pipeline()` creates the main pipeline components only once
- `st.session_state` stores the model objects and prior messages
- the sidebar reports if vector, BM25, graph, and reranker services are active

### 5.2 `src/config.py`

This file centralizes configuration.

Features:

- loads environment variables using `python-dotenv`
- reads `GROQ_API_KEY` and Neo4j credentials
- defines model names and database paths
- dynamically loads threshold values and guardrails from the knowledge JSON

Notable logic:

- `Config.load_dynamic_thresholds()` reads `retrieval_config` and `rag_guardrails`
- default fallback values are used if the JSON file is missing
- the configuration is loaded once at module import time

### 5.3 `src/retrieval/query_rewriter.py`

This component turns a raw user query into a more retrieval-friendly version.

Behavior:

- returns the original query when there is no chat history
- does not rewrite greeting-only queries
- uses Groq with a zero-temperature prompt for deterministic rewriting
- falls back to the original query if the Groq call fails

This supports conversational continuity, especially when the user’s message depends on earlier parts of the session.

### 5.4 `src/retrieval/hybrid_retriever.py`

This is the main retrieval engine.

#### Initialization

The class initializes four retrieval components:

1. ChromaDB client and collection
2. BM25 index from the in-memory document corpus
3. Neo4j connection using environment variables
4. Cross-encoder reranker model

#### Retrieval Process

The `retrieve()` method executes this pipeline:

- vector search using ChromaDB with `n_results=10`
- BM25 ranking over tokenized chunks
- graph lookup using entity words extracted from the query
- RRF fusion of vector and BM25 results
- reranking of the fused result set with the cross-encoder
- final selection of the top 4 chunks

There is a strong design choice here: the system intentionally compresses the context to a small number of chunks to keep the prompt efficient and focused.

#### RRF Fusion

The code uses reciprocal rank fusion:

- each retrieval system contributes a ranked list
- each document receives a score based on its rank in each list
- final order is based on the aggregated score

This balances semantic and keyword-based retrieval.

#### Graph Search

When Neo4j is active, graph retrieval uses entity extraction patterns and runs a Cypher query to return related entity relationships.

This is a lightweight relation-aware enhancement rather than a full knowledge graph reasoning engine.

### 5.5 `src/generation/synthesizer.py`

This module creates the final answer.

It constructs a prompt with:

- system role instructions
- dynamic user query
- retrieved context from the knowledge base
- chat history summary
- strict answer constraints

The prompt includes several explicit rules:

- no hallucination
- no invented facts
- answer using only retrieved sources
- refuse unsupported info with a fixed template
- cite the source at the end of each factual sentence

If the Groq API fails, it falls back to a polite connection error message rather than a fabricated answer.

### 5.6 `src/ingestion/document_processor.py`

This module ingests the raw knowledge base into database storage.

#### `load_rulebook()`
Reads the raw JSON file and returns a Python dictionary.

#### `ingest_to_chroma()`
Builds a vector database from hospitality content. It creates chunks from:

- venue information
- dynamic notes
- policy documents
- menus
- event records
- other structured sections

It clears the Chroma collection before reloading to avoid duplicate entries.

#### `ingest_to_neo4j()`
Creates relationships among entities such as:

- founder → venue
- architect → venue
- venue → location

This builds a simple graph structure for relation-based enrichment.

## 6. Data Model

The project uses structured JSON as the authoritative source for knowledge content.

### Main Knowledge Groups

- `assistant_rules`
- `supported_intents`
- `source_documents`
- `venue_highlights`
- `dynamic_info_notes`
- `negative_knowledge`
- `refusal_templates`
- `retrieval_config`
- `answer_validation_rules`
- `citation_rules`
- `rag_guardrails`
- `retrieval_pipeline`

### Examples of Content Included

- address and contact details
- event timing and venue capacity
- policy rules
- menu names and pricing ranges
- event calendars
- bot refusal instructions
- retrieval thresholds
- confidence rules

This makes the knowledge base a hybrid of both product domain knowledge and system governance.

## 7. Guardrails and Safety Design

The assistant is intentionally conservative.

The project enforces a policy that answers must only be based on retrieved chunks. When information is missing, the system responds with a defined refusal message instead of guessing.

Examples include:

- information not found
- low confidence answer
- unavailable live data
- unsupported reservation actions

This is a key technical feature because it helps maintain trust in the chatbot.

## 8. Configuration and Environment

The project expects a root `.env` file containing values such as:

- `GROQ_API_KEY`
- `FAST_LLM`
- `REASONING_LLM`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

The code loads these values via `dotenv` and fails gracefully if the key is missing.

## 9. Dependencies

The main library dependencies are:

- `streamlit`
- `chromadb`
- `groq`
- `neo4j`
- `python-dotenv`
- `rank-bm25`
- `sentence-transformers`

This configuration is lightweight enough for local experimentation but still includes advanced retrieval and generation components.

## 10. Strengths of the Implementation

- modular and easy to follow
- strong grounding and refusal behavior
- hybrid retrieval improves relevance
- diagnostics help verify query behavior
- dynamic thresholds from configuration JSON make the assistant easier to tune

## 11. Current Limitations

- No live operational system integration
- No real authentication or authorization
- No formal test suite included in the repo
- No deployment automation or cloud configuration
- Retrieval quality depends heavily on the quality of the source JSON

## 12. Suggested Extension Points

Potential enhancements include:

- API backend and multi-user support
- persistent user session database
- PDF ingestion from real venue documents
- production logging and telemetry
- cloud deployment on Azure or a containerized platform
- multilingual support
- admin panel for knowledge updates

## 13. Summary

This project is a domain-specific hybrid RAG chatbot that connects hospitality knowledge with modern retrieval and LLM techniques. It demonstrates how a grounded conversational assistant can be built with a control-first approach: retrieval first, generation second, and refusal before guessing.
