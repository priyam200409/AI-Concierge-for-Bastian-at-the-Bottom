# Project Overview

## Purpose

This project implements a grounded conversational assistant for Bastian at the Bottom, a beachfront hospitality venue in Mumbai. The system is designed to answer guest queries using a curated internal knowledge base instead of depending on general-purpose world knowledge alone.

The objective is to provide a reliable, elegant concierge-style experience for common guest questions about:

- venue information and contact details
- reservation and entry policies
- opening hours and timing
- menu items and pricing
- signature dishes and beverages
- seasonal and weekly events
- accessibility, parking, and family policies
- nightlife and pool access conditions

## Business Context

The project aligns with hospitality automation goals where a guest-facing assistant can reduce repetitive staff work while still requiring strict control over factual correctness. The assistant is built to behave like a premium concierge, but with constraints that prevent hallucination and enforce a human escalation path.

## Product Nature

This is not an external booking engine or a live operational dashboard. It is a knowledge-grounded chatbot that answers based on curated local data and clearly refuses unsupported questions.

That design is critical because the system must not:

- invent reservation status
- fabricate current table availability
- claim live booking or payment features
- provide information that is not explicitly present in the local knowledge base

## Core Design Goals

1. Grounded responses
   - Use retrieved context as the only source of truth.

2. High relevance
   - Use hybrid retrieval to match both exact wording and concept similarity.

3. Efficiency
   - Keep the system lightweight and responsive for interactive chat use.

4. Safety
   - Reject unsupported or low-confidence questions with a refusal template.

5. Transparency
   - Show retrieval diagnostics and source context for debugging and trust.

## Knowledge Base Model

The project stores hospitality facts in a structured JSON file:

- `data/raw/bastian_knowledge.json`

This file contains:

- rulebook and assistant instructions
- supported intents
- venue data and policy text
- menu and pricing information
- event schedules
- dynamic notes
- negative knowledge
- retrieval thresholds
- citation and validation rules

This structure allows the system to dynamically configure retrieval thresholds and operational guardrails from data instead of hard-coding all behavior in code.

## User Experience

The chatbot is presented through a Streamlit interface with:

- a title and app branding
- a live sidebar showing retrieval status
- a conversation experience similar to a hotel concierge app
- an expandable diagnostics panel showing exact query rewrite and retrieval summary

This makes the assistant practical for demos, stakeholder validation, and internal experimentation.

## Architectural Positioning

The application sits between three core layers:

- Data layer: local JSON knowledge base, ChromaDB, and optional Neo4j
- Retrieval layer: BM25, vector search, graph lookup, fusion, reranking
- Generation layer: Groq-based prompt construction and answer synthesis

The architecture is intentionally modular so the knowledge source, retrieval model, and LLM generation path can evolve independently.

## Limitations

The project is designed for controlled knowledge assistance rather than full enterprise deployment. Current limitations include:

- no live database access for bookings or inventory
- no user authentication or session persistence beyond Streamlit session state
- no multi-tenant or production-grade observability framework
- no deployment automation or container orchestration included
- dependency on local environment variables and external Groq access

## Future Potential

This codebase can be extended to:

- ingest new source documents from PDFs or websites
- add user authentication and role-based access
- connect to real booking systems
- expose a REST API backend
- deploy in a cloud environment with monitoring and CI/CD
- add multilingual support or specialized domain tuning

## Summary

The project is a focused RAG-based concierge assistant built around curated hospitality content. It demonstrates a practical implementation of hybrid retrieval, semantic grounding, conversational query rewriting, and safe answer generation in a real-world domain.
