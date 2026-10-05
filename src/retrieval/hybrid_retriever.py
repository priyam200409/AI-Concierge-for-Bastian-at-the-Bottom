import logging
import os
import re
import chromadb
from rank_bm25 import BM25Okapi
from neo4j import GraphDatabase
from sentence_transformers import CrossEncoder
from src.config import Config

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        """Initializes ChromaDB, BM25, Neo4j, and the Cross-Encoder Reranker."""
        # 1. Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.vector_collection = self.chroma_client.get_collection(name="bastian_knowledge")
        
        # 2. Build BM25 Index dynamically
        self.bm25_index = None
        self.raw_chunks_mapping = []
        self._initialize_bm25_index()

        # 3. Connect to Neo4j
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.neo4j_driver.verify_connectivity()
            self.neo4j_active = True
            logger.info("Retriever successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Retriever failed to connect to Neo4j: {e}")
            self.neo4j_active = False

        # 4. Initialize the Cross-Encoder Bouncer
        logger.info("Loading Cross-Encoder model (this may take a few seconds on first run)...")
        try:
            # We use a highly optimized MS-MARCO model specifically trained for search relevance
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            self.reranker_active = True
            logger.info("Cross-Encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder: {e}")
            self.reranker_active = False

    def _initialize_bm25_index(self):
        try:
            all_data = self.vector_collection.get()
            documents = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [])
            ids = all_data.get("ids", [])

            if not documents:
                return

            tokenized_corpus = []
            self.raw_chunks_mapping = []

            for doc, meta, doc_id in zip(documents, metadatas, ids):
                self.raw_chunks_mapping.append({
                    "id": doc_id,
                    "text": doc,
                    "source": meta.get("source", "Unknown Source")
                })
                tokenized_corpus.append(doc.lower().split(" "))

            self.bm25_index = BM25Okapi(tokenized_corpus)
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")

    def _tokenize(self, text: str) -> list:
        return text.lower().split(" ")

    def _reciprocal_rank_fusion(self, vector_results: list, bm25_results: list, k: int = 60) -> list:
        rrf_scores = {}

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank)

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank)

        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items]

    def _query_graph(self, query: str) -> list:
        if not self.neo4j_active:
            return []
        words = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b|\b\w{4,}\b', query)
        graph_chunks = []
        with self.neo4j_driver.session() as session:
            for word in words:
                cypher = """
                MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                WHERE a.name CONTAINS $word OR b.name CONTAINS $word
                RETURN a.name AS source, r.type AS relation, b.name AS target
                LIMIT 3
                """
                results = session.run(cypher, word=word)
                for record in results:
                    fact = f"Graph Relation: {record['source']} is linked via {record['relation']} to {record['target']}."
                    graph_chunks.append({
                        "id": f"graph_{word}",
                        "text": fact,
                        "source": "Neo4j Graph Database"
                    })
        return graph_chunks

    def retrieve(self, query: str) -> dict:
        logger.info(f"Beginning Hybrid Retrieval for query: '{query}'")
        
        # 1. Fetch Wide: Pull top 10 from Vector
        vector_chunks = []
        try:
            v_results = self.vector_collection.query(query_texts=[query], n_results=10)
            if v_results and v_results["documents"]:
                for doc, meta, doc_id in zip(v_results["documents"][0], v_results["metadatas"][0], v_results["ids"][0]):
                    vector_chunks.append({"id": doc_id, "text": doc, "source": meta.get("source", "ChromaDB")})
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # 2. Fetch Wide: Pull top 10 from BM25
        bm25_chunks = []
        if self.bm25_index and self.raw_chunks_mapping:
            tokenized_query = self._tokenize(query)
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            scored_docs = zip(doc_scores, self.raw_chunks_mapping)
            sorted_bm25 = sorted([item for item in scored_docs if item[0] > 0.0], key=lambda x: x[0], reverse=True)
            for score, doc in sorted_bm25[:10]:
                bm25_chunks.append(doc)

        # 3. Fetch Exact: Graph Search
        graph_chunks = self._query_graph(query)

        # 4. Combine into a master list via RRF
        fused_chunks = self._reciprocal_rank_fusion(vector_chunks, bm25_chunks)
        master_list = graph_chunks + fused_chunks

        # 5. THE BOUNCER: Cross-Encoder Reranking
        if self.reranker_active and master_list:
            logger.info("Applying Cross-Encoder reranking...")
            
            # Format pairs: [[query, chunk1], [query, chunk2], ...]
            pairs = [[query, chunk["text"]] for chunk in master_list]
            
            # Predict relevance scores
            scores = self.reranker.predict(pairs)
            
            # Attach scores to the chunks
            for chunk, score in zip(master_list, scores):
                chunk["relevance_score"] = float(score)
            
            # Sort the master list strictly by the new neural relevance score
            ranked_master_list = sorted(master_list, key=lambda x: x["relevance_score"], reverse=True)
            
            # Keep ONLY the absolute best 4 chunks to pass to the LLM (Context Compression)
            final_k = 4
            final_chunks = ranked_master_list[:final_k]
            engine_note = "Cross-Encoder Verified"
        else:
            final_chunks = master_list[:4]
            engine_note = "RRF Fusion Only"

        return {
            "context_chunks": final_chunks,
            "engine_used": f"{engine_note} (Filtered from V:{len(vector_chunks)}|B:{len(bm25_chunks)}|G:{len(graph_chunks)})"
        }