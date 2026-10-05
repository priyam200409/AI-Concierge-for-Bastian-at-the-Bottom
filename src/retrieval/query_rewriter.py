import logging
from groq import Groq
from src.config import Config

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        """Initializes the Groq client for the FAST_LLM."""
        api_key = Config.GROQ_API_KEY
        self.client = Groq(api_key=api_key) if api_key and api_key != "your-groq-api-key-here" else None
        self.model = Config.FAST_LLM

    def rewrite(self, query: str, chat_history: list) -> str:
        """Analyzes history and rewrites the query for optimal database retrieval."""
        # If there is no history to provide context, just return the original query
        if not self.client or len(chat_history) < 2:
            return query
            
        # Extract the last few messages for context
        history_lines = []
        for msg in chat_history[-5:-1]: 
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        formatted_history = "\n".join(history_lines)

        prompt = f"""You are an expert search query rewriter for a Vector Database.
        
CONVERSATION HISTORY:
{formatted_history}

LATEST RAW USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the history to understand the context of the LATEST RAW USER QUERY.
2. Rewrite the latest query so it is a standalone, highly specific search term. 
3. Example: If history is discussing the "Bastian Food Menu" and the raw query is "what are the prices?", rewrite it as "What are the prices for the Bastian Food Menu?"
4. If the raw query is a simple greeting (e.g., "hello", "kya haal hai"), DO NOT rewrite it. Just output the exact same greeting.
5. ONLY output the rewritten query string. Do not add quotes, explanations, or introductory text.
"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0, # Zero creativity, just strict rewriting
            )
            rewritten_query = response.choices[0].message.content.strip()
            logger.info(f"Query Rewritten: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            return query # Fallback to original query if the API fails