import logging
from groq import Groq
from src.config import Config

logger = logging.getLogger(__name__)

class RAGSynthesizer:
    def __init__(self):
        api_key = Config.GROQ_API_KEY
        self.client = Groq(api_key=api_key) if api_key and api_key != "" else None
        self.model = Config.REASONING_LLM

    def build_prompt(self, query: str, context_chunks: list, chat_history: list) -> str:
        # Format context WITH explicit sources so the AI can cite them properly
        formatted_context = ""
        for chunk in context_chunks:
            source = chunk.get("source", "Unknown Source")
            content = chunk.get("text", "No content provided.")
            formatted_context += f"[Source: {source}] {content}\n"

        # Format short chat history to maintain conversational memory
        formatted_history = "No previous conversation."
        if len(chat_history) > 1:
            history_lines = []
            for msg in chat_history[-5:-1]: 
                role = "Guest" if msg["role"] == "user" else "Concierge"
                history_lines.append(f"{role}: {msg['content']}")
            formatted_history = "\n".join(history_lines)

        prompt = f"""You are the elite AI Concierge for "Bastian at the Bottom," a luxury beach club.

<system_directives>
1. PERSONA: Be elegant, warm, and concise. Never break character.
2. DYNAMIC LANGUAGE MATCHING: Analyze the text in <current_query>. You must reply in the exact same language and script. 
3. NO INFO-DUMPING: If the guest just says "hello", reply with a warm 1-to-2 sentence greeting and ask how you can help. Do not recite the club's description unless explicitly asked.
</system_directives>

<rag_guardrails>
1. ZERO HALLUCINATION: Answer ONLY using the facts in the <knowledge_base>. You are strictly forbidden from inventing prices, menus, availability, or policies.
2. MISSING INFORMATION: If the information is not explicitly found in the <knowledge_base>, you MUST refuse to answer and use this exact template: "I couldn't find that information in our current knowledge base. Please contact our reservations team at +91 2250 333555 or resmanager@bastianhospitality.com."
3. CITATIONS REQUIRED: Every factual statement you make must be traced to the retrieved context. Include a short inline citation at the end of your sentences based on the source provided (e.g., [Source: venue_information.md]).
</rag_guardrails>

<knowledge_base>
{formatted_context if formatted_context else "No relevant facts found. You must trigger the MISSING INFORMATION refusal template."}
</knowledge_base>

<chat_history>
{formatted_history}
</chat_history>

<current_query>
{query}
</current_query>

INSTRUCTIONS: 
Generate your response now based strictly on the <current_query>, adhering 100% to the <rag_guardrails> and <system_directives>.
"""
        return prompt

    def generate_answer(self, query: str, retrieval_result: dict, chat_history: list) -> dict:
        context = retrieval_result.get("context_chunks", [])
        engine_used = retrieval_result.get("engine_used", "None")
        
        prompt = self.build_prompt(query, context, chat_history)
        
        try:
            logger.info("Sending highly structured prompt to Groq...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict, enterprise-grade luxury concierge. You follow XML tag instructions flawlessly and never hallucinate."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1, # Extremely low temperature to force strict adherence to your JSON guardrails
            )
            
            return {
                "answer": chat_completion.choices[0].message.content,
                "engine_used": engine_used
            }
            
        except Exception as e:
            import streamlit as st
            st.error(f"Groq API Error: {str(e)}")
            logger.error(f"Groq API error during generation: {e}")
            return {
                "answer": "I apologize, but I am experiencing a temporary system connection issue. Please reach out to us at +91 2250 333555 for immediate assistance.",
                "engine_used": engine_used
            }