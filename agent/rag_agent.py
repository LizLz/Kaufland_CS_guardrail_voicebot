import os
import re
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from core.state import SupportState
from core.rag_engine import KauflandRAG
from core.hybrid_retriever import BM25Retriever, SymSpellCorrector, reciprocal_rank_fusion
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM globally so it can be warmed up
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model=os.environ.get("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b"), 
    temperature=0.1,
    max_tokens=1024,
)

_rag_components = None

def get_rag_engine():
    """Lazily loads and warms up RAG components and LLM connections."""
    global _rag_components
    if _rag_components is None:
        print("[RAG Agent] Booting up database and hybrid retriever...")
        rag = KauflandRAG()
        _docs, _metadatas = rag.get_all_documents()

        bm25_retriever = BM25Retriever(documents=_docs, metadatas=_metadatas)
        spell_corrector = SymSpellCorrector(documents=_docs)

        # --- THE COLD START WARM-UP ---
        print("[RAG Agent] Warming up PyTorch, CUDA, and LLM connections...")
        try:
            rag.retrieve_scored("warmup", k=1)
            bm25_retriever.search("warmup", top_k=1)
            spell_corrector.correct("warmup")
            
            # NEW: Ping the Groq API to prime the HTTP connection and eliminate first-turn latency
            llm.invoke([HumanMessage(content="warmup ping")])
            
            print("[RAG Agent] Warm-up complete! RAG and LLM are ready for instant responses.")
        except Exception as e:
            print(f"[RAG Agent] Warm-up failed: {e}")

        _rag_components = {
            "rag": rag,
            "bm25": bm25_retriever,
            "spell": spell_corrector
        }
    return _rag_components


def robust_langchain_invoke(model, prompt_or_messages, retries: int = 2) -> str:
    """
    Wraps LangChain model invocations with rate-limit (429) catching 
    and safe retry logic using exponential backoff.
    """
    for attempt in range(retries + 1):
        try:
            response = model.invoke(prompt_or_messages)
            if hasattr(response, "content"):
                return response.content
            return str(response)
            
        except Exception as e:
            error_str = str(e).lower()
            # Catch HTTP 429 Rate Limit or standard Groq API quotas
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                print(f"[RAG Agent] Rate limit hit. Retrying in {attempt + 1}s...")
                time.sleep(1.5 * (attempt + 1)) # Exponential backoff
                continue
            
            if attempt < retries:
                print(f"[RAG Agent] Temporary LLM error: {e}. Retrying...")
                time.sleep(1)
                continue
                
            print(f"[RAG Agent CRITICAL] LLM invocation failed permanently: {e}")
            # Fail gracefully in German for the end-user
            return "Es tut mir leid, aktuell ist unser System überlastet. Bitte versuchen Sie es gleich noch einmal."
            
    return "Entschuldigung, es gab ein technisches Problem. Bitte versuchen Sie es erneut."


FALLBACK_MESSAGE = "Dazu habe ich leider keine Information. Möchten Sie mit einem Mitarbeiter sprechen?"
DENSE_SCORE_THRESHOLD = 0.5
LEXICAL_STRONG_MATCH_THRESHOLD = 5.0


def rag_node(state: SupportState) -> SupportState:
    print("[RAG Agent] Searching for answers...")

    # Fetch the initialized components safely via the getter
    components = get_rag_engine()
    rag = components["rag"]
    bm25_retriever = components["bm25"]
    spell_corrector = components["spell"]

    # The user message is ALREADY safe and PII-masked by guardrail_node. 
    user_message = state["messages"][-1].content

    # --- 1. Retrieval ---
    corrected_query = spell_corrector.correct(user_message)
    if corrected_query != user_message.lower():
        print(f"[RAG Agent] Query corrected: '{user_message}' -> '{corrected_query}'")

    dense_docs = rag.retrieve_scored(corrected_query, k=4, score_threshold=0.0)
    lexical_docs = bm25_retriever.search(corrected_query, top_k=4)
    fused_docs = reciprocal_rank_fusion([dense_docs, lexical_docs], top_k=4)

    best_dense_score = max((d["score"] for d in dense_docs), default=0.0)
    best_lexical_score = max((d["score"] for d in lexical_docs), default=0.0)
    lexical_found_strong_match = best_lexical_score >= LEXICAL_STRONG_MATCH_THRESHOLD

    if not fused_docs or (best_dense_score < DENSE_SCORE_THRESHOLD and not lexical_found_strong_match):
        print(
            f"[RAG Agent] No confident match "
            f"(dense: {best_dense_score:.3f}, lexical: {best_lexical_score:.2f}). Using fallback."
        )
        return {
            "messages": [AIMessage(content=FALLBACK_MESSAGE)],
            "retrieved_context": "",
        }

    # --- 2. CONTEXT COMPILATION (Trusting Internal DB) ---
    safe_context = "\n\n".join([doc['content'] for doc in fused_docs])

    # --- 3. GENERATION ---
    system_prompt = SystemMessage(content=f"""Du bist ein hilfreicher Kaufland-Kundenservice-Assistent an einem Sprachtelefon.

Beantworte die LETZTE Frage des Nutzers ausschließlich auf Deutsch und nur basierend auf den Informationen zwischen den Tags <fakten> und </fakten>. 
Nutze den restlichen Chatverlauf nur, um den Kontext der aktuellen Frage zu verstehen (z.B. worauf sich "es" bezieht).

REGELN FÜR DIE SPRACHAUSGABE (TTS):
- Schreibe in natürlicher, flüssig sprechbarer Sprache, als würdest du mit jemandem telefonieren.
- Verwende KEINE Aufzählungszeichen (wie -, *, 1. 2. 3.). Verbinde Schritte stattdessen mit Wörtern wie "Zuerst", "Dann" und "Schließlich".
- Verwende KEIN Markdown (keine Sternchen **, keine Rauten ##).
- Halte die Antwort prägnant und komme direkt auf den Punkt.

WICHTIGE REGELN:
- Du hilfst dem Nutzer ausschließlich mit seinem EIGENEN Konto.
- Du kannst KEINE Kontoaktionen durchführen (Punkte gutschreiben, Rückerstattungen etc.). Behaupte niemals, dies getan zu haben. Verweise stattdessen an den Kundenservice.

<fakten>
{safe_context}
</fakten>

Wenn die Fakten die Antwort nicht enthalten, rate nicht. Antworte exakt mit: "{FALLBACK_MESSAGE}"
""")

    # --- NEW: Safely invoke the LLM using the robust wrapper ---
    safe_messages = state["messages"][:-1] + [HumanMessage(content=user_message)]
    messages_to_send = [system_prompt] + safe_messages
    
    raw_llm_response = robust_langchain_invoke(llm, messages_to_send)
    
    # Text-to-Speech cleanup logic (safe from crashing)
    clean_text = raw_llm_response.strip().replace("**", "").replace("*", "").replace("##", "")
    clean_text = re.sub(r'^\s*[-*]\s+', '', clean_text, flags=re.MULTILINE)
    
    final_message = AIMessage(content=clean_text)
    print("[RAG Agent] Answer generated.")

    return {
        "messages": [final_message],
        "retrieved_context": safe_context,
    }


# --- Test Block ---
if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment!")
    else:
        get_rag_engine()
        print("\n" + "="*50)

        test_cases = [
            "Wie funktioniert Kaufland Pay?",
            "kauflandpay",
        ]

        for query in test_cases:
            print(f"\n--- Query: '{query}' ---")
            state: SupportState = {
                "messages": [HumanMessage(content=query)],
                "action": "rag",
                "retrieved_context": "",
                "confidence_score": 0.0,
                "confidence_tier": "high",
                "escalation_ticket": {},
                "pending_escalation": False,
                "escalation_retry_count": 0,
                "failed_attempt_count": 0,
            }
            result = rag_node(state)
            print(f"AI Answer: {result['messages'][0].content[:200]}")