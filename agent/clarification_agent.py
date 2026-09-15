import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import SupportState
from dotenv import load_dotenv

load_dotenv()

CLARIFICATION_FALLBACK_MESSAGE = "Können Sie Ihre Frage etwas genauer formulieren?"
NO_CONTEXT_CLARIFICATION_MESSAGE = "Dazu konnte ich leider nichts finden. Können Sie Ihre Frage etwas anders formulieren?"

MAX_CLARIFICATION_ATTEMPTS = 3

# --- GLOBAL WARM-UP & LLM INITIALIZATION ---
_clarification_llm = None

def get_clarification_llm():
    """Lazily loads the Groq client and warms it up to prevent cold-start latency."""
    global _clarification_llm
    if _clarification_llm is None:
        _clarification_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=os.environ.get("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b"), # Matches RAG model fallback style
            temperature=0.3, 
        )
        print("[Clarification Agent] Warming up LLM connection...")
        try:
            # Ping the API to establish the HTTPS connection early
            _clarification_llm.invoke([HumanMessage(content="warmup ping")])
            print("[Clarification Agent] Warm-up complete.")
        except Exception as e:
            print(f"[Clarification Agent] Warm-up failed: {e}")
            
    return _clarification_llm


def clarification_node(state: SupportState) -> SupportState:
    """
    Runs when confidence_node scored the answer as 'medium', which means the agent is not confident
    enough to answer directly, but not so weak that it jumps straight
    to offering human escalation. So it asks a follow-up question instead.
    """
    print("[Clarification Agent] Asking a clarifying question...")

    messages = state.get("messages", [])
    user_msg = ""
    weak_answer = ""
    for msg in reversed(messages):
        if not weak_answer and isinstance(msg, AIMessage):
            weak_answer = getattr(msg, "content", "")
        elif weak_answer and not user_msg and isinstance(msg, HumanMessage):
            user_msg = getattr(msg, "content", "")
            break

    # Fallback slice if types are generic
    if not user_msg and len(messages) >= 2:
        user_msg = getattr(messages[-2], "content", str(messages[-2]))
    if not weak_answer and len(messages) >= 1:
        weak_answer = getattr(messages[-1], "content", str(messages[-1]))

    facts = state.get("retrieved_context", "")

    if not facts.strip():
        print("[Clarification Agent] No relevant context available, so use generic clarification.")
        return {
            "messages": [AIMessage(content=NO_CONTEXT_CLARIFICATION_MESSAGE)],
            "action": "needs_clarification",
            "pending_escalation": False,
        }

    # Fetch warmed-up LLM
    llm = get_clarification_llm()

    # forbid TTS-breaking characters
    system_prompt = SystemMessage(content="""Du bist ein hilfreicher Kaufland-Kundenservice-Assistent an einem Sprachtelefon.

Die vorherige Antwort war nicht sicher genug, um sie dem Nutzer direkt zu geben. Stelle stattdessen
EINE kurze, konkrete Rückfrage auf Deutsch, um die Anfrage des Nutzers besser zu verstehen.
Wenn die vorhandenen Informationen einen Hinweis auf das gemeinte Thema geben, erwähne diesen Vorschlag (z. B. "Meinten Sie...?").

REGELN FÜR DIE SPRACHAUSGABE (TTS):
- Antworte NUR mit der Rückfrage, absolut keine weiteren Erklärungen oder Einleitungen.
- Verwende KEINE Emojis und KEIN Markdown.
- Setze den Text NICHT in Anführungszeichen.
- Halte die Frage kurz, freundlich und natürlich sprechbar.""")

    user_prompt = HumanMessage(content=f"""
    Nutzerfrage: {user_msg}
    Vorhandener Kontext (evtl. unvollständig): {facts}
    Unsichere Antwort, die verworfen wurde: {weak_answer}
    """)

    clarification_text = CLARIFICATION_FALLBACK_MESSAGE

    for attempt in range(MAX_CLARIFICATION_ATTEMPTS):
        try:
            response = llm.invoke([system_prompt, user_prompt])
            generated_text = response.content.strip()
            
            # Strip out hallucinated quotes and markdown
            generated_text = generated_text.strip('\'"').replace("**", "").replace("*", "")
            
            # If the LLM ignored instructions and wrote a long response, fall back safely
            if len(generated_text) > 150:
                print("[Clarification Agent] Warning: LLM generated a question that is too long. Using fallback.")
                clarification_text = CLARIFICATION_FALLBACK_MESSAGE
            else:
                clarification_text = generated_text
                
            break  # Success, exit retry loop
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Catch HTTP 429 Quotas and Rate Limits
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                print(f"[Clarification Agent] Rate limit hit. Retrying in {attempt + 1}s...")
                time.sleep(1.5 * (attempt + 1))
                continue
                
            print(f"[Clarification Agent] Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_CLARIFICATION_ATTEMPTS - 1:
                time.sleep(1)
                continue
                
            print("[Clarification Agent CRITICAL] All LLM calls failed, defaulting to generic fallback.")
            clarification_text = CLARIFICATION_FALLBACK_MESSAGE

    return {
        "messages": [AIMessage(content=clarification_text)],
        "action": "needs_clarification",
        "pending_escalation": False,
    }