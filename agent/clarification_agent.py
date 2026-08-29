import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import SupportState
from dotenv import load_dotenv

load_dotenv()

CLARIFICATION_FALLBACK_MESSAGE = "Können Sie Ihre Frage etwas genauer formulieren?"
NO_CONTEXT_CLARIFICATION_MESSAGE = "Dazu konnte ich leider nichts finden. Können Sie Ihre Frage etwas anders formulieren?"


def clarification_node(state: SupportState) -> SupportState:
    """
    Runs when confidence_node scored the answer as 'medium' — not confident
    enough to answer outright, but not so weak that we should jump straight
    to offering human escalation. Asks a targeted follow-up question instead.
    """
    print("[Clarification Agent] Asking a clarifying question...")

    user_msg = state["messages"][-2].content
    weak_answer = state["messages"][-1].content
    facts = state.get("retrieved_context", "")

    if not facts.strip():
        print("[Clarification Agent] No relevant context available — using generic clarification.")
        return {
            "messages": [AIMessage(content=NO_CONTEXT_CLARIFICATION_MESSAGE)],
            "action": "needs_clarification",
            "pending_escalation": False,
        }

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"),
        temperature=0.3, # Perfect temperature for slightly creative follow-ups
    )

    # 1. VOICE-OPTIMIZED PROMPT: Strictly forbid TTS-breaking characters
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

    try:
        response = llm.invoke([system_prompt, user_prompt])
        clarification_text = response.content.strip()
        
        # 2. POST-PROCESSING: Strip out hallucinated quotes and markdown just in case
        clarification_text = clarification_text.strip('\'"').replace("**", "").replace("*", "")
        
        # 3. LENGTH GUARDRAIL: If the LLM ignored instructions and wrote an essay, fall back safely
        if len(clarification_text) > 150:
            print("[Clarification Agent] Warning: LLM generated a question that is too long. Using fallback.")
            clarification_text = CLARIFICATION_FALLBACK_MESSAGE
            
    except Exception as e:
        print(f"[Clarification Agent] LLM call failed: {e}")
        clarification_text = CLARIFICATION_FALLBACK_MESSAGE

    return {
        "messages": [AIMessage(content=clarification_text)],
        "action": "needs_clarification",
        "pending_escalation": False,
    }