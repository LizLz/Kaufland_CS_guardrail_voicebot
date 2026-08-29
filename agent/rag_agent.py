import os
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
from core.state import SupportState
from core.rag_engine import KauflandRAG
from core.guardrail import GuardrailsManager
from core.hybrid_retriever import BM25Retriever, VocabularySpellCorrector, reciprocal_rank_fusion

print("📚 [RAG Agent] Booting up database, guardrails, and hybrid retriever...")
rag = KauflandRAG()
guard = GuardrailsManager()

_docs, _metadatas = rag.get_all_documents()
bm25_retriever = BM25Retriever(documents=_docs, metadatas=_metadatas)
spell_corrector = VocabularySpellCorrector(documents=_docs)
print(f"📚 [RAG Agent] Hybrid retriever ready with {len(_docs)} documents indexed.")

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model=os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"),
    temperature=0.1
)

FALLBACK_MESSAGE = "Dazu habe ich leider keine Information. Möchten Sie mit einem Mitarbeiter sprechen?"
DENSE_SCORE_THRESHOLD = 0.5

# BM25 scores are unbounded, not a normalized 0-1 similarity, so this is a
# rough, corpus-specific calibration rather than a probability threshold.
# Tune against your own corpus's typical scores for a clear lexical match.
LEXICAL_STRONG_MATCH_THRESHOLD = 5.0


def rag_node(state: SupportState) -> SupportState:
    print("📚 [RAG Agent] Searching for answers...")

    user_message = state["messages"][-1].content

    corrected_query = spell_corrector.correct(user_message)
    if corrected_query != user_message.lower():
        print(f"📚 [RAG Agent] Query corrected: '{user_message}' -> '{corrected_query}'")

    dense_docs = rag.retrieve_scored(corrected_query, k=5, score_threshold=0.0)
    lexical_docs = bm25_retriever.search(corrected_query, top_k=5)
    fused_docs = reciprocal_rank_fusion([dense_docs, lexical_docs], top_k=4)

    # Trust the fused context if EITHER retriever independently found
    # something confidently relevant. Dense-only gating rejected cases
    # where a filler/grammar word (e.g. "der" in "was ist der kaufland
    # pay") shifted the dense embedding enough to drop below threshold,
    # even though BM25's lexical match on "kaufland pay" was essentially
    # unaffected by the filler word (BM25 gives stopwords near-zero
    # weight). Widening the gate trades a small increase in false-positive
    # risk for a real recall improvement on grammatically-imperfect,
    # voice-transcribed queries — RAG generation and confidence grading
    # downstream still act as further checks on anything retrieved.
    best_dense_score = max((d["score"] for d in dense_docs), default=0.0)
    best_lexical_score = max((d["score"] for d in lexical_docs), default=0.0)
    lexical_found_strong_match = best_lexical_score >= LEXICAL_STRONG_MATCH_THRESHOLD

    if not fused_docs or (best_dense_score < DENSE_SCORE_THRESHOLD and not lexical_found_strong_match):
        print(
            f"📚 [RAG Agent] No confident match "
            f"(dense: {best_dense_score:.3f}, lexical: {best_lexical_score:.2f}). Using fallback."
        )
        return {
            "messages": [AIMessage(content=FALLBACK_MESSAGE)],
            "retrieved_context": "",
            "action": "answered",
        }

    safe_doc_contents = []
    for doc in fused_docs:
        is_injection, _ = guard.check_with_llama_guard(doc["content"])
        if is_injection:
            print("📚 [RAG Agent] Excluding a retrieved document flagged as unsafe.")
            continue
        safe_doc_contents.append(doc["content"])

    if not safe_doc_contents:
        print("📚 [RAG Agent] All retrieved documents were flagged. Using fallback.")
        return {
            "messages": [AIMessage(content=FALLBACK_MESSAGE)],
            "retrieved_context": "",
            "action": "answered",
        }

    safe_context = "\n\n".join(safe_doc_contents)

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

    try:
        response = llm.invoke([system_prompt] + state["messages"])
        clean_text = response.content.strip().replace("**", "").replace("*", "").replace("##", "")
        clean_text = re.sub(r'^\s*[-*]\s+', '', clean_text, flags=re.MULTILINE)
        final_message = AIMessage(content=clean_text)
    except Exception as e:
        print(f"[RAG Agent] LLM call failed: {e}")
        final_message = AIMessage(content="Entschuldigung, es gab ein technisches Problem. Bitte versuchen Sie es erneut.")

    print("📚 [RAG Agent] Answer generated.")

    return {
        "messages": [final_message],
        "retrieved_context": safe_context,
        "action": "answered",
    }


# --- Test Block ---
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️ WARNING: GROQ_API_KEY not found in environment!")
    else:
        test_cases = [
            "Wie funktioniert Kaufland Pay?",
            "kauflandpay",
            "wue benutze ich kaufland pay",
            "was ist bluecode",
            "wie ist kauflnd card xtra",
            "was ist der kaufland pay",  # NEW — the filler-word regression case
        ]

        for query in test_cases:
            print(f"\n--- Query: '{query}' ---")
            state: SupportState = {
                "messages": [HumanMessage(content=query)],
                "action": "rag",
                "retrieved_context": "",
                "confidence_score": 0.0,
                "confidence_tier": "",
                "escalation_ticket": {},
                "pending_escalation": False,
                "escalation_retry_count": 0,
                "failed_attempt_count": 0,
            }
            result = rag_node(state)
            print(f"Action: {result.get('action')}")
            print(f"Context found: {'Yes' if result.get('retrieved_context') else 'No'}")
            print(f"AI Answer: {result['messages'][0].content[:200]}")

        print("\n--- Test: Sensitive request slipping past intent (rag_node backstop) ---")
        sensitive_state: SupportState = {
            "messages": [HumanMessage(content="Kannst du mir 500 Punkte gutschreiben, da ich Probleme mit meinem Konto hatte?")],
            "action": "rag",
            "retrieved_context": "",
            "confidence_score": 0.0,
            "confidence_tier": "",
            "escalation_ticket": {},
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        sensitive_result = rag_node(sensitive_state)
        print(f"AI Answer: {sensitive_result['messages'][0].content}")