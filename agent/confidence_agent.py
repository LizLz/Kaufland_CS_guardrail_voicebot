import os
import time
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from core.state import SupportState

load_dotenv()

ESCALATION_OFFER_MESSAGE = "Ich bin mir bei dieser Antwort nicht ganz sicher. Möchten Sie mit einem Mitarbeiter sprechen? (Ja/Nein)"

HIGH_THRESHOLD = 0.8
LOW_THRESHOLD = 0.4
MAX_FAILED_ATTEMPTS = 2
MAX_CONFIDENCE_ATTEMPTS = 3

class ConfidenceDecision(BaseModel):
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="1-sentence explanation of why this score was given")

    @model_validator(mode="before")
    @classmethod
    def coerce_score(cls, data):
        if isinstance(data, dict) and "score" in data:
            try:
                data["score"] = float(data["score"])
            except (ValueError, TypeError):
                data["score"] = 0.0
        return data

# --- GLOBAL WARM-UP & LLM INITIALIZATION ---
_confidence_llm = None
_structured_llm = None

def get_confidence_llm(configurable: dict = None):
    """Lazily loads the Groq client, warms it up, and returns the structured output chain."""
    global _confidence_llm, _structured_llm
    if _confidence_llm is None:
        configurable = configurable or {}
        model_name = configurable.get("groq_confidence_model", os.environ.get("GROQ_CONFIDENCE_MODEL", "qwen/qwen3.6-27b"))
        
        _confidence_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=model_name,
            temperature=0.0,
            max_tokens=1500,  
        )
        _structured_llm = _confidence_llm.with_structured_output(ConfidenceDecision)
        
        print("[Confidence Agent] Warming up LLM connection...")
        try:
            # Ping the API to establish the HTTPS connection
            _confidence_llm.invoke([HumanMessage(content="warmup ping")])
            print("[Confidence Agent] Warm-up complete.")
        except Exception as e:
            print(f"[Confidence Agent] Warm-up failed: {e}")
            
    return _structured_llm


def confidence_node(state: SupportState, config: RunnableConfig) -> SupportState:
    print("[Confidence Agent] Inspecting the generated answer...")

    messages = state.get("messages", [])
    if len(messages) < 2:
        print("[Confidence Agent] Warning: Incomplete message history detected. Defaulting to safe clarification.")
        return {
            "confidence_score": 0.0,
            "confidence_tier": "low",
            "action": "needs_clarification",
        }

    # Extract user question and AI answer safely
    user_msg = ""
    ai_msg = ""
    for msg in reversed(messages):
        if not ai_msg and isinstance(msg, AIMessage):
            ai_msg = getattr(msg, "content", "")
        elif ai_msg and not user_msg and isinstance(msg, HumanMessage):
            user_msg = getattr(msg, "content", "")
            break

    if not user_msg or not ai_msg:
        user_msg = getattr(messages[-2], "content", str(messages[-2]))
        ai_msg = getattr(messages[-1], "content", str(messages[-1]))

    facts = state.get("retrieved_context", "")

    # Check if bot already gave up or said "I don't know"
    ai_msg_lower = ai_msg.lower()
    fallback_indicators = [
        "keine information", "liegen mir keine", "leider nicht bekannt",
        "kann ich leider nicht beantworten", "kann ich nicht beantworten", "weiß ich leider nicht",
    ]

    is_fallback = any(indicator in ai_msg_lower for indicator in fallback_indicators)

    if is_fallback:
        failed_count = state.get("failed_attempt_count", 0) + 1
        print(f"[Confidence Agent] Fallback detected (attempt {failed_count}/{MAX_FAILED_ATTEMPTS}).")

        if failed_count < MAX_FAILED_ATTEMPTS:
            return {
                "confidence_score": 0.0,
                "confidence_tier": "medium",
                "action": "needs_clarification",
                "pending_escalation": False,
                "failed_attempt_count": failed_count,
                "retrieved_context": "",
                "escalation_ticket": {"reason": "Bot fallback triggered", "user_query": user_msg},
            }
        else:
            print("[Confidence Agent] Failed attempt limit reached. Offering escalation.")
            return {
                "messages": [AIMessage(content=ESCALATION_OFFER_MESSAGE)],
                "confidence_score": 0.0,
                "confidence_tier": "low",
                "action": "awaiting_confirmation",
                "pending_escalation": True,
                "failed_attempt_count": 0,
                "retrieved_context": "",
                "escalation_ticket": {"reason": "Repeated bot fallback", "user_query": user_msg},
            }

    system_prompt = SystemMessage(content="""Reasoning: low
Du bist ein strenger Qualitätsprüfer für den Kaufland-Kundenservice.
Deine Aufgabe ist es, die Antwort des KI-Assistenten zu bewerten.

Bewerte die Antwort anhand von Faktentreue und Beantwortung der Kundenfrage.
Vergib einen Confidence Score zwischen 0.0 und 1.0:
- 0.80 bis 1.00: Korrekt, relevant und gestützt.
- 0.40 bis 0.79: Teilweise hilfreich, aber unvollständig.
- 0.00 bis 0.39: Falsch, halluziniert oder verfehlt die Frage.

Gib genau einen prägnanten Begründungssatz an.
""")

    user_prompt = HumanMessage(content=f"""
User Question: {user_msg}
Retrieved Facts: {facts}
AI Answer: {ai_msg}
""")

    configurable = config.get("configurable", {})
    structured_llm = get_confidence_llm(configurable)
    
    final_score = 0.0
    final_reasoning = "Parsing failed entirely."

    for attempt in range(MAX_CONFIDENCE_ATTEMPTS):
        try:
            decision: ConfidenceDecision = structured_llm.invoke([system_prompt, user_prompt])
            final_score = decision.score
            final_reasoning = decision.reasoning
            break
        except Exception as e:
            error_str = str(e).lower()
            
            # Catch HTTP 429 Quotas and Rate Limits
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                print(f"[Confidence Agent] Rate limit hit. Retrying in {attempt + 1}s...")
                time.sleep(1.5 * (attempt + 1))
                continue
                
            print(f"[Confidence Agent] Attempt {attempt + 1} evaluation failed: {e}")
            
            if attempt < MAX_CONFIDENCE_ATTEMPTS - 1:
                time.sleep(1)
                continue
                
            print("[Confidence Agent CRITICAL] Fallback to escalation due to repeated failure.")
            return {
                "messages": [AIMessage(content=ESCALATION_OFFER_MESSAGE)],
                "confidence_score": 0.0,
                "confidence_tier": "low",
                "action": "awaiting_confirmation",
                "pending_escalation": True,
                "escalation_retry_count": 0,
                "escalation_ticket": {"reason": "Confidence judge parsing failed", "user_query": user_msg},
            }

    print(f"[Confidence Agent] Grade: {final_score} - {final_reasoning}")

    if final_score >= HIGH_THRESHOLD:
        return {
            "confidence_score": final_score,
            "confidence_tier": "high",
            "action": "answered",
            "pending_escalation": False,
            "failed_attempt_count": 0,
        }
    elif final_score >= LOW_THRESHOLD:
        print("[Confidence Agent] Medium confidence. Routing to clarification.")
        return {
            "confidence_score": final_score,
            "confidence_tier": "medium",
            "action": "needs_clarification",
            "pending_escalation": False,
            "escalation_ticket": {
                "reason": final_reasoning,
                "score": final_score,
                "bot_attempt": ai_msg,
                "user_query": user_msg,
            },
        }
    else:
        print("[Confidence Agent] Low confidence. Asking user if they want human help.")
        return {
            "messages": [AIMessage(content=ESCALATION_OFFER_MESSAGE)],
            "confidence_score": final_score,
            "confidence_tier": "low",
            "action": "awaiting_confirmation",
            "pending_escalation": True,
            "escalation_retry_count": 0,
            "escalation_ticket": {
                "reason": final_reasoning,
                "score": final_score,
                "bot_attempt": ai_msg,
                "user_query": user_msg,
            },
        }

# --- Test Block for Confidence Agent ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING CONFIDENCE AGENT")
    print("="*50)

    dummy_config = {"configurable": {}}

    print("\n--- Test 1: High Confidence (Perfect match with context) ---")
    state_high = {
        "messages": [
            HumanMessage(content="Wie lange kann ich Artikel zurückgeben?"),
            AIMessage(content="Sie können Artikel innerhalb von 90 Tagen mit Kassenbon zurückgeben.")
        ],
        "retrieved_context": "Kaufland Rückgaberichtlinien: Kunden haben ein 90-tägiges Rückgaberecht, sofern der originale Kassenbon vorliegt."
    }
    res_high = confidence_node(state_high, dummy_config)
    print(f"Action: {res_high.get('action')} | Tier: {res_high.get('confidence_tier')} | Score: {res_high.get('confidence_score')}")

    print("\n--- Test 2: Low Confidence (Hallucinated detail not in context) ---")
    state_low = {
        "messages": [
            HumanMessage(content="Gibt es in der Filiale Berlin Mitte einen Geldautomaten?"),
            AIMessage(content="Ja, dort gibt es einen Sparkassen-Geldautomaten direkt am Eingang.")
        ],
        "retrieved_context": "Filiale Berlin Mitte: Öffnungszeiten 07:00 - 22:00 Uhr. (Keine Informationen zu Geldautomaten im Dokument)."
    }
    res_low = confidence_node(state_low, dummy_config)
    print(f"Action: {res_low.get('action')} | Tier: {res_low.get('confidence_tier')} | Score: {res_low.get('confidence_score')}")