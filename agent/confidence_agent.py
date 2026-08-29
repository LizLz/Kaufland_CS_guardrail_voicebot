import os
import re
import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from core.state import SupportState
from dotenv import load_dotenv

load_dotenv()

ESCALATION_OFFER_MESSAGE = "Ich bin mir bei dieser Antwort nicht ganz sicher. Möchten Sie mit einem Mitarbeiter sprechen? (Ja/Nein)"
ESCALATION_CONFIRMED_MESSAGE = "Alles klar, ich verbinde Sie mit einem Mitarbeiter."
ESCALATION_DECLINED_MESSAGE = "Kein Problem, lassen Sie mich wissen, falls ich sonst noch helfen kann."

YES_WORDS = ["ja", "yes", "gerne", "bitte", "ok", "okay"]
NO_WORDS = ["nein", "no", "nicht nötig", "nicht notwendig", "nicht"]

HIGH_THRESHOLD = 0.8
LOW_THRESHOLD = 0.4
MAX_FAILED_ATTEMPTS = 2


def _matches_any_word(text: str, words: list[str]) -> bool:
    """Match whole words/phrases only, not raw substrings."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def confidence_node(state: SupportState) -> SupportState:
    print("[Confidence Agent] Inspecting the generated answer...")

    user_msg = state["messages"][-2].content
    ai_msg = state["messages"][-1].content
    facts = state.get("retrieved_context", "")

    # FAST-PATH: bot already said "I don't know"
    if "Dazu habe ich leider keine Information" in ai_msg:
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

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=os.environ.get("GROQ_CONFIDENCE_MODEL", "qwen/qwen3.6-27b"),
        temperature=0.0,
    )

    system_prompt = SystemMessage(content="""You are a strict Quality Assurance Inspector for Kaufland customer service.
    Your job is to evaluate the AI Agent's response.

    Evaluate two things:
    1. Is the AI Answer entirely grounded in the Retrieved Facts without hallucinating?
    2. Does it successfully answer the User Question?

    You MUST respond in valid JSON format matching the exact schema below. Do NOT use markdown blocks.
    {
        "score": 0.0, 
        "reasoning": "A 1-sentence explanation of why this score was given."
    }
    """)

    user_prompt = HumanMessage(content=f"""
    User Question: {user_msg}
    Retrieved Facts: {facts}
    AI Answer: {ai_msg}
    """)

    final_score = 0.0
    final_reasoning = "Parsing failed entirely."

    # CRITICAL FIX: Bulletproof JSON parsing with Markdown stripping
    for attempt in range(2):
        try:
            response = llm.invoke([system_prompt, user_prompt])
            raw_text = response.content.strip()

            # Clean markdown wrappers if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            parsed_data = json.loads(raw_text)
            
            # Use float() to ensure it doesn't break if LLM outputs an integer (e.g. 1 instead of 1.0)
            final_score = float(parsed_data.get("score", 0.0))
            final_reasoning = str(parsed_data.get("reasoning", "No reasoning provided."))
            break

        except Exception as e:
            print(f"[Confidence Agent] Attempt {attempt + 1} parsing failed: {e}")
            if attempt == 1:
                print("[Confidence Agent] Fallback to escalation due to parsing failure.")
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
        print("[Confidence Agent] Medium confidence. Routing to clarification instead of escalation offer.")
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


def escalation_confirmation_node(state: SupportState) -> SupportState:
    """
    Runs ONLY when state['pending_escalation'] is True — i.e. the previous
    turn asked the user "would you like to talk to a human?" and this
    incoming message is the user's reply to that question, not a new query.
    """
    print("[Escalation Node] Checking user's response to escalation offer...")
    user_reply = state["messages"][-1].content.strip().lower()

    if _matches_any_word(user_reply, YES_WORDS):
        confirmed = True
    elif _matches_any_word(user_reply, NO_WORDS):
        confirmed = False
    else:
        print("[Escalation Node] Reply doesn't look like yes/no — rerouting as a new question.")
        
        # --- CRITICAL FIX: The Amnesia Protocol ---
        # state["messages"][-1] = The NEW question (e.g., "Was ist Kaufland Pay?"). KEEP IT.
        # state["messages"][-2] = The bot's escalation offer. DELETE IT.
        # state["messages"][-3] = The previous failed question. DELETE IT.
        messages_to_remove = []
        if len(state["messages"]) >= 3:
            messages_to_remove = [
                RemoveMessage(id=state["messages"][-2].id),
                RemoveMessage(id=state["messages"][-3].id)
            ]

        return {
            "action": "reroute",
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0, # Reset counter
            "messages": messages_to_remove # Wipe the failure from memory!
        }

    if confirmed:
        print("[Escalation Node] User confirmed. Escalating.")
        return {
            "messages": [AIMessage(content=ESCALATION_CONFIRMED_MESSAGE)],
            "action": "escalate",
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0, # Reset counter
        }
    else:
        print("[Escalation Node] User declined escalation.")
        return {
            "messages": [AIMessage(content=ESCALATION_DECLINED_MESSAGE)],
            "action": "answered",
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0, # Reset counter
            "escalation_ticket": {},
        }


# --- Test Block ---
if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found!")
    else:
        print("\n--- Test 1: Good Answer ---")
        good_state: SupportState = {
            "messages": [
                HumanMessage(content="Kann ich mit der Kaufland App bezahlen?"),
                AIMessage(content="Ja, Sie können mit Kaufland Pay bezahlen."),
            ],
            "action": "answered",
            "retrieved_context": "Kaufland Pay ist eine mobile Bezahlfunktion innerhalb der Kaufland App.",
            "confidence_score": 0.0,
            "confidence_tier": "",
            "escalation_ticket": {},
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        res_good = confidence_node(good_state)
        print(f"Action: {res_good.get('action')} | Tier: {res_good.get('confidence_tier')} | Score: {res_good.get('confidence_score')}\n")

        print("--- Test 2: Hallucination (Should ask user if they want help) ---")
        bad_state: SupportState = {
            "messages": [
                HumanMessage(content="Gibt es in der Filiale einen Geldautomaten?"),
                AIMessage(content="Ja, jede Filiale hat einen Geldautomaten im Eingangsbereich."),
            ],
            "action": "answered",
            "retrieved_context": "Wir bieten kein Bargeldabheben an Automaten an.",
            "confidence_score": 0.0,
            "confidence_tier": "",
            "escalation_ticket": {},
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        res_bad = confidence_node(bad_state)
        print(f"Action: {res_bad.get('action')} | Tier: {res_bad.get('confidence_tier')} | Pending Esc: {res_bad.get('pending_escalation')}")
        if res_bad.get("messages"):
            print(f"Bot Asks: {res_bad['messages'][0].content}\n")

        print("--- Test 3: Unclear reply during escalation offer -> should REROUTE, not loop ---")
        unclear_state: SupportState = {
            "messages": [
                AIMessage(content=ESCALATION_OFFER_MESSAGE),
                HumanMessage(content="wie ist kauflandpay"),
            ],
            "action": "awaiting_confirmation",
            "retrieved_context": "",
            "confidence_score": 0.5,
            "confidence_tier": "low",
            "escalation_ticket": {"reason": "Low AI confidence score"},
            "pending_escalation": True,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        res_unclear = escalation_confirmation_node(unclear_state)
        print(f"Action: {res_unclear.get('action')} | Pending: {res_unclear.get('pending_escalation')}")
        assert res_unclear.get("action") == "reroute", "FAIL: unclear reply was not rerouted!"
        assert res_unclear.get("pending_escalation") is False, "FAIL: escalation lock was not released!"
        print("PASS: unclear reply rerouted, user's real question can now be answered.\n")

        print("--- Test 3b: Word 'no'/'ok' embedded in an unrelated word -> should STILL reroute, not misfire ---")
        embedded_state: SupportState = {
            "messages": [
                AIMessage(content=ESCALATION_OFFER_MESSAGE),
                HumanMessage(content="Wann ist die nächste Novemberaktion?"),
            ],
            "action": "awaiting_confirmation",
            "retrieved_context": "",
            "confidence_score": 0.5,
            "confidence_tier": "low",
            "escalation_ticket": {"reason": "Low AI confidence score"},
            "pending_escalation": True,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        res_embedded = escalation_confirmation_node(embedded_state)
        assert res_embedded.get("action") == "reroute", "FAIL: 'no' inside 'November' incorrectly matched!"
        print("PASS: 'no' embedded inside 'November' did not falsely match.\n")

        print("--- Test 4: User says 'Ja' to Human Help ---")
        confirm_state: SupportState = {
            "messages": [
                AIMessage(content=ESCALATION_OFFER_MESSAGE),
                HumanMessage(content="Ja, bitte verbinden Sie mich."),
            ],
            "action": "awaiting_confirmation",
            "retrieved_context": "",
            "confidence_score": 0.5,
            "confidence_tier": "low",
            "escalation_ticket": {"reason": "Low AI confidence score"},
            "pending_escalation": True,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        res_confirm = escalation_confirmation_node(confirm_state)
        print(f"Action: {res_confirm.get('action')}")
        print(f"Bot Replies: {res_confirm['messages'][0].content}\n")

        print("--- Test 5: User says 'Nein' to Human Help ---")
        decline_state: SupportState = {
            "messages": [
                AIMessage(content=ESCALATION_OFFER_MESSAGE),
                HumanMessage(content="Nein danke, ich frage noch einmal anders."),
            ],
            "action": "awaiting_confirmation",
            "retrieved_context": "",
            "confidence_score": 0.5,
            "confidence_tier": "low",
            "escalation_ticket": {"reason": "Low AI confidence score"},
            "pending_escalation": True,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
        res_decline = escalation_confirmation_node(decline_state)
        print(f"Action: {res_decline.get('action')}")
        print(f"Bot Replies: {res_decline['messages'][0].content}\n")