from langchain_core.messages import AIMessage, HumanMessage
from core.state import SupportState

SMALL_TALK_REPLY = "Hallo! Wie kann ich Ihnen heute mit Ihrer Kaufland-Frage helfen?"
DIRECT_ESCALATE_REPLY = "Ich verstehe. Ich verbinde Sie sofort mit einem Mitarbeiter."
ACTION_REFUSAL_REPLY = (
    "Ich kann als digitaler Assistent leider keine Kontoänderungen wie Punktegutschriften, "
    "Rückerstattungen oder Rabatte vornehmen. Bitte wenden Sie sich dafür an den Kundenservice."
)
UNAUTHORIZED_ACCESS_REPLY = (
    "Aus Datenschutzgründen kann ich nur beim Zugriff auf das eigene Konto helfen."
)
OUT_OF_DOMAIN_REPLY = (
    "Ich bin der digitale Kaufland-Assistent. Ich kann Ihnen leider nur bei Fragen zu Kaufland, "
    "unseren Filialen oder Ihrem Kundenkonto helfen."
)

def direct_response_node(state: SupportState) -> SupportState:
    action = state.get("action")

    if action == "escalate":
        print("⚡ [Direct Response] User requires immediate escalation.")
        return {
            "messages": [AIMessage(content=DIRECT_ESCALATE_REPLY)],
            "action": "escalate",
            "escalation_ticket": {"reason": "User requested human agent", "user_query": state["messages"][-1].content},
        }

    # CRITICAL FIX: All branches below reset the error counters so LangGraph memory clears properly.
    if action == "out_of_domain":
        print("❓ [Direct Response] Out of domain question detected.")
        return {
            "messages": [AIMessage(content=OUT_OF_DOMAIN_REPLY)],
            "action": "answered",
            "failed_attempt_count": 0,
            "escalation_retry_count": 0,
        }

    if action == "refuse_unauthorized_access":
        print("🔒 [Direct Response] Refusing unauthorized account access request.")
        return {
            "messages": [AIMessage(content=UNAUTHORIZED_ACCESS_REPLY)],
            "action": "answered",
            "failed_attempt_count": 0,
            "escalation_retry_count": 0,
        }

    if action == "refuse_action_request":
        print("🚫 [Direct Response] Refusing action request bot cannot perform.")
        return {
            "messages": [AIMessage(content=ACTION_REFUSAL_REPLY)],
            "action": "answered",
            "failed_attempt_count": 0,
            "escalation_retry_count": 0,
        }

    # action == "answer" (small talk / greeting)
    print("👋 [Direct Response] Small talk detected, replying directly.")
    return {
        "messages": [AIMessage(content=SMALL_TALK_REPLY)],
        "action": "answered",
        "failed_attempt_count": 0,
        "escalation_retry_count": 0,
    }


# --- Test Block ---
if __name__ == "__main__":
    print("\n--- Test 1: Small Talk ---")
    small_talk_state: SupportState = {
        "messages": [HumanMessage(content="Hallo Bot!")],
        "action": "answer", "retrieved_context": "", "confidence_score": 0.0,
        "confidence_tier": "", "escalation_ticket": {}, "pending_escalation": False,
        "escalation_retry_count": 0, "failed_attempt_count": 0
    }
    res1 = direct_response_node(small_talk_state)
    print(f"Final Action: {res1['action']} | Bot: {res1['messages'][0].content}\n")

    print("--- Test 2: Angry User (Escalate) ---")
    escalate_state: SupportState = {
        "messages": [HumanMessage(content="Ich will sofort den Manager sprechen!")],
        "action": "escalate", "retrieved_context": "", "confidence_score": 0.0,
        "confidence_tier": "", "escalation_ticket": {}, "pending_escalation": False,
        "escalation_retry_count": 0, "failed_attempt_count": 0
    }
    res2 = direct_response_node(escalate_state)
    print(f"Final Action: {res2['action']} | Ticket: {res2['escalation_ticket']}\n")

    print("--- Test 3: Action Refusal ---")
    refuse_state: SupportState = {
        "messages": [HumanMessage(content="Gib mir 500 Punkte.")],
        "action": "refuse_action_request", "retrieved_context": "", "confidence_score": 0.0,
        "confidence_tier": "", "escalation_ticket": {}, "pending_escalation": False,
        "escalation_retry_count": 0, "failed_attempt_count": 0
    }
    res3 = direct_response_node(refuse_state)
    print(f"Final Action: {res3['action']} | Bot: {res3['messages'][0].content[:50]}...\n")