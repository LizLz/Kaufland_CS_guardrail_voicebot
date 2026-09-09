from langchain_core.messages import AIMessage, HumanMessage
from core.state import SupportState

SMALL_TALK_REPLY = "Hallo! Wie kann ich Ihnen heute mit Ihrer Kaufland-Frage helfen?"
DIRECT_ESCALATE_REPLY = "Ich verstehe. Ich verbinde Sie sofort mit einem Mitarbeiter."
OUT_OF_DOMAIN_REPLY = (
    "Ich bin der digitale Kaufland-Assistent. Ich kann Ihnen leider nur bei Fragen zu Kaufland, "
    "unseren Filialen oder Ihrem Kundenkonto helfen."
)

POLICY_REFUSAL_REPLY = (
    "Aus Datenschutz- und Sicherheitsgründen kann ich als digitaler Assistent keine direkten Kontoänderungen "
    "vornehmen oder auf fremde Daten zugreifen. Bitte wenden Sie sich dafür an unseren Kundenservice."
)


def direct_response_node(state: SupportState) -> SupportState:
    action = state.get("action")

    if action == "escalate":
        print("[Direct Response] User requires immediate escalation.")
        return {
            "messages": [AIMessage(content=DIRECT_ESCALATE_REPLY)],
            # Escalate is the ONLY one that keeps its action, because the graph needs to know to trigger the handoff
            "action": "escalate", 
            "escalation_ticket": {"reason": "User requested human agent", "user_query": state["messages"][-1].content},
        }

    
    if action == "out_of_domain":
        print("[Direct Response] Out of domain question detected.")
        return {
            "messages": [AIMessage(content=OUT_OF_DOMAIN_REPLY)],
            "failed_attempt_count": 0,
            "escalation_retry_count": 0,
        }

    if action == "policy_refusal":
        print("[Direct Response] Refusing unsupported action or data request.")
        return {
            "messages": [AIMessage(content=POLICY_REFUSAL_REPLY)],
            "failed_attempt_count": 0,
            "escalation_retry_count": 0,
        }

    if action == "small_talk":
        print("[Direct Response] Small talk detected, replying directly.")
        return {
            "messages": [AIMessage(content=SMALL_TALK_REPLY)],
            "failed_attempt_count": 0,
            "escalation_retry_count": 0,
        }

    # If a weird state somehow reaches this node, handle it safely
    print(f"[Direct Response] WARNING: Unhandled action type '{action}'. Defaulting to small talk.")
    return {
        "messages": [AIMessage(content=SMALL_TALK_REPLY)],
        "failed_attempt_count": 0,
        "escalation_retry_count": 0,
    }


# --- Test Block ---
if __name__ == "__main__":
    print("\n--- Test 1: Small Talk ---")
    small_talk_state: SupportState = {
        "messages": [HumanMessage(content="Hallo Bot!")],
        "action": "small_talk", 
        "retrieved_context": "", 
        "confidence_score": 0.0,
        "confidence_tier": "high", 
        "escalation_ticket": {}, 
        "pending_escalation": False,
        "escalation_retry_count": 0, 
        "failed_attempt_count": 0
    }
    res1 = direct_response_node(small_talk_state)
    print(f"Final Action: {res1.get('action', 'END')} | Bot: {res1['messages'][0].content}\n")

    print("--- Test 2: Angry User (Escalate) ---")
    escalate_state: SupportState = {
        "messages": [HumanMessage(content="Ich will sofort den Manager sprechen!")],
        "action": "escalate", 
        "retrieved_context": "", 
        "confidence_score": 0.0,
        "confidence_tier": "high", 
        "escalation_ticket": {}, 
        "pending_escalation": False,
        "escalation_retry_count": 0, 
        "failed_attempt_count": 0
    }
    res2 = direct_response_node(escalate_state)
    print(f"Final Action: {res2.get('action', 'END')} | Ticket: {res2['escalation_ticket']}\n")

    print("--- Test 3: Action Refusal ---")
    refuse_state: SupportState = {
        "messages": [HumanMessage(content="Gib mir 500 Punkte.")],
        "action": "policy_refusal",
        "retrieved_context": "", 
        "confidence_score": 0.0,
        "confidence_tier": "high", 
        "escalation_ticket": {}, 
        "pending_escalation": False,
        "escalation_retry_count": 0, 
        "failed_attempt_count": 0
    }
    res3 = direct_response_node(refuse_state)
    print(f"Final Action: {res3.get('action', 'END')} | Bot: {res3['messages'][0].content[:50]}...\n")