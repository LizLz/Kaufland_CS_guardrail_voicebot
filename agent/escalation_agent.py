import re
from langchain_core.messages import AIMessage, RemoveMessage
from core.state import SupportState

ESCALATION_CONFIRMED_MESSAGE = "Alles klar, ich verbinde Sie mit einem Mitarbeiter."
ESCALATION_DECLINED_MESSAGE = "Kein Problem, lassen Sie mich wissen, falls ich sonst noch helfen kann."

YES_WORDS = ["ja", "yes", "bitte", "ok", "okay"]
NO_WORDS = ["nein", "no", "nicht nötig", "nicht notwendig", "lieber nicht", "nein danke", "nö"]

def _matches_any_word(text: str, words: list[str]) -> bool:
    """Match whole words/phrases only, not raw substrings."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

def escalation_confirmation_node(state: SupportState) -> SupportState:
    print("[Escalation Node] Checking user's response to escalation offer...")
    
    messages = state.get("messages", [])
    if not messages:
        return state
        
    user_reply = messages[-1].content.strip().lower()

    if _matches_any_word(user_reply, YES_WORDS):
        print("[Escalation Node] User confirmed. Escalating.")
        return {
            "messages": [AIMessage(content=ESCALATION_CONFIRMED_MESSAGE)],
            "action": "escalate",
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
        }
    elif _matches_any_word(user_reply, NO_WORDS):
        print("[Escalation Node] User declined escalation.")
        return {
            "messages": [AIMessage(content=ESCALATION_DECLINED_MESSAGE)],
            "action": "answered",
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
            "escalation_ticket": {},
        }
    else:
        print("[Escalation Node] Reply doesn't look like yes/no — rerouting as a new question.")
        # If the user ignores the AI's offer and asks a new question, delete the 
        # "Do you want human help?" message and their new question, and route them back 
        # to the intent node so it processes like a brand new query.
        messages_to_remove = []
        if len(messages) >= 3:
            msg_2 = messages[-2] # The bot's escalation offer
            msg_3 = messages[-3] # The user's original failed query
            if hasattr(msg_2, "id") and msg_2.id and hasattr(msg_3, "id") and msg_3.id:
                messages_to_remove = [
                    RemoveMessage(id=msg_2.id),
                    RemoveMessage(id=msg_3.id)
                ]

        return {
            "action": "reroute",
            "pending_escalation": False,
            "escalation_retry_count": 0,
            "failed_attempt_count": 0,
            "messages": messages_to_remove,
        }