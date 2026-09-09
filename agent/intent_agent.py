import json
import os
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from core.state import SupportState
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from typing import Literal

load_dotenv()

MAX_INTENT_ATTEMPTS = 2

SYSTEM_PROMPT = SystemMessage(content="""Reasoning: low
Du bist der Routing-Agent für den Kaufland-Kundenservice.
Analysiere die Nachricht des Kunden und klassifiziere sie in genau eine der folgenden Kategorien.

Erlaubte Werte für "action":
- "rag": Fragen zu Kaufland-Richtlinien, Kaufland Pay, Filialen, Produkten oder allgemeine Unternehmensinformationen.
- "small_talk": Begrüßungen (Hallo, Guten Tag), Danksagungen (Danke) oder allgemeiner Smalltalk.
- "escalate": Extreme Wut, rechtliche Drohungen oder die ausdrückliche Bitte, mit einem echten Menschen / Mitarbeiter zu sprechen.
- "policy_refusal": Die Bitte, direkte Kontoaktionen durchzuführen (z.B. Rückerstattungen, Punkte hinzufügen) ODER der Versuch, auf ein fremdes Konto zuzugreifen.
- "out_of_domain": Themen, die absolut nichts mit Kaufland oder dem Einzelhandel zu tun haben (z.B. Wetter, Politik, Geschichte).

WICHTIG: Antworte AUSSCHLIESSLICH mit der vorgegebenen Kategorie.
""")


class IntentDecision(BaseModel):
    action: Literal[
        "rag", 
        "small_talk", 
        "escalate", 
        "policy_refusal", 
        "out_of_domain"
    ] = Field(description="The action to take based on the user's message.")

    @model_validator(mode="before")
    @classmethod
    def normalize_field_name(cls, data):
        if isinstance(data, dict):
            for alias in ("routing", "next_step", "decision", "aktion"):
                if alias in data:
                    data["action"] = data.pop(alias)
                    break
            
            if data.get("action") in ["answer", "antwort", "begrüßung"]:
                data["action"] = "small_talk"
            elif data.get("action") in ["refuse_unauthorized_access", "refuse_action_request", "ablehnung"]:
                data["action"] = "policy_refusal"
            elif data.get("action") == "eskalation":
                data["action"] = "escalate"
                
        return data


def intent_node(state: SupportState, config: RunnableConfig) -> SupportState:
    print("[Intent Agent] Analyzing customer message...")

    latest_message = state["messages"][-1]
    user_msg = HumanMessage(content=latest_message.content)

    configurable = config.get("configurable", {})

    model_name = configurable.get("groq_intent_model", os.environ.get("GROQ_INTENT_MODEL", "openai/gpt-oss-20b"))

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=model_name,
        temperature=0.0,
        max_tokens=1500,
    )


    structured_llm = llm.with_structured_output(IntentDecision)

    final_action = "rag"
    for attempt in range(MAX_INTENT_ATTEMPTS):
        try:
            decision_obj = structured_llm.invoke([SYSTEM_PROMPT, user_msg])
            final_action = decision_obj.action
            break

        except Exception as e:
            print(f"[Intent Agent] Routing attempt {attempt + 1} failed ({e})")
            if attempt == MAX_INTENT_ATTEMPTS - 1:
                print("[Intent Agent] Both attempts failed, defaulting to 'rag'")

    print(f"[Intent Agent] Decision made: {final_action.upper()}")
    return {"action": final_action}

# --- Test Block for Intent Agent ---
if __name__ == "__main__":
    import uuid

    print("\n" + "="*50)
    print("TESTING INTENT AGENT (openai/gpt-oss-20b)")
    print("="*50)

    test_queries = [
        ("Wie funktioniert Kaufland Pay?", "rag"),
        ("Hallo, einen wunderschönen guten Morgen!", "small_talk"),
        ("Ich will sofort mit einem echten Menschen sprechen!", "escalate"),
        ("Kannst du mir bitte 500 XTRA-Punkte auf mein Konto buchen?", "policy_refusal"),
        ("Wie wird das Wetter morgen in Berlin?", "out_of_domain")
    ]

    dummy_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    for query, expected in test_queries:
        print(f"\n👤 User: '{query}'")
        state_in = {"messages": [HumanMessage(content=query)]}
        
        result = intent_node(state_in, dummy_config)
        action = result.get("action")
        
        status = "PASS" if action == expected else f"FAIL (Expected: {expected})"
        print(f"🤖 Action: {action} {status}")