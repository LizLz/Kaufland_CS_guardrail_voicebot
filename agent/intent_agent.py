import json
import os
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from core.state import SupportState
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from typing import Literal

load_dotenv()


class IntentDecision(BaseModel):
    action: Literal["rag", "answer", "escalate", "refuse_unauthorized_access", "refuse_action_request", "out_of_domain"] = Field(
        description="The action to take based on the user's message."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_field_name(cls, data):
        """Accepts alternative action names from the LLM just in case."""
        if isinstance(data, dict) and "action" not in data:
            for alias in ("routing", "next_step", "decision"):
                if alias in data:
                    data["action"] = data.pop(alias)
                    break
        return data


def intent_node(state: SupportState) -> SupportState:
    print("[Intent Agent] Analyzing customer message...")

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=os.environ.get("GROQ_INTENT_MODEL", "qwen/qwen3.6-27b"),
        temperature=0.0,
    )

    system_prompt = SystemMessage(content="""You are the routing agent for Kaufland customer support.
Analyze the customer's message and output a raw JSON object with a single key "action". 
Allowed values for "action":
- "rag": questions about Kaufland policies, Kaufland Pay, stores, general company info.
- "answer": greetings, thank you, or small talk.
- "escalate": extreme anger, legal threats, or explicitly asking for a human agent.
- "refuse_unauthorized_access": asking how to access someone else's account.
- "refuse_action_request": asking you to directly perform an account action.
- "out_of_domain": unrelated to Kaufland or retail.

Output format must be strictly JSON, like this:
{"action": "rag"}
""")

    latest_message = state["messages"][-1]
    user_msg = HumanMessage(content=latest_message.content)

    final_action = "rag"
    for attempt in range(2):
        try:
            # 1. Text generation (use json wrapper to avoid json_mode 400 errors)
            response = llm.invoke([system_prompt, user_msg])
            raw_text = response.content.strip()

            # 2. Regex extraction 
            json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON object found in LLM output: {raw_text}")

            parsed_data = json.loads(json_match.group(0))

            # 3. Pydantic validation & normalization
            decision = IntentDecision.model_validate(parsed_data)
            final_action = decision.action
            break

        except Exception as e:
            print(f"[Intent Agent] Routing attempt {attempt + 1} failed ({e})")
            if attempt == 1:
                print("[Intent Agent] Both attempts failed, defaulting to 'rag'")

    print(f"[Intent Agent] Decision made: {final_action.upper()}")
    return {"action": final_action}