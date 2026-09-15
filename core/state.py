from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class SupportState(TypedDict):
    # new messages are appended to the chat history, not overwriting
    messages: Annotated[list[AnyMessage], add_messages]
    
    action: Literal[
            # 1. Core Intents (From Intent Agent)
            "rag",                    # Search the knowledge base
            "escalate",               # Transfer to human
            
            # 2. Hardcoded Direct Responses (Grouped together)
            "small_talk",             # (Formerly "answer")
            "out_of_domain",          # Weather, politics, etc.
            "policy_refusal",         # (Combines unauthorized_access & action_request)
            
            # 3. System Interventions
            "blocked",                # Guardrails caught an injection
            "needs_clarification",    # RAG confidence was medium
            "awaiting_confirmation",# Low confidence, waiting for yes/no to escalate 
            "answered",             # Task successfully completed / interaction finished 
            "reroute"                 # User ignored the escalation prompt and asked a new question
        ]
    
    # The context retrieved from chromaDB for the current user query, used to provide context to the AI for generating a response.
    retrieved_context: str
    
    # A confidence score from the AI to determine if a human needs to step in
    confidence_score: float

    # Which confidence tier the score fits into
    confidence_tier: Literal["high", "medium", "low"]
    
    # If the user needs to be escalated to a human, save ticket details
    escalation_ticket: dict

    # True if the bot is waiting for the user to confirm if they want to escalate to a human
    pending_escalation: bool
    
    # Counts consecutive "I didn't understand" replies during an escalation
    # confirmation. Prevents the situation of getting permanently stuck if users
    # ignore the yes/no question and ask something else instead.
    escalation_retry_count: int

    # Counts consecutive failed attempts to answer the user's query.
    failed_attempt_count: int