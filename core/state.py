from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Define data types between AI agents.
class SupportState(TypedDict):
    # new messages are appended to the chat history, not overwriting the old history.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # The current action/decision 
    action: Literal[
        "validated", "blocked",
        "rag", "answer", "escalate", "refuse_unauthorized_access", "refuse_action_request",
        "answered", "needs_clarification", "awaiting_confirmation", "reroute", "out_of_domain"
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

    failed_attempt_count: int