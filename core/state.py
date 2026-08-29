from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SupportState(TypedDict):
    # new messages are appended to the chat history, not overwriting
    messages: Annotated[list[BaseMessage], add_messages]
    
    action: Literal[
    # --- Guardrail Agent Actions ---
    "validated",              # Input or retrieved context passed all safety checks successfully
    "blocked",                # Guardrails caught a prompt injection or safety violation

    # --- Intent Agent Router Actions ---
    "rag",                    # Query requires factual knowledge base search (routes to rag_node)
    "answer",                 # Query is small talk or a conversational greeting (routes to direct_response_node)
    "escalate",               # System is transferring the user to human support
    "refuse_unauthorized_access", # Attempt to access/reset another user's account data (refused safely)
    "refuse_action_request",  # Request for an unsupported action like crediting points/refunds (refused safely)
    "out_of_domain",          # Query is completely unrelated to Kaufland (weather, politics...)

    # --- Confidence & Fallback Actions ---
    "answered",               # High confidence retrieval
    "needs_clarification",    # Medium confidence retrieval and triggers a voice-optimized clarification question
    "awaiting_confirmation",  # Low confidence or max failure threshold reached and triggers the Two-Strike human support offer
    "reroute"                 # User replied to the escalation offer with a new question instead of Yes/No
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