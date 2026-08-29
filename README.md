# Kaufland Voice Support Guardrail Bot

A production-grade, multi-agent, voice-enabled customer support assistant for Kaufland (grocery retail, loyalty program, and in-app payments), built to demonstrate practical LLM guardrails in a domain with real financial and account-related stakes — going far beyond a generic FAQ chatbot.

The core deliverable of this project is **adversarial safety architecture**: robust PII masking, runtime prompt-injection defense, hallucination prevention, unauthorized-access refusal, and a multi-turn conversation flow designed never to dead-end users when confidence drops. Advanced retrieval (hybrid dense + lexical search with Reciprocal Rank Fusion and corpus-derived spell correction) and a low-latency voice pipeline (Deepgram STT/TTS) support this safety foundation.

- **Architecture & Trade-offs:** See [`DESIGN.md`](./DESIGN.md)
- **Evaluation & Reflection:** See [`docs/evaluation.md`](./docs/evaluation.md)

---

## Pipeline Architecture

The workflow is managed as a state machine (`SupportState`) via LangGraph with short-term memory (`MemorySaver`). Every turn evaluates conditional routing rules based on state flags (`pending_escalation`, `action`, etc.):

```text
                       [ Incoming User Message ]
                                   │
                                   ▼
                   { pending_escalation == True? }
                      ├── Yes ──► [ Escalation Confirmation Node ]
                      │                       │
                      └── No ──► [ Guardrail Node ] ──(Blocked)──► [ END ]
                                              │ (Safe)
                                              ▼
                                        [ Intent Agent ]
                                      /        |        \
                           (RAG Query)    (Small Talk)  (Refusal/OOD)
                                   │           │              │
                                   ▼           └──────┬───────┘
                            [ RAG Agent ]             │
                          (Spell Correction +         │
                            Dense + BM25 +            │
                                 RRF)                 │
                                   │                  │
                                   ▼                  ▼
                          [ Confidence Agent ] [ Direct Response ]
                           /       │        \         │
                       (High)   (Med)      (Low)      │
                         │       │           │        │
                         │       ▼           ▼        │
                         │ [Clarification] [Escalate /│
                         │  Agent]          Fallback] │
                         │       │           │        │
                         └───────┴─────┬─────]        │
                                       │              │
                                       ▼              ▼
                                    [ END ] ◄─────────┘
```

## What This Project Addresses 
- **PII Exposure:** Automatically sanitizes sensitive user attributes (names, IBANs, phone numbers) before data hits embedding or retrieval layers, while protecting brand-specific vocabulary (e.g., Kaufland Pay, Bluecode, Kaufland Card XTRA) from false-positive named-entity masking.

- **Prompt Injection:** Evaluates incoming utterances for jailbreak patterns and systemic prompt overrides via safety inspection layers.

- **Hallucinated / Ungrounded Answers:** Enforces strict grounding constraints on retrieved context, governed by a three-tier confidence grader and a Two-Strike Voice UX fallback rule.

- **Unauthorized Access & Action Requests:** Detects and refuses out-of-scope actions the bot cannot perform (e.g., modifying accounts, issuing refunds, crediting XTRA loyalty points, or accessing third-party passwords).

## What This Project Does Not Cover
- **Complaints or Feedback Handling:** Optimized strictly for knowledge retrieval and policy guidance, not complaint intake.
- **Account-Specific / Order-Specific Backend Data:** Operates entirely on a curated FAQ knowledge base without live CRM database integration.

## Project Structure
```text
Kaufland_CS_guardrail/
├── DESIGN.md                  # Comprehensive architectural documentation & trade-offs
├── README.md                  # Project overview, setup, and navigation
├── requirements.txt           # Python dependency manifests
├── src/
│   ├── graph.py               # LangGraph state machine definition & conditional routing
│   └── chatbot.py             # Voice assistant orchestrator (LiveTranscriber, GraphProcessor, SpeechSynthesizer)
├── agent/
│   ├── guardrail_agent.py     # PII masking & safety check node
│   ├── intent_agent.py        # Pydantic-validated routing & intent classification node
│   ├── rag_agent.py           # Spell correction, hybrid retrieval (BM25 + Dense + RRF) node
│   ├── confidence_agent.py    # Three-tier confidence grading & escalation manager
│   ├── clarification_agent.py # Short, voice-optimized clarification node
│   └── direct_response_agent.py # Small talk, out-of-domain, and refusal responses
├── core/
│   ├── state.py               # TypedDict SupportState definition
│   ├── guardrail.py           # GuardrailsManager (Presidio/regex PII masking & safety filters)
│   ├── rag_engine.py          # Chroma vector store loader & dense retriever
│   └── hybrid_retriever.py    # BM25 lexical search + corpus spell correction + RRF fusion
├── utility/
│   ├── faq_crawl.py           # Web scraper for Kaufland FAQs
│   ├── vector.py              # Embedding generation & Chroma indexing
│   ├── audio.py               # Microphone capturing live sound
│   └── logger.py              # Log management & visualization
├── data/                      # Scraped CSV & Chroma database artifacts
├── tests/
│   └── tests.py               # Comprehensive adversarial test suite
├── examples/
│   └── demo.ipynb              # demonstration demo
└── docs/
    └── evaluation.md          # Architectural reflection, metrics, and limitations
  ```

## Setup & Installation
### Install Dependencies
`pip install -r requirements.txt`

### Configure Environment Variables
Create a .env file in the project root directory:
```bash
GROQ_API_KEY=your-groq-api-key
DEEPGRAM_API_KEY=your-deepgram-api-key
```

### Build the Knowledge Base
```bash
python utility/faq_crawl.py   # Scrapes Kaufland's public FAQ pages into CSV
python utility/vector.py      # Builds the Chroma vector store from the CSV
```

## How to run
### Text Mode 
To verify graph flows and agent decisions interactively or via scripts:
`python src/graph.py`

### Voice Assistant Mode
To run the live voice assistant with Deepgram STT/TTS and local audio playback:
`python src/chatbot.py`
Prerequisite: Requires an active microphone and ffplay (bundled with ffmpeg) installed and added to your system PATH for audio streaming playback.

### Test the Model
The project includes an adversarial test suite designed to validate failure modes, regression fixes, and state memory persistence:
```bash
pytest tests/tests.py -v               # Full test suite (requires GROQ_API_KEY)
pytest tests/tests.py -m "not integration" # Fast unit tests only
```
