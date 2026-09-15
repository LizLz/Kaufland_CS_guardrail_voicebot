# Kaufland Voice Support Guardrail Bot

A multi-agent, voice-enabled customer support assistant for Kaufland, built to demonstrate practical LLM guardrails.

The project focuses on **adversarial safety architecture**: PII masking, runtime prompt-injection defense, hallucination prevention, unauthorized-access refusal, and a multi-turn conversation flow designed never to dead-end users when confidence drops. Advanced retrieval (hybrid dense + lexical search with Reciprocal Rank Fusion and corpus-derived spell correction) and a low-latency voice pipeline (Deepgram STT/TTS) support this safety foundation.

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
- **PII Exposure:** Automatically anonymizes sensitive user information (names, IBANs, phone numbers) before data is transformed into embedding or reaches retrieval layers, while protecting brand-specific vocabulary like Kaufland Pay, Bluecode, Kaufland Card XTRA.

- **Prompt Injection:** Evaluates incoming utterances for jailbreak patterns and systemic prompt overrides via safety inspection layers.

- **Hallucinated / Ungrounded Answers:** Enforces strict grounding constraints on retrieved context, governed by a three-tier confidence grader and a Two-Strike Voice UX fallback rule.

- **Unauthorized Access & Action Requests:** Detects and refuses out-of-scope actions the bot cannot perform including modifying accounts, issuing refunds, crediting XTRA loyalty points, and accessing third-party passwords.

## What This Project Does Not Cover
- **Complaints or Feedback Handling:** Optimized strictly for knowledge retrieval and policy guidance, not complaint intake.
- **Account-Specific / Order-Specific Backend Data:** Operates entirely on a FAQ knowledge base without open access to live CRM database.

## Project Structure
```text
Kaufland_CS_guardrail/
├── DESIGN.md                  # Comprehensive architectural documentation & trade-offs
├── README.md                  # Project overview, setup, and navigation
├── pyproject.toml             # uv project specification & dependencies
├── uv.lock                    # Locked exact dependency tree
├── src/
│   ├── graph.py               # LangGraph state machine definition & conditional routing
│   └── chatbot.py             # Voice assistant orchestrator (LiveTranscriber, GraphProcessor, SpeechSynthesizer)
├── agent/
│   ├── guardrail_agent.py     # PII masking & safety check node
│   ├── intent_agent.py        # Pydantic-validated routing & intent classification node
│   ├── rag_agent.py           # Spell correction, hybrid retrieval (BM25 + Dense + RRF) node
│   ├── confidence_agent.py    # Three-tier confidence grading & escalation manager
│   ├── clarification_agent.py # Short, voice-optimized clarification node
│   ├── escalation_agent.py    # Escalation confirmation node
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
│   └── kaufland_faqs.csv        # Scraped FAQ data from Kaufland's public pages
├── tests/
│   └── tests.py               # Comprehensive adversarial test suite
├── examples/
│   └── demo.ipynb              # demonstration demo
└── docs/
    └── evaluation.md          # Architectural reflection, metrics, and limitations
  ```

## Setup & Installation

### Prerequisites
- **Python >= 3.11** installed on your system.
- **`uv`** package manager installed (run `pip install uv` or visit [astral.sh/uv](https://astral.sh/uv) if you haven't already).
- **`ffmpeg`** installed on your system path (provides `ffplay` for real-time text-to-speech audio playback).
  - *Mac:* `brew install ffmpeg`
  - *Windows:* `winget install ffmpeg`

### Install Dependencies
Clone the repository and run `uv` to instantly build the isolated environment:
```bash
uv sync
```

### Configure Environment Variables
Create a .env file in the project root directory:
```bash
GROQ_API_KEY=your-groq-api-key
DEEPGRAM_API_KEY=your-deepgram-api-key
```

For Windows users, PyTorch and Hugging Face models can occasionally trigger OpenMP runtime collisions (`Error #15: Initializing libiomp5md.dll`) or memory access violations. Set these environment variables in your terminal before running scripts or tests:
**Bash:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

### Build the Knowledge Base
```bash
python utility/faq_crawl.py   # Scrapes Kaufland's public FAQ pages into CSV
python utility/vector.py      # Builds the Chroma vector store from the CSV
```

## How to run
### Text Mode 
To verify graph flows and agent decisions interactively:
```bash
uv run python -m src.graph
```

### Voice Assistant Mode
To run the live voice assistant with Deepgram STT/TTS and local audio playback:
```bash
uv run python -m src.chatbot
```
Prerequisite: Requires an active microphone and `ffplay` available in your system path.

### Test the Model
Run the comprehensive adversarial test suite via `uv`:
```bash
uv run pytest tests/tests.py -v            
```
