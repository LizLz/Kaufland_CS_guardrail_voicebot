# Evaluation & Reflection

## What Worked Well
- **Deterministic State Routing via LangGraph (`src/graph.py`)**: Decoupling intent classification, guardrails, retrieval, and confidence grading into isolated nodes eliminated chaotic agent loops and made failure tracing straightforward.
- **The Two-Strike Voice UX**: Balancing automated assistance with human fallback prevented frustrating conversational dead-ends while respecting voice channel constraints.
- **Robust PII & Brand Protection**: Combining deterministic regex for structured financial data (IBANs) with Presidio and de_core_news_md for unstructured entities successfully masked real PII while preserving protected brand vocabulary (e.g., Kaufland Pay, Bluecode, Kaufland Card XTRA).
- **Latency Balance**: Migrating dense embeddings to intfloat/multilingual-e5-base (~278M params) and spaCy NER to de_core_news_md (~40 MB) slashed RAM and VRAM overhead by roughly 50% to 92% compared to heavier baseline models, achieving faster local inference without sacrificing German MTEB retrieval accuracy.
- **Adversarial Test Coverage (`tests/tests.py`)**: Rigorous test cases successfully caught and verified regression fixes for intent routing aliases, Windows thread crashes, and escalation memory persistence.

## Known limitations
- **Knowledge base size and structure.** The FAQ corpus is small, which caps what the bot can answer. A larger corpus, or ideally a proper internal knowledge graph/ontology linking related concepts, would meaningfully extend both coverage and retrieval quality for multi-hop questions.
- **Startup Latency.** Initializing the local application (src/chatbot.py or the test suite) incurs an 11-second cold-start delay due to PyTorch loading model weights from disk into memory, initializing the CUDA/CPU runtime, and binding to the ChromaDB SQLite backend. While a programmatic warm-up call successfully absorbs the JIT compilation penalty, this boot overhead remains a factor for local execution.
- **Confidence thresholds are reasoned, not measured** Thresholds were set based on observed behavior during testing, not tuned against a labeled evaluation set.

## Surprises During Implementation
- **STT Phonetic Drift**: Speech-to-text engines frequently merge brand terms into single tokens (e.g., `"kauflandpay"`), which initially broke vector embedding similarity matches until corpus-derived spell correction and regex normalization were implemented.
- **Checkpointer Scalar Overwrites**: Passing full state dictionaries via `_full_state()` on every turn in production accidentally wiped out multi-turn scalar counters like `failed_attempt_count`. Fixing this to pass only message payloads preserved checkpoint state across turns.
- **Windows PyTorch DLL Collisions**: Running multi-threaded test runners on Windows triggered low-level C++ memory access violations, solved by configuring thread limits and duplicate OpenMP allowances at entry points.

## Future Improvements 
- Transition the local orchestrator into an asynchronous FastAPI server using a lifespan context manager to execute the 11-second model loading sequence once at server startup, maintaining singletons in memory for concurrent client WebSockets.
- Implement mid-generation streaming TTS to reduce Time-To-First-Byte (TTFB) latency.
- Integrate lightweight webhook backends for live order status checking and authenticated user session handling.
- The models and platforms are used here because of prior familiarity or AI tool consultation exploring alternative model architectures (such as zero-shot GLiNER for PII) would provide valuable performance comparisons.