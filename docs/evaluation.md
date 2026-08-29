# Evaluation & Reflection

## What Worked Well
- **Deterministic State Routing via LangGraph (`src/graph.py`)**: Decoupling intent classification, guardrails, retrieval, and confidence grading into isolated nodes eliminated chaotic agent loops and made failure tracing straightforward.
- **The Two-Strike Voice UX**: Balancing automated assistance with human fallback prevented frustrating conversational dead-ends while respecting voice channel constraints.
- **Robust PII & Brand Protection**: Preserving protected brand keywords while successfully masking real PII ensured high retrieval precision without compromising user privacy.
- **Adversarial Test Coverage (`tests/tests.py`)**: Rigorous test cases successfully caught and verified regression fixes for intent routing aliases, Windows thread crashes, and escalation memory persistence.

## Known limitations
- **Knowledge base size and structure.** The FAQ corpus is small, which caps what the bot can answer. A larger corpus, or ideally a proper internal knowledge graph/ontology linking related concepts, would meaningfully extend both coverage and retrieval quality for multi-hop questions.
- **Latency.** In practice, the voice pipeline takes roughly 5-6 seconds to initialize/start capturing a user's speech and a further 3-4 seconds to process and retrieve an answer once speech ends. This stacks multiple sequential LLM calls per turn (guardrail, intent, RAG, confidence, sometimes clarification) on top of STT/TTS round-trips. This is noticeably slower than a natural conversational pace and would need lighter-weight models and/or parallelized guardrail checks to be practical at real scale.
- **Confidence thresholds are reasoned, not measured** Thresholds were set based on observed behavior during testing, not tuned against a labeled evaluation set.

## Surprises During Implementation
- **STT Phonetic Drift**: Speech-to-text engines frequently merge brand terms into single tokens (e.g., `"kauflandpay"`), which initially broke vector embedding similarity matches until corpus-derived spell correction and regex normalization were implemented.
- **Checkpointer Scalar Overwrites**: Passing full state dictionaries via `_full_state()` on every turn in production accidentally wiped out multi-turn scalar counters like `failed_attempt_count`. Fixing this to pass only message payloads preserved checkpoint state across turns.
- **Windows PyTorch DLL Collisions**: Running multi-threaded test runners on Windows triggered low-level C++ memory access violations, solved by configuring thread limits and duplicate OpenMP allowances at entry points.

## Future Improvements 
- Implement mid-generation streaming TTS to reduce Time-To-First-Byte (TTFB) latency.
- Integrate lightweight webhook backends for live order status checking and authenticated user session handling.
- The models and platforms are used here because I had previous experience using them, and it would be better if more models could be tried for comparison. 