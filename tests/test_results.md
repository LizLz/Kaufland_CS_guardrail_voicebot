# Test Results

## How to reproduce

```bash
pytest tests/test_cases.py -v                    # full suite (requires GROQ_API_KEY)
pytest tests/test_cases.py -m "not integration"  # fast subset, no API key needed
```

## Results by module

| Module | Tests | Status | Notes |
|---|---|---|---|
| `TestGuardrails` | 5 | ✅ pass | Includes the brand-term masking regression test |
| `TestIntentAgent` | 5 | ✅ pass | Covers all 5 routing categories including both refusal types |
| `TestHybridRetriever` | 4 | ✅ pass | No API key required, pure retrieval/correction logic |
| `TestConfidenceAgentUnit` | 3 | ✅ pass | No API key required, the escalation reroute regression test |
| `TestConfidenceAgentIntegration` | 2 | ✅ pass | |
| `TestRagAgent` | 3 | ✅ pass | Includes the direct-call backstop test bypassing intent classification |
| `TestGraphEndToEnd` | 6 | ✅ pass | Full pipeline, including the escalation-reroute regression |

