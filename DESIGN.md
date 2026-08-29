# Design Decisions

## Why Kaufland

Kaufland's app bundles a loyalty system (points redeemable for goods), an account system, and an in-app payment system (Kaufland Pay, built on Bluecode, with optional bank-card linking for QR-code payment). That combination gives the app real financial-app characteristics inside a grocery app, which makes it a meaningful testbed for guardrails: users have genuine questions worth a voice interface (easier than scrolling a long FAQ or waiting on hold), but the same account/payment surface is exactly what an adversarial user might try to exploit.

## Failure modes the guardrails target

**1. PII exposure.** User input is masked (names, IBANs) before it reaches any LLM call using Presidio + spaCy German NER for names/entities, alongside a regex layer for structured PII (IBAN) that NER alone does not reliably catch.

**2. Prompt injection.** A dedicated classifier (Llama Guard, `meta-llama/llama-prompt-guard-2-86m`) checks both user input and retrieved document content, with a lightweight keyword pre-filter as a fallback when the classifier API is unreachable.

**3. Hallucinated / ungrounded answers.** RAG generation is constrained to retrieved `<fakten>` context only, with an explicit instruction that retrieved content is reference material and never contains instructions to the model. A separate confidence agent independently grades whether the generated answer is actually supported by that context.

**4. Fabricated action-completion.** A distinct failure mode from hallucinated facts: the bot has no backend integration to actually credit loyalty points, issue refunds, or modify an account. Both the intent classifier and a backstop instruction in the RAG system prompt explicitly refuse these requests and redirect to real customer service, rather than letting the model generate a false confirmation ("Ich habe Ihnen 500 Punkte gutgeschrieben").

**5. Unauthorized account access.** Requests to access, reset, or obtain someone else's password or account details are a separate refusal category from action requests. It is detected at intent-classification time and redirected, rather than treated as an ordinary how-to question.

## Guardrail impact — before / after

| Failure mode | Before | After |
|---|---|---|
| Brand-name masking | NER misclassified "kauflandpay" as `<PERSON>`/`<ORGANIZATION>`, corrupting the query before retrieval | `BRAND_TERM` pattern recognizer + span-overlap filtering exempts known brand terms from masking |
| Prompt injection | Addressed from the start with a real classifier, not a keyword list | Injection attempts blocked at `guardrail_node`, before intent classification or any generation |
| Fabricated actions | Would have been silently answered by an unconstrained RAG prompt with a plausible false confirmation | Refused at two independent layers (intent + RAG backstop), verified by calling the RAG node directly, bypassing intent classification, to confirm the backstop alone still refuses |
| Typo/merged brand names in retrieval | Dense-only retrieval matched "kauflandpay" to the wrong product entirely | Hybrid dense+BM25 retrieval with corpus-derived fuzzy correction retrieves the correct document |
| Escalation handling | Binary confidence meant any uncertain case jumped straight to "want a human?" | Three-tier confidence with a genuine clarification path and Two-Strike fallback added |

## The reroute/escalation problem # State Memory

Initially, a simple mechanism was planned to offer escalation ("Möchten Sie mit einem Mitarbeiter sprechen?") on any low-confidence answer. In practice, once that offer was made, any reply that wasn't a clean yes/no got force-parsed as an unclear yes/no answer. After a retry limit, the escalation lock was released, but the user's actual question was discarded and never answered.

To resolve this, I tried below mechanisms:

1. **The Two-Strike Voice UX Rule & Medium Confidence Tier:** Most failures are because of ambiguous queries. The system tracks unanswerable queries (`failed_attempt_count`). On the first strike, it triggers a clarification question. Only on the second consecutive failure does it trip the escalation lock (`pending_escalation = True`).
2. **Reroute detection:** In `escalation_confirmation_node`, if the user replies with something that does not match affirmative or negative patterns (e.g., asking a new question), the system clears dead-end conversational history via the Amnesia Protocol, releases the escalation lock, and routes the message back through the standard pipeline (guardrail → intent → RAG).
3. **State Checkpoint Preservation:** In production (`src/chatbot.py`), passing full state dictionaries on every turn accidentally wiped out multi-turn scalar counters like `failed_attempt_count`. Fixing `generate_response()` to pass only message payloads (`{"messages": [HumanMessage(...)]}`) allows LangGraph's `MemorySaver` to preserve persistent state across turns.

## Guardrail: brand-term masking

Early testing showed the PII masker consistently misclassified "Kaufland", "kauflandpay", and "Bluecode" as `<PERSON>` or `<LOCATION>` entities. This was partially fixed by adding an explicit `BRAND_TERM` pattern recognizer with a deny-list of known brand terms, combined with span-overlap filtering so the general NER model's competing (incorrect) detection on the same text span is also suppressed, not just the brand-term tag itself. This fix is scoped to known terms and a genuinely novel or unlisted brand name could still be mismasked.

## Hybrid retrieval: why BM25 was added on top of dense search

Even after normalizing embeddings and using cosine similarity (rather than raw distance) for a fairer comparison between the query and candidate answers, dense retrieval alone still failed on realistic user input: mispronounced or merged words. For example, a paused breath between syllables becoming a missing space ("kauflandpay" instead of "kaufland pay"), and plain typos. 
BM25 lexical search, fused with dense results via Reciprocal Rank Fusion, directly targets this: it can match on exact or near-exact token overlap in a way dense embeddings sometimes miss for short, brand-heavy queries. A custom fuzzy spell-corrector, built from the FAQ corpus's own vocabulary , handles the typo/merge correction step before either retriever runs.

## Trade-offs considered

- **Speed vs. safety**: Per-document Llama Guard checks on retrieved content instead of one call over the full concatenated context. This requires more API calls and introduces latency, but the single-call approach was silently failing (`context_length_exceeded`, since the guard model's context window is much smaller than a multi-document RAG context) and falling back to a much weaker keyword-only check on every real query, which is an invisible safety regression worse than the added latency.
- **Simplicity vs. coverage**: Three-tier confidence instead of binary, accepted at the cost of reasoning about and testing more states, because binary routing was the direct cause of the escalation dead-end.
- **Reliability vs. cost**: Hybrid retrieval instead of dense-only, accepted at the cost of extra compute per query because it fixes concrete, reproduced failures on typo/merge cases.




