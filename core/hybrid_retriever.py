# core/hybrid_retriever.py
import re
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz


class BM25Retriever:
    def __init__(self, documents: list[str], metadatas: list[dict] | None = None):
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"content": self.documents[i], "metadata": self.metadatas[i], "score": scores[i]}
            for i in ranked_idx if scores[i] > 0
        ]


class VocabularySpellCorrector:
    """Builds candidates from your own corpus — no hand-maintained list.
    Handles both typos (kauflnd -> kaufland) and merged words
    (kauflandpay -> kaufland pay), distinguishing the two by whether the
    best whole-word match is a close length fit."""

    def __init__(self, documents: list[str], min_word_len: int = 3):
        vocab = set()
        for doc in documents:
            vocab.update(re.findall(r"\w+", doc.lower()))
        self.vocab = list(vocab)
        self.vocab_set = set(w for w in vocab if len(w) >= min_word_len)

    def _best_match(self, token: str, cutoff: int) -> str | None:
        match = process.extractOne(token, self.vocab, scorer=fuzz.ratio)
        return match[0] if match and match[1] >= cutoff else None

    def _try_split(self, token: str, cutoff: int) -> str | None:
        if len(token) < 6:
            return None
        for i in range(3, len(token) - 2):
            left, right = token[:i], token[i:]
            if len(right) < 3:
                continue
            left_match = self._best_match(left, cutoff)
            right_match = self._best_match(right, cutoff)
            if left_match and right_match:
                return f"{left_match} {right_match}"
        return None

    def correct(self, query: str, cutoff: int = 82) -> str:
        corrected = []
        for token in query.lower().split():
            if token in self.vocab_set or len(token) < 3:
                corrected.append(token)
                continue

            whole_match = self._best_match(token, cutoff)

            if whole_match and abs(len(token) - len(whole_match)) <= 2:
                corrected.append(whole_match)
                continue

            split_match = self._try_split(token, cutoff) if len(token) >= 6 else None
            if split_match:
                corrected.append(split_match)
                continue

            corrected.append(whole_match if whole_match else token)

        return " ".join(corrected)


def reciprocal_rank_fusion(result_lists: list[list[dict]], key: str = "content", k: int = 60, top_k: int = 5) -> list[dict]:
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc[key]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[doc_id] = doc
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**doc_lookup[doc_id], "rrf_score": score} for doc_id, score in ranked]


# --- Standalone Test Block ---
if __name__ == "__main__":
    from core.rag_engine import KauflandRAG

    rag = KauflandRAG()
    docs, metadatas = rag.get_all_documents()

    print(f"Loaded {len(docs)} documents for testing.\n")

    corrector = VocabularySpellCorrector(docs)
    bm25 = BM25Retriever(docs, metadatas)

    test_queries = [
        "kauflandpay",
        "wue benutze ich kaufland pay",
        "was ist bluecode",
        "wie ist kauflnd card xtra",
    ]

    for q in test_queries:
        corrected = corrector.correct(q)
        print(f"--- Query: '{q}' ---")
        print(f"Corrected: '{corrected}'")

        lexical_results = bm25.search(corrected, top_k=3)
        print(f"BM25 top results ({len(lexical_results)} found):")
        for r in lexical_results:
            print(f"  score={r['score']:.2f} | {r['content'][:70]}...")
        print()