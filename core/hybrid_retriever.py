import re
import os
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from rank_bm25 import BM25Okapi
from symspellpy import SymSpell

class BM25Retriever:
    def __init__(self, documents: list[str], metadatas: list[dict] | None = None):
        self.stemmer = SnowballStemmer("german") # Initialize German stemmer
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        # Extract words and stem them down to their root form
        words = re.findall(r"\w+", text.lower())
        return [self.stemmer.stem(w) for w in words]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"content": self.documents[i], "metadata": self.metadatas[i], "score": scores[i]}
            for i in ranked_idx if scores[i] > 0
        ]


class SymSpellCorrector:
    def __init__(self, documents: list[str], max_edit_distance: int = 2):
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )
        
        # Define where to save the pre-compiled dictionary
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dict_path = os.path.join(script_dir, "..", "data", "symspell_dict.txt")

        # 1. FAST PATH: If the dictionary already exists, load it instantly
        if os.path.exists(self.dict_path):
            print("[SymSpell] Loading pre-compiled dictionary from disk (Instant)...")
            self.sym_spell.load_dictionary(self.dict_path, term_index=0, count_index=1)
        
        # 2. SLOW PATH: First run only (or if you add a new larger corpus)
        else:
            print("[SymSpell] First run: Building and caching dictionary (This will take a moment)...")
            words = []
            for doc in documents:
                words.extend(re.findall(r"\w+", doc.lower()))
            
            word_counts = Counter(words)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
            
            # Save it to a text file for future instant loads
            with open(self.dict_path, "w", encoding="utf-8") as f:
                for word, count in word_counts.items():
                    # SymSpell expects format: "word count"
                    f.write(f"{word} {count}\n")
                    self.sym_spell.create_dictionary_entry(word, count)

    def correct(self, query: str) -> str:
        clean_query = query.lower().strip()
        if not clean_query:
            return query

        # lookup_compound handles typos, merged tokens, and punctuation in a single pass
        suggestions = self.sym_spell.lookup_compound(
            clean_query,
            max_edit_distance=2,
            ignore_non_words=True,
        )

        return suggestions[0].term if suggestions else query


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

    corrector = SymSpellCorrector(docs)
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