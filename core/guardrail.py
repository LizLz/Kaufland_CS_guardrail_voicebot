import os
import re
from typing import Tuple, Dict, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from transformers import pipeline 
from dotenv import load_dotenv

load_dotenv()

PROTECTED_TERMS = [
    "Kaufland", "Kaufland Pay", "KauflandPay", "kauflandpay",
    "Kaufland Card XTRA", "real.de", "Kaufland.de", "BlueCode", "bluecode",
]

class SecurityError(Exception):
    pass

class GuardrailsManager:
    def __init__(self):
        print("[Guardrails] Loading PII NLP models...")
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["de"])
        self.anonymizer = AnonymizerEngine()

        brand_recognizer = PatternRecognizer(
            supported_entity="BRAND_TERM", deny_list=PROTECTED_TERMS, supported_language="de"
        )
        self.analyzer.registry.add_recognizer(brand_recognizer)

        self.injection_keywords = [
            "ignore previous instructions", "ignore all previous", "developer mode", "you are now",
            "system prompt", "bypass your",
            "ignoriere die vorherigen", "entwicklermodus", "ignoriere alle vorherigen",
            "du bist jetzt", "systemprompt",
        ]

        print("[Guardrails] Loading local offline security model...")
        
        self.guard_pipeline = pipeline(
            "text-classification", 
            model="patronus-studio/wolf-defender-prompt-injection",
            truncation=True,
            max_length=2048 
        )

        self.dangerous_code_patterns = [
            r"os\.", r"subprocess\.", r"eval\(", r"exec\(", 
            r"__import__", r";\s*rm", r"\|\s*bash", r"&\s*sh"
        ]
        self.safe_document_dir = os.path.abspath("./safe_docs")


    def check_with_prompt_guard(self, text: str) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, ""
            
        try:
            result = self.guard_pipeline(text)[0]
            label = result['label'].upper()
            score = float(result.get('score', 1.0))
            
            # Deepset/Patronus model labels malicious inputs as 'INJECTION'
            if label == 'INJECTION' and score > 0.80:
                return True, "Diese Anfrage konnte nicht aus Sicherheitsgründen bearbeitet werden."
                
            return False, ""
            
        except Exception as e:
            print(f"[Guardrails CRITICAL] Error during local classification: {e}")
            # If the model crashes (OOM), block the request to prevent bypass.
            return True, "Sicherheitsprüfung fehlgeschlagen. System blockiert."

    def mask_pii(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language="de")
        
        protected_spans = [(r.start, r.end) for r in results if r.entity_type == "BRAND_TERM"]
        def overlaps_protected(result) -> bool:
            return any(result.start < end and result.end > start for start, end in protected_spans)
            
        results = [r for r in results if r.entity_type != "BRAND_TERM" and not overlaps_protected(r)]
        
        anonymized_result = self.anonymizer.anonymize(
            text=text, 
            analyzer_results=results
        )
        
        # Angle bracket normalization for classifier safety
        safe_masked_text = re.sub(r'<([A-Z_]+)>', r'\1', anonymized_result.text)
        return safe_masked_text

    def _keyword_prefilter(self, text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.injection_keywords)

    def validate_input(self, user_input: str) -> str:
        clean_text = self.mask_pii(user_input)

        if self._keyword_prefilter(clean_text):
            raise SecurityError("Diese Anfrage konnte nicht verarbeitet werden (Keyword Filter).")

        # Sliding window chunking to prevent "Seam Injection" evasion.
        # Uses 6000 chars (~1200 tokens) with a 500-char overlap so payloads are never cut in half.
        chunk_size = 6000
        overlap = 500
        input_chunks = []
        
        if len(clean_text) <= chunk_size:
            input_chunks = [clean_text]
        else:
            for i in range(0, len(clean_text) - overlap, chunk_size - overlap):
                input_chunks.append(clean_text[i : i + chunk_size])
        
        for chunk in input_chunks:
            is_injection, _ = self.check_with_prompt_guard(chunk)
            if is_injection:
                raise SecurityError("Diese Anfrage konnte nicht verarbeitet werden (Prompt Guard).")

        return clean_text

    def validate_tool_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in args.items():
            if isinstance(value, str):
                for pattern in self.dangerous_code_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise SecurityError(f"Dangerous code signature detected in argument '{key}'.")
        return args

    def validate_file_path(self, requested_filename: str) -> str:
        absolute_requested_path = os.path.abspath(os.path.join(self.safe_document_dir, requested_filename))
        if not absolute_requested_path.startswith(self.safe_document_dir):
            raise SecurityError(f"Path traversal attempt detected: {requested_filename}")
        return absolute_requested_path

    def safe_chroma_metadata_filter(self, filter_key: str, provided_value: Any) -> dict:
        if not isinstance(provided_value, (str, int, float)):
            raise SecurityError("Invalid ChromaDB metadata filter type. Primitives only.")
        return {filter_key: {"$eq": provided_value}}