import os
import re
from typing import Tuple, Dict, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
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

        # ====================================================
        # Loading Prompt Guard LOCALLY into server RAM
        # ====================================================
        print("[Guardrails] Loading local offline security model...")
        
        # Switched to Deepset's model and added tokenizer truncation
        self.guard_pipeline = pipeline(
            "text-classification", 
            model="patronus-studio/wolf-defender-prompt-injection",
            truncation=True,
            max_length=512
        )

        self.dangerous_code_patterns = [
            r"os\.", r"subprocess\.", r"eval\(", r"exec\(", 
            r"__import__", r";\s*rm", r"\|\s*bash", r"&\s*sh"
        ]
        self.safe_document_dir = os.path.abspath("./safe_docs")


    # ==========================================
    # LOCAL OFFLINE SECURITY CHECK
    # ==========================================
    def check_with_prompt_guard(self, text: str) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, ""
            
        try:
            result = self.guard_pipeline(text)[0]
            label = result['label'].upper()
            score = float(result.get('score', 1.0))

            # DEBUG: Print what the guardrail is seeing during testing
            print(f"[Guardrails Debug] Text: '{text[:40]}...' | Label: {label} | Score: {score:.3f}")
            
            # Deepset model labels malicious inputs as 'INJECTION' and safe as 'LEGIT'
            if label == 'INJECTION' and score > 0.80:
                return True, "Diese Anfrage konnte nicht aus Sicherheitsgründen bearbeitet werden."
                
            return False, ""
            
        except Exception as e:
            print(f"[Guardrails] Error during local classification: {e}")
            return False, ""

    # ==========================================
    # INPUT VALIDATION
    # ==========================================
    def mask_pii(self, text: str) -> str:
            results = self.analyzer.analyze(text=text, language="de")
            
            # Keep protected brand terms safe from being redacted
            protected_spans = [(r.start, r.end) for r in results if r.entity_type == "BRAND_TERM"]
            def overlaps_protected(result) -> bool:
                return any(result.start < end and result.end > start for start, end in protected_spans)
                
            results = [r for r in results if r.entity_type != "BRAND_TERM" and not overlaps_protected(r)]
            
            anonymized_result = self.anonymizer.anonymize(
                text=text, 
                analyzer_results=results
            )
            
            raw_masked_text = anonymized_result.text

            # Automatically convert ANY Presidio angle-bracket tag 
            # into safe square brackets [TAG] for the injection classifier.
            # This handles every current and future PII category automatically.
            safe_masked_text = re.sub(r'<([A-Z_]+)>', r'\1', raw_masked_text)
            
            return safe_masked_text

    def _keyword_prefilter(self, text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.injection_keywords)

    def validate_input(self, user_input: str) -> str:
        """Safely checks user input, chunking it if it's extremely long to avoid 512 token limit."""
        clean_text = self.mask_pii(user_input)

        if self._keyword_prefilter(clean_text):
            raise SecurityError("Diese Anfrage konnte nicht verarbeitet werden.")

        # Slice into 1500-character chunks to safely fit inside context window
        input_chunks = [clean_text[i:i+1500] for i in range(0, len(clean_text), 1500)]
        
        for chunk in input_chunks:
            is_injection, _ = self.check_with_prompt_guard(chunk)
            if is_injection:
                raise SecurityError("Diese Anfrage konnte nicht verarbeitet werden.")

        return clean_text

    # ==========================================
    # TRADITIONAL APP SECURITY
    # ==========================================
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