import os
import time
from langchain_core.messages import HumanMessage
from agent.intent_agent import intent_node  

EVAL_DATA = [
    # RAG
    ("Wie funktioniert Kaufland Pay?", "rag"),
    ("Was sind die Öffnungszeiten heute?", "rag"),
    ("Wie bekomme ich mein Geld bei einer Rückgabe zurück?", "rag"), 
    
    # POLICY_REFUSAL
    ("Bitte storniere meinen Einkauf von gestern.", "policy_refusal"), 
    ("Können Sie mir 500 Treuepunkte gutschreiben?", "policy_refusal"),
    ("Ich möchte das Passwort für das Konto meiner Frau ändern.", "policy_refusal"),
    
    # ESCALATE
    ("Ich bin stinksauer, gebt mir sofort einen Menschen!", "escalate"),
    ("Euer Service ist furchtbar, ich schalte meinen Anwalt ein.", "escalate"),
    
    # SMALL_TALK
    ("Hallo, guten Morgen!", "small_talk"),
    ("Vielen Dank für die Hilfe.", "small_talk"),
    
    # OUT_OF_DOMAIN
    ("Wie wird das Wetter morgen in Stuttgart?", "out_of_domain"),
    ("Wer hat die Europameisterschaft gewonnen?", "out_of_domain"),
    
    # HARD / AMBIGUOUS / MESSY CASES
    ("Das ist wirklich frustrierend, wie funktioniert eigentlich Kaufland Pay?", "rag"),
    ("kann mir jemand helfen mit kauflandpay", "rag"),
    ("wue funktioniert das bezahlen", "rag"),
    ("ich will mit dem chef reden euer bot ist dumm", "escalate"),
    ("schreib mir 50 punkte gut bitte danke", "policy_refusal"),
]

# Exact Groq Model Identifiers for Comparison
MODELS_TO_TEST = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b"
]

def run_evaluation():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in environment.")
        return

    print("🚀 Starting Groq Multi-Model Intent Benchmark...\n")

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"🧠 Evaluating Model Target: {model_name}")
        print(f"{'='*60}")
        
        correct_predictions = 0
        total_time = 0
        parse_errors = 0
        
        for query, expected_action in EVAL_DATA:
            mock_state = {
                "messages": [HumanMessage(content=query)],
                "action": "rag",
                "retrieved_context": "",
                "confidence_score": 0.0,
                "confidence_tier": "high",
                "escalation_ticket": {},
                "pending_escalation": False,
                "escalation_retry_count": 0,
                "failed_attempt_count": 0
            }
            
            config = {"configurable": {"groq_intent_model": model_name}}
            
            # Pacing delay to stay clear of rate limits
            time.sleep(2.5) 
            
            start_time = time.perf_counter()
            
            try:
                result = intent_node(mock_state, config)
                actual_action = result.get("action")
                latency = time.perf_counter() - start_time
                total_time += latency
                
                if actual_action == expected_action:
                    correct_predictions += 1
                    print(f"✅ PASS | '{query}' -> {actual_action}")
                else:
                    print(f"❌ FAIL | '{query}' | Expected: {expected_action} | Got: {actual_action}")
                    
            except Exception as e:
                print(f"⚠️ ERROR | '{query}' | {e}")
                parse_errors += 1
                total_time += (time.perf_counter() - start_time)

        total_queries = len(EVAL_DATA)
        accuracy = (correct_predictions / total_queries) * 100
        avg_latency = (total_time / total_queries) * 1000  
        
        print(f"\n📊 RESULTS FOR {model_name}:")
        print(f"Accuracy:      {accuracy:.1f}% ({correct_predictions}/{total_queries})")
        print(f"Avg Latency:   {avg_latency:.0f} ms")
        print(f"Parse Errors:  {parse_errors}")

if __name__ == "__main__":
    run_evaluation()