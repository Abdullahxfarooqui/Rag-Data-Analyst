
import sys
import os
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.query_router import classify_query_mode, QueryMode, detect_intent

def test_query(query):
    print(f"\nTesting query: '{query}'")
    
    # Test detect_intent directly
    intent = detect_intent(query)
    print(f"  detect_intent result: {intent}")
    
    # Test classify_query_mode
    mode, confidence, reason = classify_query_mode(query)
    print(f"  Mode: {mode}")
    print(f"  Confidence: {confidence}")
    print(f"  Reason: {reason}")
    
    if mode == QueryMode.DOCUMENT_OVERVIEW:
        print("  ✅ Correctly classified as DOCUMENT_OVERVIEW")
    elif mode == QueryMode.DATA_QUERY:
        print("  ❌ Incorrectly classified as DATA_QUERY")
    else:
        print(f"  ❌ Classified as {mode}")

if __name__ == "__main__":
    queries = [
        "what is in this document?",
        "what is in this document",
        "what's in this document",
        "describe this file",
        "give me a summary",
        "show me oil production", # Should be DATA_QUERY
        "what is the total gas", # Should be DATA_QUERY
    ]
    
    for q in queries:
        test_query(q)
