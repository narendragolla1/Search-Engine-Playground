import requests
import json
import time

API_URL = "http://localhost:8000"

def test_chat():
    print("--- Starting Chat API Test ---")
    
    # 1. First, search for context
    print("\n1. Searching for context...")
    search_payload = {
        "query": "Which movie is about a mafia family?",
        "top_k": 3
    }
    
    search_resp = requests.post(f"{API_URL}/search", json=search_payload)
    if search_resp.status_code != 200:
        print(f"❌ Failed to search: {search_resp.text}")
        return
        
    results = search_resp.json().get('results', [])
    context_items = [r['item'] for r in results]
    print(f"✅ Retrieved {len(context_items)} items for context.")

    # 2. Call /chat endpoint
    print("\n2. Testing /chat streaming endpoint...")
    chat_payload = {
        "messages": [
            {"role": "user", "content": "Which movie is about a mafia family?"}
        ],
        "context": context_items
    }
    
    try:
        with requests.post(f"{API_URL}/chat", json=chat_payload, stream=True) as r:
            r.raise_for_status()
            print("✅ Connected to stream. Receiving chunks:")
            
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
            print("\n✅ Stream completed successfully.")
    except Exception as e:
        print(f"\n❌ Chat API failed: {e}")

if __name__ == "__main__":
    test_chat()
