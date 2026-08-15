import requests
import json
import time

API_URL = "http://localhost:8000"

def test_e2e():
    print("--- Starting End-to-End API Test ---")
    
    # 1. Load Data
    with open('data/movies_test.json', 'r') as f:
        data = json.load(f)['movies']
        
    print(f"Loaded {len(data)} movies for testing.")
    
    # 2. Test /index endpoint
    print("\n1. Testing /index endpoint...")
    payload = {
        "data": data,
        "searchable_fields": ["Title", "Genre", "Plot"]
    }
    
    start_time = time.time()
    resp = requests.post(f"{API_URL}/index", json=payload)
    
    if resp.status_code == 200:
        print(f"✅ Successfully indexed data in {time.time() - start_time:.2f}s")
        print(f"Response: {resp.json()}")
    else:
        print(f"❌ Failed to index data: {resp.status_code} - {resp.text}")
        return

    # 3. Test /search endpoint (Keyword + Semantic + Reranker)
    print("\n2. Testing /search endpoint...")
    search_payload = {
        "query": "An American classic mafia film",
        "top_k": 3
    }
    
    start_time = time.time()
    search_resp = requests.post(f"{API_URL}/search", json=search_payload)
    
    if search_resp.status_code == 200:
        print(f"✅ Successfully completed search in {time.time() - start_time:.2f}s")
        results = search_resp.json().get('results', [])
        
        print("\nTop Results:")
        for i, r in enumerate(results):
            print(f"  #{i+1}: {r['item'].get('Title')} (Score: {r['score']:.4f})")
            
        # Verify the CrossEncoder put Godfather at the top
        if results and "Godfather" in results[0]['item'].get('Title', ''):
            print("\n✅ End-to-End Test Passed: The Godfather is the top result!")
        else:
            print("\n❌ End-to-End Test Failed: Expected 'The Godfather' to be the top result.")
    else:
        print(f"❌ Failed to search: {search_resp.status_code} - {search_resp.text}")

if __name__ == "__main__":
    test_e2e()
