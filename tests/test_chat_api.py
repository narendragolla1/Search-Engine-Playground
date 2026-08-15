import httpx
import pytest
from src.core.config import settings

API_URL = settings.api_url

def test_chat():
    search_payload = {
        "query": "Which movie is about a mafia family?",
        "top_k": 3
    }
    
    try:
        search_resp = httpx.post(f"{API_URL}/search", json=search_payload)
    except httpx.RequestError:
        pytest.skip(f"API is not running at {API_URL}")
        
    assert search_resp.status_code == 200
        
    results = search_resp.json().get('results', [])
    context_items = [r['item'] for r in results]

    chat_payload = {
        "messages": [
            {"role": "user", "content": "Which movie is about a mafia family?"}
        ],
        "context": context_items
    }
    
    with httpx.stream("POST", f"{API_URL}/chat", json=chat_payload) as r:
        assert r.status_code == 200
        
        full_text = ""
        for chunk in r.iter_text():
            if chunk:
                full_text += chunk
        
        assert len(full_text) > 0
        assert "Godfather" in full_text
