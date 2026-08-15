import httpx
import json
import pytest
from src.core.config import settings

API_URL = settings.api_url

def test_e2e():
    # 1. Load Data
    try:
        with open('data/movies_test.json', 'r') as f:
            data = json.load(f)['movies']
    except FileNotFoundError:
        pytest.skip("data/movies_test.json not found")
        
    # 2. Test /index endpoint
    payload = {
        "data": data,
        "searchable_fields": ["Title", "Genre", "Plot"]
    }
    
    try:
        resp = httpx.post(f"{API_URL}/index", json=payload)
    except httpx.RequestError:
        pytest.skip(f"API is not running at {API_URL}")
        
    assert resp.status_code == 200
    assert resp.json().get("items_indexed") == len(data)

    # 3. Test /search endpoint (Keyword + Semantic + Reranker)
    search_payload = {
        "query": "An American classic mafia film",
        "top_k": 3
    }
    
    search_resp = httpx.post(f"{API_URL}/search", json=search_payload)
    assert search_resp.status_code == 200
    
    results = search_resp.json().get('results', [])
    assert len(results) > 0
    assert "Godfather" in results[0]['item'].get('Title', '')
