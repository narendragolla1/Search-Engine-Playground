import httpx
import json
import pytest
from src.core.config import settings

API_URL = settings.api_url

@pytest.fixture(scope="module", autouse=True)
def setup_index():
    """Fixture to load test data and index it before running tests."""
    try:
        with open('data/movies_test.json', 'r') as f:
            movies_data = json.load(f)['movies']
    except FileNotFoundError:
        movies_data = []
        pytest.skip("data/movies_test.json not found")

    if movies_data:
        payload = {
            "data": movies_data,
            "searchable_fields": ["Title", "Genre", "Plot", "Actors", "Director"]
        }
        try:
            httpx.post(f"{API_URL}/index", json=payload)
        except httpx.RequestError:
            pytest.skip(f"API is not running at {API_URL}")
    yield

def test_search_movie_by_title():
    """Test searching for a specific movie by exact title match or close semantic match"""
    payload = {
        "query": "The Shawshank Redemption",
        "top_k": 1
    }
    resp = httpx.post(f"{API_URL}/search", json=payload)
    assert resp.status_code == 200
    
    results = resp.json().get('results', [])
    assert len(results) > 0
    assert results[0]['item']['Title'] == "The Shawshank Redemption"

def test_search_movie_by_concept():
    """Test searching for a movie using a semantic concept rather than keywords"""
    payload = {
        "query": "a dream within a dream heist",
        "top_k": 3
    }
    resp = httpx.post(f"{API_URL}/search", json=payload)
    assert resp.status_code == 200
    
    results = resp.json().get('results', [])
    assert len(results) > 0
    
    titles = [r['item'].get('Title') for r in results]
    assert any(t in ["Inception", "Memento", "The Matrix", "Shutter Island"] for t in titles)

def test_rag_chat_with_context():
    """Test the RAG /chat endpoint to ensure it can answer questions based on search context"""
    search_payload = {
        "query": "Which movie involves a wrongful imprisonment and escaping through a tunnel?",
        "top_k": 2
    }
    search_resp = httpx.post(f"{API_URL}/search", json=search_payload)
    assert search_resp.status_code == 200
    
    context_items = [r['item'] for r in search_resp.json().get('results', [])]
    
    chat_payload = {
        "messages": [
            {"role": "user", "content": "What is the name of the movie about wrongful imprisonment? Name the director too."}
        ],
        "context": context_items
    }
    
    chat_resp = httpx.post(f"{API_URL}/chat", json=chat_payload)
    assert chat_resp.status_code == 200
    
    answer = chat_resp.text
    assert len(answer) > 0
    assert "Shawshank" in answer
    assert "Frank" in answer and "Darabont" in answer
