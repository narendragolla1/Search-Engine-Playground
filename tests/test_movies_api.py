import requests
import json
import unittest
import time

API_URL = "http://localhost:8000"

class TestMoviesAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the movies test data
        try:
            with open('data/movies_test.json', 'r') as f:
                cls.movies_data = json.load(f)['movies']
        except FileNotFoundError:
            cls.movies_data = []
            print("Warning: data/movies_test.json not found.")

        # Index the data before running tests
        if cls.movies_data:
            payload = {
                "data": cls.movies_data,
                "searchable_fields": ["Title", "Genre", "Plot", "Actors", "Director"]
            }
            try:
                requests.post(f"{API_URL}/index", json=payload)
            except requests.exceptions.ConnectionError:
                print("Warning: API is not running at http://localhost:8000")

    def test_1_search_movie_by_title(self):
        """Test searching for a specific movie by exact title match or close semantic match"""
        payload = {
            "query": "The Shawshank Redemption",
            "top_k": 1
        }
        resp = requests.post(f"{API_URL}/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        
        results = resp.json().get('results', [])
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['item']['Title'], "The Shawshank Redemption")

    def test_2_search_movie_by_concept(self):
        """Test searching for a movie using a semantic concept rather than keywords"""
        payload = {
            "query": "a dream within a dream heist",
            "top_k": 3
        }
        resp = requests.post(f"{API_URL}/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        
        results = resp.json().get('results', [])
        self.assertTrue(len(results) > 0)
        
        # Memento is also a good conceptual match in some models
        titles = [r['item'].get('Title') for r in results]
        self.assertTrue(any(t in ["Inception", "Memento", "The Matrix", "Shutter Island"] for t in titles))
        
    def test_3_rag_chat_with_context(self):
        """Test the RAG /chat endpoint to ensure it can answer questions based on search context"""
        # First get the context
        search_payload = {
            "query": "Which movie involves a wrongful imprisonment and escaping through a tunnel?",
            "top_k": 2
        }
        search_resp = requests.post(f"{API_URL}/search", json=search_payload)
        self.assertEqual(search_resp.status_code, 200)
        
        context_items = [r['item'] for r in search_resp.json().get('results', [])]
        
        # Now use the chat endpoint
        chat_payload = {
            "messages": [
                {"role": "user", "content": "What is the name of the movie about wrongful imprisonment? Name the director too."}
            ],
            "context": context_items
        }
        
        chat_resp = requests.post(f"{API_URL}/chat", json=chat_payload)
        self.assertEqual(chat_resp.status_code, 200)
        
        # Since it's a streaming response, we need to read the content
        answer = chat_resp.text
        self.assertTrue(len(answer) > 0)
        self.assertIn("Shawshank", answer)
        self.assertTrue("Frank" in answer and "Darabont" in answer)

if __name__ == '__main__':
    unittest.main()
