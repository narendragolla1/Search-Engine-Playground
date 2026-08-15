import json
from sentence_transformers import SentenceTransformer
from search_engine import SearchEngine

def main():
    with open('movies_test.json', 'r') as f:
        data = json.load(f)['movies']
        
    print(f"Loaded {len(data)} movies.")
    
    # Init engine
    model = SentenceTransformer("all-MiniLM-L6-v2")
    engine = SearchEngine(model=model)
    
    engine.index(data, searchable_fields=['Title', 'Genre', 'Plot'])
    
    print("\n--- Test 1: Empty Query, Filter imdbRating > 9.0 ---")
    filters = {'imdbRating': {'min': 9.1, 'max': 10.0}}
    results = engine.search(query="", filters=filters, top_k=5)
    for r in results:
        print(f"{r.item['Title']} - Rating: {r.item['imdbRating']}")
        
    print("\n--- Test 2: Text Query 'mafia', Filter imdbRating > 8.5 ---")
    filters = {'imdbRating': {'min': 8.6, 'max': 10.0}}
    results = engine.search(query="mafia organized crime", filters=filters, top_k=3)
    for r in results:
        print(f"{r.item['Title']} - Rating: {r.item['imdbRating']} - Score: {r.score:.3f}")
        
if __name__ == "__main__":
    main()
