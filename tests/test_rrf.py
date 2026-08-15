import json
from sentence_transformers import SentenceTransformer
from src.core.search_engine import SearchEngine

def main():
    with open('data/movies_test.json', 'r') as f:
        data = json.load(f)['movies']
        
    print(f"Loaded {len(data)} movies.")
    
    # Init engine
    model = SentenceTransformer("all-MiniLM-L6-v2")
    engine = SearchEngine(model=model)
    engine.index(data, searchable_fields=['Title', 'Genre', 'Plot'])
    
    print("\n--- Test RRF: Exact Keyword Match ---")
    query = "The Godfather"
    print(f"Query: '{query}'")
    results = engine.search(query=query, top_k=3)
    for r in results:
        print(f"  {r.item['Title']} (Score: {r.score:.4f}, Kw: {r.keyword_score:.4f}, Sem: {r.semantic_score:.4f})")

    print("\n--- Test RRF: Massive Spelling Mistake (Semantic takes over) ---")
    query = "godfther pt 2" # Typo
    print(f"Query: '{query}'")
    results = engine.search(query=query, top_k=3)
    for r in results:
        print(f"  {r.item['Title']} (Score: {r.score:.4f}, Kw: {r.keyword_score:.4f}, Sem: {r.semantic_score:.4f})")

if __name__ == "__main__":
    main()
