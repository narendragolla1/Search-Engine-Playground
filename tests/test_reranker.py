import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.core.search_engine import SearchEngine

def main():
    with open('data/movies_test.json', 'r') as f:
        data = json.load(f)['movies']
        
    print(f"Loaded {len(data)} movies.")
    
    # Init engine with BOTH models
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    engine = SearchEngine(model=model, reranker=reranker)
    engine.index(data, searchable_fields=['Title', 'Genre', 'Plot'])
    
    print("\n--- Test Reranker: 'An American classic mafia film' ---")
    query = "An American classic mafia film"
    print(f"Query: '{query}'")
    results = engine.search(query=query, top_k=3)
    
    for i, r in enumerate(results):
        print(f"  #{i+1}: {r.item['Title']} (Cross-Encoder Score: {r.score:.4f})")

if __name__ == "__main__":
    main()
