# 🔍 Search Engine Playground

A full-stack **semantic search engine** built for experimentation and learning. It combines dense (semantic) and sparse (keyword) retrieval using a hybrid approach — fused via Reciprocal Rank Fusion (RRF) — and optionally re-ranks results with a cross-encoder. A RAG-powered chat endpoint and an interactive Streamlit UI round out the stack.

---

## ✨ Features

- **Hybrid Search** — Combines dense embeddings (SentenceTransformers) + sparse embeddings (SPLADE) using Qdrant's native RRF fusion
- **Re-ranking** — Optional cross-encoder re-ranking pass for higher precision
- **Filters** — Numeric range and categorical filters applied at the vector-store level
- **Pagination** — Offset-based pagination on all search results
- **CRUD** — Index, update, and delete documents at runtime via REST API
- **Intent Extraction** — LLM-powered query intent extraction to auto-populate filters
- **RAG Chat** — Streaming `/chat` endpoint backed by Groq LLMs with search context
- **Schema Analysis** — LLM-assisted analysis of uploaded data to suggest searchable fields
- **Streamlit UI** — Interactive playground for uploading data, tuning search, and chatting

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│              src/frontend/app.py                     │
└─────────────────────┬────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼────────────────────────────────┐
│              FastAPI Backend  (src/api/)              │
│  /index  /search  /chat  /schema/analyze             │
│  /document (PUT / DELETE)                            │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│            SearchEngine Core (src/core/)             │
│  SentenceTransformer (dense)                         │
│  SPLADE fastembed    (sparse)                        │
│  CrossEncoder        (re-rank, optional)             │
│  Qdrant              (vector store, local on disk)   │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager
- A free [Groq API key](https://console.groq.com/) (required only for chat / intent extraction)

### 1. Clone the repository

```bash
git clone <repo-url>
cd "Search Engine"
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Start the API server

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 5. Launch the Streamlit UI

In a separate terminal:

```bash
uv run streamlit run src/frontend/app.py
```

The playground opens at `http://localhost:8501`.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/index` | Index a list of documents |
| `PUT` | `/document` | Update an existing document |
| `DELETE` | `/document/{id}` | Delete a document by ID |
| `POST` | `/search` | Hybrid search with optional filters |
| `POST` | `/chat` | Streaming RAG chat (SSE) |
| `POST` | `/schema/analyze` | LLM schema analysis for uploaded data |

### Example: Index documents

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"id": "1", "title": "Inception", "genre": "Sci-Fi", "year": "2010"}
    ],
    "searchable_fields": ["title", "genre"],
    "field_weights": {"title": 2.0}
  }'
```

### Example: Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mind-bending thriller",
    "top_k": 5,
    "filters": {"genre": "Sci-Fi"}
  }'
```

---

## 🧪 Running Tests

```bash
uv run pytest
```

Tests are organized by marker:

| Marker | Description |
|--------|-------------|
| `search` | Core search quality — keyword, semantic, hybrid |
| `filter` | Numeric range and categorical filter logic |
| `paginate` | Pagination and offset behavior |
| `crud` | Index / update / delete document lifecycle |
| `chat` | RAG `/chat` endpoint (requires `GROQ_API_KEY`) |

Run a specific group:

```bash
uv run pytest -m search
uv run pytest -m filter
```

---

## 🗂️ Project Structure

```
Search Engine/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app & route handlers
│   │   ├── schemas.py       # Pydantic request/response models
│   │   ├── dependencies.py  # Dependency injection (SearchEngine singleton)
│   │   └── llm_service.py   # Groq LLM integration (chat, intent, schema)
│   ├── core/
│   │   ├── search_engine.py # Hybrid search engine (dense + sparse + rerank)
│   │   └── config.py        # App configuration via pydantic-settings
│   └── frontend/
│       └── app.py           # Streamlit UI playground
├── tests/                   # Pytest test suite
├── data/                    # Sample datasets
├── qdrant_data/             # Local Qdrant vector store (auto-created)
├── .env.example             # Environment variable template
├── pyproject.toml           # Project metadata & dependencies
└── uv.lock                  # Locked dependency tree
```

---

## 🔧 Key Technologies

| Layer | Technology |
|-------|-----------|
| Vector Store | [Qdrant](https://qdrant.tech/) (local on-disk mode) |
| Dense Embeddings | [SentenceTransformers](https://www.sbert.net/) |
| Sparse Embeddings | [SPLADE via fastembed](https://github.com/qdrant/fastembed) |
| Re-ranking | CrossEncoder (SentenceTransformers) |
| LLM / Chat | [Groq](https://groq.com/) API |
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| UI | [Streamlit](https://streamlit.io/) |
| Package Manager | [uv](https://docs.astral.sh/uv/) |

---

## 📝 License

This project is for personal learning and experimentation. Feel free to adapt it for your own projects.
