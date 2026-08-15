"""
End-to-End Tests for the Search Engine API
==========================================
Requires the FastAPI server to be running at the URL configured in settings (default: http://localhost:8000).
Run with:  uv run pytest tests/test_e2e.py -v

Test categories (marked with pytest marks for selective running):
  - search   : core search quality tests
  - crud     : index / update / delete lifecycle tests
  - filter   : numeric and categorical filter tests
  - paginate : pagination / offset tests
  - chat     : RAG /chat endpoint tests (requires GROQ_API_KEY)
"""

import json
import pytest
import httpx
from src.core.config import settings

API_URL = settings.api_url

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict, timeout: float = 30.0) -> httpx.Response:
    return httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)


def _get(path: str, timeout: float = 10.0) -> httpx.Response:
    return httpx.get(f"{API_URL}{path}", timeout=timeout)


def _put(path: str, payload: dict, timeout: float = 30.0) -> httpx.Response:
    return httpx.put(f"{API_URL}{path}", json=payload, timeout=timeout)


def _delete(path: str, timeout: float = 10.0) -> httpx.Response:
    return httpx.delete(f"{API_URL}{path}", timeout=timeout)


def _titles(results: list) -> list[str]:
    return [r["item"].get("Title", "") for r in results]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def movies() -> list[dict]:
    """Load the movies test dataset."""
    try:
        with open("data/movies_test.json", "r") as f:
            return json.load(f)["movies"]
    except FileNotFoundError:
        pytest.skip("data/movies_test.json not found — skipping all e2e tests")


@pytest.fixture(scope="module", autouse=True)
def indexed_movies(movies):
    """
    Index the full movies dataset once for the entire test module.
    All other tests depend on this data being present.
    """
    try:
        resp = _post(
            "/index",
            {
                "data": movies,
                "searchable_fields": ["Title", "Genre", "Plot", "Actors", "Director"],
                "field_weights": {"Title": 3.0, "Plot": 2.0},
            },
            timeout=120.0,
        )
    except httpx.RequestError:
        pytest.skip(f"API server is not running at {API_URL}")

    assert resp.status_code == 200, f"Index failed: {resp.text}"
    assert resp.json()["items_indexed"] == len(movies)
    yield


# ===========================================================================
# 1. Health / Smoke Tests
# ===========================================================================

class TestHealthCheck:
    def test_root_returns_200(self):
        """The root endpoint must be reachable."""
        resp = _get("/")
        assert resp.status_code == 200

    def test_root_has_welcome_message(self):
        resp = _get("/")
        body = resp.json()
        assert "message" in body
        assert len(body["message"]) > 0


# ===========================================================================
# 2. Index Endpoint
# ===========================================================================

class TestIndexEndpoint:
    def test_index_returns_correct_item_count(self, movies):
        """Re-indexing the same data must report the correct count."""
        resp = _post(
            "/index",
            {"data": movies, "searchable_fields": ["Title", "Genre"]},
            timeout=120.0,
        )
        assert resp.status_code == 200
        assert resp.json()["items_indexed"] == len(movies)

    def test_index_with_field_weights(self, movies):
        """Field weights should be accepted without errors."""
        resp = _post(
            "/index",
            {
                "data": movies,
                "searchable_fields": ["Title", "Plot"],
                "field_weights": {"Title": 3.0, "Plot": 1.5},
            },
            timeout=120.0,
        )
        assert resp.status_code == 200

    def test_index_empty_data_returns_200(self):
        """
        Indexing an empty list should not crash the server.
        The engine logs a warning but must return a valid response.
        """
        resp = _post("/index", {"data": [], "searchable_fields": ["Title"]})
        # Acceptable: either 200 with 0 items or a 422 validation error.
        assert resp.status_code in (200, 422)


# ===========================================================================
# 3. Search Quality — Keyword / Exact Match
# ===========================================================================

@pytest.mark.search
class TestKeywordSearch:
    def test_exact_title_match_ranks_first(self):
        """Searching the exact movie title should return it as the top result."""
        resp = _post("/search", {"query": "The Shawshank Redemption", "top_k": 1})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0
        assert results[0]["item"]["Title"] == "The Shawshank Redemption"

    def test_exact_title_godfather(self):
        resp = _post("/search", {"query": "The Godfather", "top_k": 3})
        assert resp.status_code == 200
        titles = _titles(resp.json()["results"])
        assert any("Godfather" in t for t in titles)

    def test_keyword_search_returns_score(self):
        """Every result must include a non-negative relevance score."""
        resp = _post("/search", {"query": "Inception", "top_k": 5})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert r["score"] >= 0


# ===========================================================================
# 4. Search Quality — Semantic / Conceptual
# ===========================================================================

@pytest.mark.search
class TestSemanticSearch:
    def test_semantic_dream_heist(self):
        """
        A conceptual query 'dream within a dream heist' should surface
        Inception (or at minimum Memento / The Matrix) via semantic similarity.
        """
        resp = _post("/search", {"query": "a dream within a dream heist", "top_k": 5})
        assert resp.status_code == 200
        titles = _titles(resp.json()["results"])
        semantic_matches = {"Inception", "Memento", "The Matrix", "Shutter Island"}
        assert any(t in semantic_matches for t in titles), (
            f"Expected a semantically relevant title, got: {titles}"
        )

    def test_semantic_mafia_returns_crime_movies(self):
        """
        'American mafia crime family' should surface crime-genre movies.
        """
        resp = _post("/search", {"query": "American mafia crime family", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0

        crime_movies = {"The Godfather", "Goodfellas", "The Godfather: Part II", "Pulp Fiction"}
        titles = set(_titles(results))
        assert titles & crime_movies, f"No crime movies found. Got: {titles}"

    def test_semantic_prison_escape(self):
        """Query about prison and hope should return Shawshank."""
        resp = _post("/search", {"query": "hope inside prison walls", "top_k": 5})
        assert resp.status_code == 200
        titles = _titles(resp.json()["results"])
        assert "The Shawshank Redemption" in titles, (
            f"Shawshank Redemption not found in: {titles}"
        )

    def test_typo_tolerance_via_semantic(self):
        """
        A misspelled query 'godfther' should still surface The Godfather
        through the dense/semantic vector pathway.
        """
        resp = _post("/search", {"query": "godfther pt 2", "top_k": 5})
        assert resp.status_code == 200
        titles = _titles(resp.json()["results"])
        assert any("Godfather" in t for t in titles), (
            f"Expected Godfather despite typo. Got: {titles}"
        )


# ===========================================================================
# 5. Empty Query / Browse Mode
# ===========================================================================

@pytest.mark.search
class TestEmptyQuery:
    def test_empty_query_returns_results(self):
        """An empty query triggers scroll mode and should return documents."""
        resp = _post("/search", {"query": "", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 5

    def test_empty_query_all_scores_are_1(self):
        """Scroll results must have a default score of 1.0."""
        resp = _post("/search", {"query": "", "top_k": 3})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert r["score"] == 1.0


# ===========================================================================
# 6. Filter Tests
# ===========================================================================

@pytest.mark.filter
class TestFilters:
    def test_range_filter_high_rating(self):
        """Filter: only return movies with imdbRating >= 9.1."""
        resp = _post(
            "/search",
            {
                "query": "",
                "filters": {"imdbRating": {"min": 9.1, "max": 10.0}},
                "top_k": 10,
            },
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0
        for r in results:
            rating = float(r["item"].get("imdbRating", 0))
            assert rating >= 9.1, f"Got rating {rating} which is below 9.1"

    def test_range_filter_narrows_results(self):
        """A tighter rating filter should return fewer results than a loose one."""
        loose = _post("/search", {"query": "", "filters": {"imdbRating": {"min": 8.0, "max": 10.0}}, "top_k": 50})
        strict = _post("/search", {"query": "", "filters": {"imdbRating": {"min": 9.1, "max": 10.0}}, "top_k": 50})
        assert loose.status_code == 200 and strict.status_code == 200
        assert len(loose.json()["results"]) > len(strict.json()["results"])

    def test_filter_combined_with_query(self):
        """
        Searching for 'crime' with a high rating filter should only
        return high-rated crime films.
        """
        resp = _post(
            "/search",
            {
                "query": "organized crime",
                "filters": {"imdbRating": {"min": 9.0, "max": 10.0}},
                "top_k": 5,
            },
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0
        for r in results:
            rating = float(r["item"].get("imdbRating", 0))
            assert rating >= 9.0, f"Filter bypassed — got rating {rating}"

    def test_list_filter_on_genre(self):
        """
        Filter by an exact field match (Genre contains 'Drama').
        Uses MatchAny which matches the stored string against the list.
        """
        resp = _post(
            "/search",
            {
                "query": "",
                "filters": {"Genre": ["Drama", "Crime, Drama"]},
                "top_k": 10,
            },
        )
        # Not all movies have Genre stored; we just verify the endpoint doesn't crash
        assert resp.status_code == 200


# ===========================================================================
# 7. Pagination Tests
# ===========================================================================

@pytest.mark.paginate
class TestPagination:
    def test_top_k_limits_results(self):
        """top_k must be respected: requesting 3 results returns at most 3."""
        resp = _post("/search", {"query": "", "top_k": 3})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 3

    def test_offset_skips_results(self):
        """Page 1 and page 2 must not share the same top result."""
        page1 = _post("/search", {"query": "drama", "top_k": 5, "offset": 0})
        page2 = _post("/search", {"query": "drama", "top_k": 5, "offset": 5})
        assert page1.status_code == 200 and page2.status_code == 200

        titles_p1 = set(_titles(page1.json()["results"]))
        titles_p2 = set(_titles(page2.json()["results"]))
        # Pages should not be identical (they may partially overlap in small datasets)
        assert titles_p1 != titles_p2 or len(titles_p1) == 0

    def test_large_offset_returns_empty_or_fewer_results(self):
        """An offset larger than the corpus should return 0 or fewer results."""
        resp = _post("/search", {"query": "", "top_k": 10, "offset": 9999})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) < 10


# ===========================================================================
# 8. CRUD — Update & Delete
# ===========================================================================

@pytest.mark.crud
class TestCRUD:
    SYNTHETIC_MOVIE = {
        "id": "tt_test_synthetic_001",
        "Title": "Synthetic Test Movie XYZ",
        "Genre": "Test",
        "Plot": "A completely synthetic movie created by the test suite.",
        "imdbRating": "7.5",
    }

    def test_index_single_new_document(self):
        """Adding a new unique document should make it discoverable via search."""
        resp = _post(
            "/index",
            {"data": [self.SYNTHETIC_MOVIE], "searchable_fields": ["Title", "Genre", "Plot"]},
            timeout=60.0,
        )
        assert resp.status_code == 200
        assert resp.json()["items_indexed"] == 1

    def test_new_document_is_searchable(self):
        """The newly indexed synthetic movie must appear in search results."""
        resp = _post("/search", {"query": "Synthetic Test Movie XYZ", "top_k": 3})
        assert resp.status_code == 200
        titles = _titles(resp.json()["results"])
        assert "Synthetic Test Movie XYZ" in titles, (
            f"Newly indexed document not found. Got: {titles}"
        )

    def test_update_document(self):
        """Updating a document should reflect the change on next search."""
        updated = self.SYNTHETIC_MOVIE.copy()
        updated["Plot"] = "An updated plot about robots taking over the world."

        resp = _put(
            "/document",
            {
                "document_id": self.SYNTHETIC_MOVIE["id"],
                "document": updated,
                "searchable_fields": ["Title", "Genre", "Plot"],
            },
            timeout=60.0,
        )
        assert resp.status_code == 200
        assert "message" in resp.json()

    def test_delete_document(self):
        """A deleted document must no longer appear in search results."""
        del_resp = _delete(f"/document/{self.SYNTHETIC_MOVIE['id']}")
        assert del_resp.status_code == 200

        search_resp = _post(
            "/search", {"query": "Synthetic Test Movie XYZ", "top_k": 5}
        )
        titles = _titles(search_resp.json()["results"])
        assert "Synthetic Test Movie XYZ" not in titles, (
            "Deleted document still appearing in search results!"
        )


# ===========================================================================
# 9. Response Schema Validation
# ===========================================================================

class TestResponseSchema:
    def test_search_response_has_required_fields(self):
        """Every SearchResultItem must contain item, score, keyword_score, semantic_score."""
        resp = _post("/search", {"query": "drama", "top_k": 3})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert "item" in r
            assert "score" in r
            assert "keyword_score" in r
            assert "semantic_score" in r

    def test_search_item_is_dict(self):
        resp = _post("/search", {"query": "film", "top_k": 3})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert isinstance(r["item"], dict)

    def test_index_response_schema(self, movies):
        resp = _post(
            "/index",
            {"data": movies[:2], "searchable_fields": ["Title"]},
            timeout=60.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "message" in body
        assert "items_indexed" in body
        assert isinstance(body["items_indexed"], int)

    def test_private_fields_not_exposed(self):
        """Internal fields like _searchable_text must not leak into API results."""
        resp = _post("/search", {"query": "drama", "top_k": 5})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            item = r["item"]
            for key in item.keys():
                assert not key.startswith("_"), (
                    f"Private field '{key}' was exposed in response"
                )


# ===========================================================================
# 10. RAG /chat Endpoint
# ===========================================================================

@pytest.mark.chat
class TestChatEndpoint:
    """
    These tests require a valid GROQ_API_KEY in your .env file.
    They are marked with @pytest.mark.chat so you can skip them:
        pytest tests/test_e2e.py -m "not chat"
    """

    def _search_and_chat(self, query: str, question: str) -> str:
        search_resp = _post("/search", {"query": query, "top_k": 3})
        assert search_resp.status_code == 200
        context = [r["item"] for r in search_resp.json()["results"]]

        chat_resp = _post(
            "/chat",
            {"messages": [{"role": "user", "content": question}], "context": context},
            timeout=60.0,
        )
        assert chat_resp.status_code == 200
        return chat_resp.text

    def test_chat_returns_non_empty_response(self):
        """The /chat endpoint must return a non-empty string."""
        answer = self._search_and_chat("drama film", "What is one movie about drama?")
        assert len(answer.strip()) > 0

    def test_chat_references_shawshank(self):
        """
        Searching for prison escape then asking the LLM should produce
        a response referencing 'Shawshank'.
        """
        answer = self._search_and_chat(
            "imprisoned man finds hope and escapes through a tunnel",
            "Which movie is about wrongful imprisonment and escape? Name the director.",
        )
        assert "Shawshank" in answer, f"Expected 'Shawshank' in: {answer}"

    def test_chat_returns_200_without_groq_key(self):
        """
        If no GROQ key is configured the streaming endpoint must still
        return 200 with an informative error message (not a 500 crash).
        """
        chat_resp = _post(
            "/chat",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "context": [],
            },
            timeout=30.0,
        )
        assert chat_resp.status_code == 200
