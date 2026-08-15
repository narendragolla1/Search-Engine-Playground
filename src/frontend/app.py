import json
import streamlit as st
import httpx
import base64
from typing import Dict, Any, Optional
from src.core.config import settings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Replace requests with httpx client
API_URL = settings.api_url
# Use a timeout for httpx
HTTP_TIMEOUT = 10.0

# Decorator to retry network requests automatically
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
def fetch_api(method: str, endpoint: str, **kwargs):
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.request(method, f"{API_URL}{endpoint}", **kwargs)
        response.raise_for_status()
        return response

# --- UI Components ---
def render_result_card(item: dict, score: float, rank: int):
    """Dynamically renders a JSON object as a beautiful UI card."""
    # 1. Try to find a Title
    title_keys = ['Title', 'Name', 'Headline', 'Subject', 'title', 'name']
    title = next((str(item[k]) for k in title_keys if k in item), f"Result #{rank}")
    
    # 2. Try to find an Image URL
    image_url = None
    for k, v in item.items():
        if isinstance(v, str) and v.startswith('http') and any(ext in v.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', 'm.media-amazon.com']):
            image_url = v
            break
            
    # 3. Try to find a long description field
    desc_key = None
    max_len = 0
    for k, v in item.items():
        if isinstance(v, str) and len(v) > max_len and k not in title_keys and v != image_url:
            max_len = len(v)
            desc_key = k

    with st.container(border=True):
        if image_url:
            cols = st.columns([1, 4])
            with cols[0]:
                st.image(image_url, use_container_width=True)
            text_col = cols[1]
        else:
            text_col = st.container()
            
        with text_col:
            # Header
            st.subheader(title)
            st.caption(f"Rank: #{rank} | RRF Score: `{score:.4f}`")
            
            # Show a snippet of the longest text field
            if desc_key and max_len > 40:
                desc = item[desc_key]
                st.markdown(f"*{desc[:250]}{'...' if len(desc) > 250 else ''}*")
                
            # Remaining data in an expander
            with st.expander("View full JSON data"):
                st.json(item)

# --- Helper Functions ---
def infer_field_types(data):
    numeric_fields = {} # field -> {'min': float, 'max': float}
    categorical_fields = {} # field -> set of unique values
    
    for item in data[:200]: # analyze up to 200 items
        if not isinstance(item, dict): continue
        for k, v in item.items():
            if v is None or v == "N/A": continue
            
            if isinstance(v, (int, float)):
                if k not in numeric_fields:
                    numeric_fields[k] = {'min': float(v), 'max': float(v)}
                else:
                    numeric_fields[k]['min'] = min(numeric_fields[k]['min'], float(v))
                    numeric_fields[k]['max'] = max(numeric_fields[k]['max'], float(v))
            elif isinstance(v, str):
                try:
                    # attempt to parse as float (removing commas and checking)
                    float_val = float(v.replace(',', ''))
                    if k not in numeric_fields:
                        numeric_fields[k] = {'min': float_val, 'max': float_val}
                    else:
                        numeric_fields[k]['min'] = min(numeric_fields[k]['min'], float_val)
                        numeric_fields[k]['max'] = max(numeric_fields[k]['max'], float_val)
                except ValueError:
                    # It's a string category
                    if k not in categorical_fields:
                        categorical_fields[k] = set()
                    
                    if ',' in v:
                        parts = [p.strip() for p in v.split(',')]
                        categorical_fields[k].update(parts)
                    else:
                        categorical_fields[k].add(v)
            elif isinstance(v, list):
                if k not in categorical_fields:
                    categorical_fields[k] = set()
                for cat in v:
                    if isinstance(cat, str):
                        categorical_fields[k].add(cat)
                        
    # Filter categories to only those with < 50 unique values (ignore descriptions)
    valid_categorical = {k: sorted(list(v)) for k, v in categorical_fields.items() if 0 < len(v) < 50}
    return numeric_fields, valid_categorical

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Search Engine Playground",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Search Engine Playground")
st.markdown("""
Upload a JSON file containing an array of objects. Select the fields you want to index, 
and then experiment with the hybrid search combining exact keyword matching (BM25) and semantic similarity.
""")

# --- Session State Initialization ---
if 'indexed_data' not in st.session_state:
    st.session_state.indexed_data = None
if 'available_fields' not in st.session_state:
    st.session_state.available_fields = []
if 'numeric_fields' not in st.session_state:
    st.session_state.numeric_fields = {}
if 'categorical_fields' not in st.session_state:
    st.session_state.categorical_fields = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar UI ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload JSON file", type=['json'])

    if uploaded_file is not None:
        try:
            # Read and parse JSON
            raw_data = json.load(uploaded_file)
            data = None
            
            # Handle case where JSON is a dict containing a list (e.g., {"movies": [...]})
            if isinstance(raw_data, dict):
                list_values = [v for v in raw_data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    data = list_values[0]
                    st.info("Automatically extracted the list of objects from the JSON file.")
                else:
                    st.error("Uploaded JSON is an object. Please upload a file containing a single list of objects.")
            elif isinstance(raw_data, list):
                data = raw_data
            else:
                st.error("Uploaded JSON must be a list of objects.")

            if data is not None:
                if len(data) == 0:
                    st.warning("The JSON list is empty.")
                else:
                    st.session_state.indexed_data = data
                    # Extract all possible keys from the first few items
                    keys = set()
                    for item in data[:10]:
                        if isinstance(item, dict):
                            keys.update(item.keys())
                    st.session_state.available_fields = sorted(list(keys))
                    
                    # Infer types for filtering
                    num_f, cat_f = infer_field_types(data)
                    st.session_state.numeric_fields = num_f
                    st.session_state.categorical_fields = cat_f

                # Field Selection
                st.header("2. Configure Index")
                searchable_fields = st.multiselect(
                    "Select fields to search across:",
                    st.session_state.available_fields,
                    default=st.session_state.available_fields[:2] if len(st.session_state.available_fields) >= 2 else st.session_state.available_fields
                )

                if st.button("Build Index"):
                    if not searchable_fields:
                        st.error("Please select at least one field to index.")
                    else:
                        with st.spinner("Indexing data in backend API... This may take a moment."):
                            try:
                                # Call FastAPI /index
                                payload = {
                                    "data": st.session_state.indexed_data,
                                    "searchable_fields": searchable_fields
                                }
                                resp = fetch_api("POST", "/index", json=payload)
                                st.success(f"Successfully built index for {resp.json().get('items_indexed')} items!")
                            except Exception as e:
                                logger.error(f"Failed to build index: {e}")
                                st.error(f"API Error: {e}")
        except Exception as e:
            logger.exception("Error processing file upload.")
            st.error(f"Error reading JSON file: {e}")

    st.divider()
    st.header("3. Hyperparameters")
    
    top_k = st.number_input(
        "Number of Results (Top K)",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    st.divider()
    st.header("4. Dynamic Filters")
    active_filters = {}
    
    if st.session_state.indexed_data is not None:
        if st.session_state.numeric_fields:
            st.subheader("Numeric")
            for field, bounds in st.session_state.numeric_fields.items():
                if bounds['min'] < bounds['max']: # only show slider if there's a range
                    selected_range = st.slider(
                        field,
                        min_value=float(bounds['min']),
                        max_value=float(bounds['max']),
                        value=(float(bounds['min']), float(bounds['max']))
                    )
                    if selected_range[0] > bounds['min'] or selected_range[1] < bounds['max']:
                        active_filters[field] = {'min': selected_range[0], 'max': selected_range[1]}

        if st.session_state.categorical_fields:
            st.subheader("Categories")
            for field, options in st.session_state.categorical_fields.items():
                selected_cats = st.multiselect(f"{field}", options)
                if selected_cats:
                    active_filters[field] = selected_cats
    else:
        st.info("Index data to see filters.")

# --- Main UI Area ---
if st.session_state.indexed_data is not None:
    st.subheader("Unified Search")
    
    # Toggle for AI Overview
    enable_ai = st.toggle("Enable ✨ AI Overview (RAG)", value=True)
    
    query = st.text_input("Enter your search query...", placeholder="e.g., 'An American classic mafia film'")
    
    if query or active_filters:
        with st.spinner("Searching..."):
            try:
                # 1. Fetch Search Results
                payload = {
                    "query": query,
                    "filters": active_filters if active_filters else None,
                    "top_k": top_k
                }
                response = fetch_api("POST", "/search", json=payload)
                results = response.json().get("results", [])
                
                if not results:
                    st.info("No results found.")
                else:
                    st.success(f"Found {len(results)} results")
                    
                    # 2. If AI Overview is enabled, stream the RAG response
                    if enable_ai and query:
                        st.markdown("### ✨ AI Overview")
                        context_items = [r['item'] for r in results]
                        
                        # Use a single turn message for the unified search bar overview
                        chat_payload = {
                            "messages": [{"role": "user", "content": query}],
                            "context": context_items
                        }
                        
                        def stream_chat():
                            with httpx.stream("POST", f"{API_URL}/chat", json=chat_payload, timeout=HTTP_TIMEOUT) as r:
                                r.raise_for_status()
                                for chunk in r.iter_text():
                                    if chunk:
                                        yield chunk
                                        
                        try:
                            # Stream the response into a container
                            with st.container(border=True):
                                st.write_stream(stream_chat())
                        except Exception as e:
                            logger.exception("AI Overview failed")
                            st.error(f"AI Overview failed: {e}")
                            
                        st.divider()

                    # 3. Display Standard Results
                    st.markdown(f"**Found {len(results)} results**")
                    for i, res in enumerate(results):
                        render_result_card(res['item'], res['score'], i + 1)
            except Exception as e:
                st.error(f"Search API failed: {e}")

else:
    st.info("👈 Please upload a JSON file and build the index to start searching.")
