import json
import logging
import streamlit as st
import pandas as pd
import requests

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"

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
                st.image(image_url, use_column_width=True)
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
                                response = requests.post(f"{API_URL}/index", json=payload)
                                response.raise_for_status()
                                
                                st.success(f"Successfully built index for {len(st.session_state.indexed_data)} items!")
                            except Exception as e:
                                st.error(f"Failed to build index: {e}")
        except Exception as e:
            st.error(f"Error reading JSON file: {e}")
            logger.exception("Error processing file upload.")

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
    tab1, tab2 = st.tabs(["🔍 Standard Search", "🤖 AI Assistant (RAG)"])
    
    with tab1:
        query = st.text_input("Enter your search query...", placeholder="e.g., 'fresh organic apples'")
    
        if query or active_filters:
            with st.spinner("Searching..."):
                try:
                    payload = {
                        "query": query,
                        "filters": active_filters if active_filters else None,
                        "top_k": top_k
                    }
                    response = requests.post(f"{API_URL}/search", json=payload)
                    response.raise_for_status()
                    results = response.json().get("results", [])
                    
                    if not results:
                        st.info("No results found.")
                    else:
                        st.success(f"Found {len(results)} results")
                        for i, res in enumerate(results):
                            render_result_card(res['item'], res['score'], i + 1)
                except Exception as e:
                    st.error(f"Search API failed: {e}")

    with tab2:
        st.markdown("### Chat with your Catalog")
        st.caption("Powered by Groq and openai/gpt-oss-120b. The AI retrieves context from your active index dynamically.")
        
        # Display chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat input
        if chat_query := st.chat_input("Ask a question about the products..."):
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": chat_query})
            with st.chat_message("user"):
                st.markdown(chat_query)
                
            with st.chat_message("assistant"):
                # 1. Retrieve context silently
                with st.status("Retrieving context from catalog...", expanded=False) as status:
                    search_payload = {
                        "query": chat_query,
                        "filters": active_filters if active_filters else None,
                        "top_k": top_k
                    }
                    try:
                        resp = requests.post(f"{API_URL}/search", json=search_payload)
                        resp.raise_for_status()
                        results = resp.json().get("results", [])
                        context_items = [r['item'] for r in results]
                        status.update(label=f"Retrieved {len(context_items)} items for context.", state="complete", expanded=False)
                        
                        # Render the top 3 cards quickly so user sees what the AI is talking about
                        if results:
                            st.markdown("**Referenced Items:**")
                            for i, res in enumerate(results[:3]):
                                render_result_card(res['item'], res['score'], i + 1)
                    except Exception as e:
                        st.error(f"Context retrieval failed: {e}")
                        context_items = []
                
                # 2. Stream AI response
                chat_payload = {
                    "messages": st.session_state.chat_history,
                    "context": context_items
                }
                
                def stream_chat():
                    with requests.post(f"{API_URL}/chat", json=chat_payload, stream=True) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                yield chunk
                                
                try:
                    full_response = st.write_stream(stream_chat())
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Chat API failed: {e}")

else:
    st.info("👈 Please upload a JSON file and build the index to start searching.")
