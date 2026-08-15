import json
import logging
import streamlit as st
import pandas as pd

from search_engine import SearchEngine
from sentence_transformers import SentenceTransformer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# --- Global Model Loading ---
@st.cache_resource
def load_nlp_model(model_name: str = "all-MiniLM-L6-v2"):
    """Loads the heavy NLP model once globally to speed up re-runs."""
    logger.info(f"Loading SentenceTransformer model globally: {model_name}")
    return SentenceTransformer(model_name)

# Ensure the model is loaded immediately when the server starts or script runs
nlp_model = load_nlp_model()

# --- Session State Initialization ---
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = SearchEngine(model=nlp_model)
if 'indexed_data' not in st.session_state:
    st.session_state.indexed_data = None
if 'available_fields' not in st.session_state:
    st.session_state.available_fields = []

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
                    # Extract all possible keys from the first few items
                    keys = set()
                    for item in data[:10]:
                        if isinstance(item, dict):
                            keys.update(item.keys())
                    st.session_state.available_fields = sorted(list(keys))

                # Field Selection
                st.header("2. Configure Index")
                selected_fields = st.multiselect(
                    "Select fields to search across:",
                    st.session_state.available_fields,
                    default=st.session_state.available_fields[:2] if len(st.session_state.available_fields) >= 2 else st.session_state.available_fields
                )

                if st.button("Build Index"):
                    if not selected_fields:
                        st.error("Please select at least one field to index.")
                    else:
                        with st.spinner("Indexing data..."):
                            st.session_state.search_engine.index(data, selected_fields)
                            st.session_state.indexed_data = data
                        st.success(f"Successfully indexed {len(data)} items!")
        except Exception as e:
            st.error(f"Error reading JSON file: {e}")
            logger.exception("Error processing file upload.")

    st.divider()
    st.header("3. Hyperparameters")
    
    alpha = st.slider(
        "Hybrid Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 = Semantic Search Only\n1.0 = Keyword Search Only\n0.5 = Equal Weight"
    )
    
    top_k = st.number_input(
        "Number of Results (Top K)",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

# --- Main Search UI ---
if st.session_state.indexed_data is not None:
    query = st.text_input("Enter your search query...", placeholder="e.g., 'fresh organic apples'")

    if query:
        with st.spinner("Searching..."):
            results = st.session_state.search_engine.search(query, alpha=alpha, top_k=top_k)
        
        if not results:
            st.info("No results found.")
        else:
            st.subheader(f"Top {len(results)} Results")
            
            # Display results in a clean way
            for i, res in enumerate(results):
                with st.container(border=True):
                    # Header with score
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"**Result #{i+1}** (Combined Score: `{res.score:.4f}`)")
                    with cols[1]:
                        st.caption(f"Kw: `{res.keyword_score:.2f}` | Sem: `{res.semantic_score:.2f}`")
                    
                    # Display the actual item data as a formatted dictionary or dataframe
                    st.json(res.item, expanded=False)
else:
    st.info("👈 Please upload a JSON file and build the index to start searching.")
