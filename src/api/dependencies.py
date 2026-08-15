from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
import pylate.models.colbert as pylate_models
from src.core.search_engine import SearchEngine

# Monkey-patch pylate to fix sentence-transformers 5.x compatibility
def _patch_pylate():
    import json
    import os
    from pylate.models import Dense
    
    # 1. Patch Dense.load to filter out ST 5.x specific keys like module_input_name
    original_dense_load = Dense.load
    
    @staticmethod
    def patched_dense_load(input_path) -> "Dense":
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        from sentence_transformers.models.Dense import import_from_string
        config["activation_function"] = import_from_string(config["activation_function"])()
        
        _KNOWN_KEYS = {"in_features", "out_features", "bias", "activation_function", "init_weight", "init_bias"}
        filtered_config = {k: v for k, v in config.items() if k in _KNOWN_KEYS}
        model = Dense(**filtered_config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            from sentence_transformers.util import load_safetensors_model
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
            return model
        return model
    
    Dense.load = patched_dense_load

    # 2. Patch ColBERT.save to safely extract config dict
    original_save = pylate_models.ColBERT.save
    
    def patched_save(self, path: str, model_name: str | None = None, create_model_card: bool = True, train_datasets: list[str] | None = None, safe_serialization: bool = True):
        # We only override the file saving part by calling super then doing the JSON write manually
        super(pylate_models.ColBERT, self).save(
            path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )
        with open(os.path.join(path, "config_sentence_transformers.json"), "w") as fOut:
            try:
                config = self._get_model_config()
            except AttributeError:
                config = getattr(self, "_model_config", {})
            config = dict(config).copy()
            config["prompts"] = getattr(self, "prompts", {})
            config["default_prompt_name"] = getattr(self, "default_prompt_name", None)
            config["similarity_fn_name"] = getattr(self, "similarity_fn_name", None)
            config["query_prefix"] = getattr(self, "query_prefix", "query: ")
            config["document_prefix"] = getattr(self, "document_prefix", "document: ")
            config["query_length"] = getattr(self, "query_length", 32)
            config["document_length"] = getattr(self, "document_length", 256)
            config["attend_to_expansion_tokens"] = getattr(self, "attend_to_expansion_tokens", False)
            config["skiplist_words"] = getattr(self, "skiplist_words", [])
            json.dump(config, fOut, indent=2)
            
    pylate_models.ColBERT.save = patched_save

    # 3. Patch ColBERT.encode to fix _text_length
    original_encode = pylate_models.ColBERT.encode
    
    def patched_encode(self, sentences, *args, **kwargs):
        # We can't easily hook into the middle of encode.
        # But wait, we can just inject _text_length back into the class!
        if not hasattr(self, "_text_length"):
            def _text_length(x):
                return len(x) if isinstance(x, str) else sum(len(s) for s in x) if isinstance(x, list) else 1
            self._text_length = _text_length
        return original_encode(self, sentences, *args, **kwargs)
        
    pylate_models.ColBERT.encode = patched_encode

# All models are saved here on first download and loaded from here on every
# subsequent startup — no network access needed after the first run.
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

class AppState:
    engine: SearchEngine = None

app_state = AppState()

def init_search_engine():
    """Loads ML models from ./models/ (downloads once on first run) and
    initialises the SearchEngine singleton. Called once at server startup."""

    dense_path = MODELS_DIR / DENSE_MODEL_NAME.replace("/", "_")
    colbert_path = MODELS_DIR / COLBERT_MODEL_NAME.replace("/", "_")

    # --- Dense bi-encoder ---
    if dense_path.exists():
        logger.info(f"Loading dense model from local cache: {dense_path}")
        model = SentenceTransformer(str(dense_path))
    else:
        logger.info(f"Downloading dense model '{DENSE_MODEL_NAME}' for the first time...")
        model = SentenceTransformer(DENSE_MODEL_NAME)
        model.save(str(dense_path))
        logger.info(f"Dense model saved to {dense_path}")

    # --- ColBERT reranker ---
    if colbert_path.exists():
        logger.info(f"Loading ColBERT model from local cache: {colbert_path}")
        reranker = pylate_models.ColBERT(model_name_or_path=str(colbert_path))
    else:
        logger.info(f"Downloading ColBERT model '{COLBERT_MODEL_NAME}' for the first time (~500MB)...")
        from sentence_transformers.models import Transformer
        from pylate.models import Dense, ColBERT
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        
        # Manually load the transformer and add the custom Dense projection
        # This bypasses a bug where ST 5.x adds a Pooling layer and crashes pylate
        transformer = Transformer(COLBERT_MODEL_NAME)
        dense = Dense(in_features=768, out_features=128, bias=False)
        
        # Load the projection weights from the huggingface model cache
        model_path = hf_hub_download(repo_id=COLBERT_MODEL_NAME, filename='model.safetensors')
        tensors = safetensors.torch.load_file(model_path)
        dense.linear.weight.data = tensors['linear.weight']
        
        reranker = ColBERT(modules=[transformer, dense])
        reranker.save(str(colbert_path))
        logger.info(f"ColBERT model saved to {colbert_path}")

    app_state.engine = SearchEngine(model=model, reranker=reranker)
    logger.info("Search Engine initialised successfully.")

def get_search_engine() -> SearchEngine:
    """Dependency injection provider for the SearchEngine singleton."""
    if app_state.engine is None:
        raise RuntimeError("Search Engine is not initialised")
    return app_state.engine
