
#5.12+sub,Found 34434 unique entities and 1130 unique relations.
import argparse
import json
import torch
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.embedding_config import embedding_filename, get_embedding_config, known_embedding_aliases

# Patch for older sentence-transformers with new huggingface_hub
import huggingface_hub


def _parse_hf_resolve_url(url: str):
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "resolve" not in parts:
        raise ValueError(f"Unsupported Hugging Face URL format: {url}")

    resolve_idx = parts.index("resolve")
    if resolve_idx < 2 or resolve_idx + 2 >= len(parts):
        raise ValueError(f"Unsupported Hugging Face URL format: {url}")

    repo_id = "/".join(parts[:resolve_idx])
    revision = parts[resolve_idx + 1]
    filename = "/".join(parts[resolve_idx + 2 :])
    return repo_id, revision, filename


def _cached_download_compat(*args, **kwargs):
    url = kwargs.pop("url", None)
    if url is None and args:
        url = args[0]
        args = args[1:]

    cache_dir = kwargs.pop("cache_dir", None)
    force_filename = kwargs.pop("force_filename", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    kwargs.pop("legacy_cache_layout", None)

    if url is None:
        return huggingface_hub.hf_hub_download(
            *args,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=use_auth_token,
            **kwargs,
        )

    repo_id, revision, filename = _parse_hf_resolve_url(url)
    downloaded_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=use_auth_token,
        **kwargs,
    )

    if force_filename and cache_dir:
        target_path = os.path.join(cache_dir, force_filename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if not os.path.exists(target_path):
            shutil.copy2(downloaded_path, target_path)
        return target_path

    return downloaded_path


if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = _cached_download_compat

from sentence_transformers import SentenceTransformer

MODERNBERT_MIN_TRANSFORMERS = "4.48.0"
MODERNBERT_MIN_TOKENIZERS = "0.21.0"


def _require_modernbert_runtime():
    from packaging import version
    import tokenizers
    import transformers

    problems = []
    if version.parse(transformers.__version__) < version.parse(MODERNBERT_MIN_TRANSFORMERS):
        problems.append(f"transformers>={MODERNBERT_MIN_TRANSFORMERS} (current {transformers.__version__})")
    if version.parse(tokenizers.__version__) < version.parse(MODERNBERT_MIN_TOKENIZERS):
        problems.append(f"tokenizers>={MODERNBERT_MIN_TOKENIZERS} (current {tokenizers.__version__})")

    if problems:
        required = ", ".join(problems)
        raise RuntimeError(
            "Alibaba-NLP/gte-modernbert-base requires a newer Hugging Face runtime: "
            f"{required}. Upgrade with: "
            f"{sys.executable} -m pip install -U 'transformers>={MODERNBERT_MIN_TRANSFORMERS}' "
            f"'tokenizers>={MODERNBERT_MIN_TOKENIZERS}'"
        )


def _find_local_snapshot(candidates, required_files):
    for candidate in candidates:
        if candidate is None:
            continue
        if not candidate.exists():
            continue
        child_names = {path.name for path in candidate.iterdir()}
        if any(required_file in child_names for required_file in required_files):
            return str(candidate)

        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.is_dir():
            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                snapshot_child_names = {path.name for path in snapshot_dir.iterdir()}
                if any(required_file in snapshot_child_names for required_file in required_files):
                    return str(snapshot_dir)
    return None


def _cache_model_candidates(model_cache_name: str):
    candidates = []
    for env_name in ("LLAGA_HF_CACHE_ROOT", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        cache_root = Path(env_value)
        candidates.extend(
            [
                cache_root / model_cache_name,
                cache_root / "hub" / model_cache_name,
                cache_root,
            ]
        )

    candidates.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub" / model_cache_name,
            Path.home() / ".cache" / "torch" / "sentence_transformers" / model_cache_name,
        ]
    )
    return candidates


def resolve_model_name_or_path(embedding_type: str, explicit_model: Optional[str] = None) -> str:
    config = get_embedding_config(embedding_type)
    env_value = explicit_model or os.environ.get("LLAGA_EMBEDDING_MODEL") or os.environ.get(config.env_var)
    if env_value:
        return env_value

    if config.canonical_name == "E5-base":
        local_candidates = [
            *_cache_model_candidates("models--intfloat--multilingual-e5-base"),
            Path.home() / ".cache" / "torch" / "sentence_transformers" / "intfloat_multilingual-e5-base",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-base",
        ]
        local_model = _find_local_snapshot(local_candidates, ("config.json", "modules.json", "sentence_bert_config.json"))
        return local_model or config.model_name

    local_candidates = [
        *_cache_model_candidates("models--Alibaba-NLP--gte-modernbert-base"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Alibaba-NLP--gte-modernbert-base",
        Path.home() / ".cache" / "torch" / "sentence_transformers" / "Alibaba-NLP_gte-modernbert-base",
    ]
    local_model = _find_local_snapshot(local_candidates, ("config.json", "tokenizer.json", "tokenizer_config.json"))
    return local_model or config.model_name


class SentenceTransformerEncoder:
    def __init__(self, model_name_or_path: str, device: str):
        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            cache_folder=str(Path.home() / ".cache" / "torch" / "sentence_transformers"),
        )

    def encode(self, inputs, batch_size: int):
        if not inputs:
            return torch.tensor([])
        with torch.no_grad():
            embs = self.model.encode(
                inputs,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
        return embs.cpu()


class TransformerMeanPoolingEncoder:
    def __init__(self, model_name_or_path: str, device: str, max_length: int):
        _require_modernbert_runtime()

        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs, batch_size: int):
        if not inputs:
            return torch.tensor([])

        outputs = []
        for start in tqdm(range(0, len(inputs), batch_size), desc="Encoding"):
            batch = inputs[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                model_output = self.model(**encoded)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * attention_mask).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu())
        return torch.cat(outputs, dim=0)


def build_encoder(embedding_type: str, model_name_or_path: str, device: str, max_length: int):
    config = get_embedding_config(embedding_type)
    if config.canonical_name == "E5-base":
        return SentenceTransformerEncoder(model_name_or_path, device)
    if config.canonical_name == "Alibaba-NLP/gte-modernbert-base":
        return TransformerMeanPoolingEncoder(model_name_or_path, device, max_length=max_length)
    raise ValueError(f"Unsupported embedding type: {embedding_type}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Precompute reasoning-path embeddings for graph-conditioned training/eval."
    )
    parser.add_argument(
        "--embedding-type",
        default=os.environ.get("LLAGA_EMBEDDING_TYPE", "E5-base"),
        choices=known_embedding_aliases(),
        help="Encoder to use. Supports E5-base and Alibaba-NLP/gte-modernbert-base.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help=(
            "Override encoder model path/name. Env fallback: LLAGA_EMBEDDING_MODEL, "
            "LLAGA_E5_MODEL, or LLAGA_GTE_MODERNBERT_MODEL."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default="data/webqsp_may_merged",
        help="Directory containing train/test/validation JSON files.",
    )
    parser.add_argument(
        "--dataset",
        choices=("webqsp", "cwq"),
        default="webqsp",
        help="Dataset prefix to use in output embedding filenames.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def process_webqsp(args=None):
    args = build_parser().parse_args(args)
    config = get_embedding_config(args.embedding_type)
    base_dir = args.base_dir
    files = ["train.json", "test.json", "validation.json"]
    # files = [ "validation.json"]
    # output_file = os.path.join(base_dir, "webqsp_paths_e5.pt")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_or_path = resolve_model_name_or_path(args.embedding_type, args.model_name_or_path)
    print(f"Loading model {model_name_or_path} on {device}...")
    try:
        encoder = build_encoder(args.embedding_type, model_name_or_path, device, max_length=args.max_length)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Tip: set --model-name-or-path, LLAGA_EMBEDDING_MODEL, "
            f"or {config.env_var} to a local model directory if network download is unavailable."
        )
        raise SystemExit(1)

    # 1. First Pass: Collect unique entities and relations separately
    unique_entities = set()
    unique_relations = set()
    
    print("Scanning files for unique entities and relations...")
    for fname in files:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            paths = item.get("reasoning_paths", [])
            for path in paths:
                for triple in path:
                    if len(triple) >= 3:
                        unique_entities.add(triple[0]) # Head
                        unique_relations.add(triple[1]) # Relation
                        unique_entities.add(triple[2]) # Tail
    
    sorted_entities = sorted(list(unique_entities))
    sorted_relations = sorted(list(unique_relations))
    
    print(f"Found {len(sorted_entities)} unique entities and {len(sorted_relations)} unique relations.")
    
    if not sorted_entities and not sorted_relations:
        print("No strings found.")
        return

    # 2. Encode
    def encode_list(text_list, prefix):
        inputs = [f"{prefix}{s}" for s in text_list]
        return encoder.encode(inputs, batch_size=args.batch_size)

    print("Encoding entities...")
    entity_embs = encode_list(sorted_entities, "passage: entity: ")
    
    print("Encoding relations...")
    relation_embs = encode_list(sorted_relations, "passage: relation: ")
    
    # Maps
    entity_map = {s: entity_embs[i] for i, s in enumerate(sorted_entities)}
    relation_map = {s: relation_embs[i] for i, s in enumerate(sorted_relations)}
    
    # Cache for triples to save memory and stacking time
    # Key: (h, r, t) -> Value: Tensor shape (3, hidden_size)
    triple_cache = {}

    # 3. Second Pass: Build structure
    print("Constructing final structure...")
    
    for fname in files:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath): continue

        dataset_name = fname.replace(".json", "")
        cur_output_file = os.path.join(base_dir, embedding_filename(args.embedding_type, dataset_name, dataset=args.dataset))
        
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset_output = {}
            
        for item in tqdm(data, desc=f"Processing {fname}"):
            item_id = item["id"]
            paths = item.get("reasoning_paths", [])
            
            # List of paths (each path is a list of triple tensors)
            item_paths_list = []
            
            for path in paths:
                path_triples = []
                for triple in path:
                    if len(triple) >= 3:
                        h, r, t = triple[0], triple[1], triple[2]
                        triple_key = (h, r, t)
                        
                        if triple_key in triple_cache:
                            triple_emb = triple_cache[triple_key]
                        else:
                            # Re-encode check? No, we have maps.
                            # Need to ensure keys exist (they should from pass 1)
                            if h in entity_map and r in relation_map and t in entity_map:
                                triple_emb = torch.stack([
                                    entity_map[h],
                                    relation_map[r],
                                    entity_map[t]
                                ], dim=0)
                                triple_cache[triple_key] = triple_emb
                                path_triples.append(triple_emb)
                            else:
                                # Should typically not happen if file didn't change between passes
                                continue
                        
                        if triple_key in triple_cache: # Double check because of else block structure
                             # Only append if we successfully retrieved or created it
                             if len(path_triples) == 0 or path_triples[-1] is not triple_cache[triple_key]:
                                 path_triples.append(triple_cache[triple_key])
                
                # Append list of triples
                if path_triples:
                     item_paths_list.append(path_triples)
            
            if item_paths_list:
                dataset_output[item_id] = item_paths_list

        # 4. Save
        print(f"Saving {len(dataset_output)} items to {cur_output_file}...")
        torch.save(dataset_output, cur_output_file)

    print("Done.")

if __name__ == "__main__":
    process_webqsp()
