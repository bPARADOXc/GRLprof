from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Union


@dataclass(frozen=True)
class EmbeddingConfig:
    canonical_name: str
    safe_name: str
    hidden_size: int
    webqsp_file_prefix: str
    model_name: str
    env_var: str
    aliases: tuple[str, ...]


_CONFIGS = (
    EmbeddingConfig(
        canonical_name="E5-base",
        safe_name="E5-base",
        hidden_size=768,
        webqsp_file_prefix="webqsp_paths_e5",
        model_name="intfloat/multilingual-e5-base",
        env_var="LLAGA_E5_MODEL",
        aliases=("E5-base", "e5-base", "e5", "intfloat/multilingual-e5-base"),
    ),
    EmbeddingConfig(
        canonical_name="Alibaba-NLP/gte-modernbert-base",
        safe_name="gte-modernbert-base",
        hidden_size=768,
        webqsp_file_prefix="webqsp_paths_gte_modernbert_base",
        model_name="Alibaba-NLP/gte-modernbert-base",
        env_var="LLAGA_GTE_MODERNBERT_MODEL",
        aliases=(
            "Alibaba-NLP/gte-modernbert-base",
            "gte-modernbert-base",
            "GTE-modernbert-base",
            "gte-modernbert",
            "GTE-modernbert",
        ),
    ),
)

_ALIASES: Dict[str, EmbeddingConfig] = {}
for config in _CONFIGS:
    for alias in config.aliases:
        _ALIASES[alias] = config
        _ALIASES[alias.lower()] = config


def known_embedding_types() -> list[str]:
    return [config.canonical_name for config in _CONFIGS]


def known_embedding_aliases() -> list[str]:
    aliases = []
    for config in _CONFIGS:
        aliases.extend(config.aliases)
    return aliases


def get_embedding_config(embedding_type: str) -> EmbeddingConfig:
    key = str(embedding_type).strip()
    config = _ALIASES.get(key) or _ALIASES.get(key.lower())
    if config is None:
        allowed = ", ".join(known_embedding_types())
        raise ValueError(f"Unsupported pretrained_embedding_type={embedding_type!r}. Supported: {allowed}")
    return config


def normalize_embedding_type(embedding_type: str) -> str:
    return get_embedding_config(embedding_type).canonical_name


def embedding_safe_name(embedding_type: str) -> str:
    return get_embedding_config(embedding_type).safe_name


def embedding_hidden_size(embedding_type: str) -> int:
    return get_embedding_config(embedding_type).hidden_size


def webqsp_embedding_filename(embedding_type: str, split: str) -> str:
    return embedding_filename(embedding_type, split, dataset="webqsp")


def embedding_filename(embedding_type: str, split: str, dataset: str = "webqsp") -> str:
    config = get_embedding_config(embedding_type)
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "webqsp":
        file_prefix = config.webqsp_file_prefix
    elif config.webqsp_file_prefix.startswith("webqsp_paths_"):
        file_prefix = f"{dataset_key}_paths_{config.webqsp_file_prefix[len('webqsp_paths_'):]}"
    else:
        file_prefix = f"{dataset_key}_{config.webqsp_file_prefix}"
    return f"{file_prefix}_{split}.pt"


def embedding_path(data_dir: Union[str, Path], embedding_type: str, split: str, dataset: str = "webqsp") -> Path:
    return Path(data_dir) / embedding_filename(embedding_type, split, dataset=dataset)


def webqsp_embedding_path(data_dir: Union[str, Path], embedding_type: str, split: str) -> Path:
    return embedding_path(data_dir, embedding_type, split, dataset="webqsp")


def format_embedding_choices(aliases: bool = False) -> str:
    values: Iterable[str]
    values = known_embedding_aliases() if aliases else known_embedding_types()
    return ", ".join(values)
