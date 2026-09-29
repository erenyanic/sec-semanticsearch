"""Configuration module — settings and constants."""

from sec_semantic_search.config.constants import (
    AMENDMENT_FORMS,
    BASE_FORMS,
    COLLECTION_NAME,
    DEFAULT_FORM_TYPES,
    EMBEDDING_DIMENSION,
    SUPPORTED_FORMS,
    parse_form_types,
)
from sec_semantic_search.config.settings import (
    ApiSettings,
    ChunkingSettings,
    DatabaseSettings,
    EdgarSettings,
    EmbeddingSettings,
    HuggingFaceSettings,
    LoggingSettings,
    SearchSettings,
    Settings,
    get_settings,
    reload_settings,
)

__all__ = [
    # Constants
    "SUPPORTED_FORMS",
    "BASE_FORMS",
    "AMENDMENT_FORMS",
    "DEFAULT_FORM_TYPES",
    "parse_form_types",
    "EMBEDDING_DIMENSION",
    "COLLECTION_NAME",
    # Settings
    "ApiSettings",
    "Settings",
    "EdgarSettings",
    "EmbeddingSettings",
    "ChunkingSettings",
    "DatabaseSettings",
    "SearchSettings",
    "HuggingFaceSettings",
    "LoggingSettings",
    "get_settings",
    "reload_settings",
]
