"""
Tests for configuration management and constants.

The settings module was the site of two Pydantic v2 bugs (extra="ignore"
and load_dotenv before nested defaults). These tests serve as regression
guards. We also verify the singleton pattern, default values, and that
constants used by other modules have the expected values.
"""

import pytest

import sec_semantic_search.config.settings as settings_module
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

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------


class TestConstants:
    """Verify critical constants that other modules depend on."""

    def test_supported_forms(self):
        """SUPPORTED_FORMS must include base and amendment form types."""
        assert "8-K" in SUPPORTED_FORMS
        assert "10-K" in SUPPORTED_FORMS
        assert "10-Q" in SUPPORTED_FORMS
        assert "8-K/A" in SUPPORTED_FORMS
        assert "10-K/A" in SUPPORTED_FORMS
        assert "10-Q/A" in SUPPORTED_FORMS

    def test_base_forms(self):
        """BASE_FORMS contains only the non-amendment form types."""
        assert BASE_FORMS == ("8-K", "10-K", "10-Q")

    def test_amendment_forms(self):
        """AMENDMENT_FORMS contains only the /A variants."""
        assert AMENDMENT_FORMS == ("8-K/A", "10-K/A", "10-Q/A")

    def test_base_plus_amendment_equals_supported(self):
        """BASE_FORMS + AMENDMENT_FORMS should cover all SUPPORTED_FORMS."""
        assert set(BASE_FORMS) | set(AMENDMENT_FORMS) == set(SUPPORTED_FORMS)

    def test_embedding_dimension(self):
        """Must match google/embeddinggemma-300m's output dimension."""
        assert EMBEDDING_DIMENSION == 768

    def test_collection_name(self):
        assert COLLECTION_NAME == "sec_filings"

    def test_no_default_constants_shadow_settings(self):
        """Defaults live on the settings classes only; the copies here had
        drifted (DEFAULT_MAX_FILINGS was 500 against the 2,500 setting)."""
        import sec_semantic_search.config as config_package
        import sec_semantic_search.config.constants as constants_module

        shadowed = {
            "EMBEDDING_MODEL_NAME",
            "DEFAULT_CHUNK_TOKEN_LIMIT",
            "DEFAULT_CHUNK_TOLERANCE",
            "DEFAULT_CHROMADB_PATH",
            "DEFAULT_METADATA_DB_PATH",
            "DEFAULT_MAX_FILINGS",
            "DEFAULT_SEARCH_TOP_K",
            "DEFAULT_MIN_SIMILARITY",
        }
        for module in (constants_module, config_package):
            assert not shadowed & set(vars(module))

    def test_frontend_form_types_match(self):
        """The UI's form-type chips list exactly the forms the pipeline accepts."""
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "frontend/src/lib/ui-constants.ts"
        ).read_text()
        declared = re.search(r"export const FORM_TYPES = \[(.*?)\] as const;", source, re.S)
        assert declared is not None
        assert tuple(re.findall(r'"([^"]+)"', declared.group(1))) == SUPPORTED_FORMS

    def test_default_form_types_value(self):
        """DEFAULT_FORM_TYPES must list both supported forms."""
        assert DEFAULT_FORM_TYPES == "10-K,10-Q"

    def test_default_form_types_roundtrip(self):
        """DEFAULT_FORM_TYPES must be compatible with parse_form_types()."""
        assert parse_form_types(DEFAULT_FORM_TYPES) == ("10-K", "10-Q")


# -----------------------------------------------------------------------
# parse_form_types()
# -----------------------------------------------------------------------


class TestParseFormTypes:
    """parse_form_types() validates, normalises, deduplicates, and sorts."""

    def test_single_valid_form(self):
        assert parse_form_types("10-K") == ("10-K",)

    def test_both_forms(self):
        assert parse_form_types("10-K,10-Q") == ("10-K", "10-Q")

    def test_order_independence(self):
        """Input order should not affect the output."""
        assert parse_form_types("10-Q,10-K") == ("10-K", "10-Q")

    def test_case_insensitivity(self):
        assert parse_form_types("10-k") == ("10-K",)

    def test_whitespace_handling(self):
        assert parse_form_types("10-K , 10-Q") == ("10-K", "10-Q")

    def test_deduplication(self):
        assert parse_form_types("10-K,10-K") == ("10-K",)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty form type"):
            parse_form_types("")

    def test_8k_valid(self):
        assert parse_form_types("8-K") == ("8-K",)

    def test_all_three_forms(self):
        assert parse_form_types("8-K,10-K,10-Q") == ("10-K", "10-Q", "8-K")

    def test_amendment_10ka(self):
        assert parse_form_types("10-K/A") == ("10-K/A",)

    def test_amendment_10qa(self):
        assert parse_form_types("10-Q/A") == ("10-Q/A",)

    def test_amendment_8ka(self):
        assert parse_form_types("8-K/A") == ("8-K/A",)

    def test_base_and_amendment_together(self):
        result = parse_form_types("10-K,10-K/A")
        assert "10-K" in result
        assert "10-K/A" in result

    def test_all_six_forms(self):
        result = parse_form_types("8-K,8-K/A,10-K,10-K/A,10-Q,10-Q/A")
        assert len(result) == 6

    def test_invalid_form_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_form_types("20-F")

    def test_mixed_valid_invalid_raises(self):
        """Even one invalid form in a comma-separated list should fail."""
        with pytest.raises(ValueError, match="Unsupported"):
            parse_form_types("10-K,20-F")


# -----------------------------------------------------------------------
# Nested settings defaults
# -----------------------------------------------------------------------


class TestSettingsDefaults:
    """Nested settings classes should have sensible defaults."""

    def test_embedding_defaults(self, monkeypatch):
        """Code defaults should apply when env vars are absent."""
        monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
        monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
        monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)
        s = EmbeddingSettings()
        assert s.model_name == "google/embeddinggemma-300m"
        assert s.device == "auto"
        assert s.batch_size == 32

    def test_chunking_defaults(self):
        s = ChunkingSettings()
        assert s.token_limit == 500
        assert s.tolerance == 50

    def test_database_defaults(self):
        s = DatabaseSettings()
        assert s.chroma_path == "./data/chroma_db"
        assert s.metadata_db_path == "./data/metadata.sqlite"
        assert s.max_filings == 2500

    def test_search_defaults(self):
        s = SearchSettings()
        assert s.top_k == 5
        assert s.min_similarity == 0.0

    def test_hugging_face_defaults(self, monkeypatch):
        """Token should be None when no env var is set."""
        monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
        s = HuggingFaceSettings()
        assert s.token is None


# -----------------------------------------------------------------------
# Root Settings and singleton
# -----------------------------------------------------------------------


class TestRootSettings:
    """The root Settings class composes all nested settings."""

    def test_has_all_sections(self, monkeypatch):
        """Root Settings composes all nested settings sections.

        monkeypatch sets EDGAR credentials so the test does not depend
        on a real .env file.
        """
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
        s = Settings()
        assert hasattr(s, "edgar")
        assert hasattr(s, "embedding")
        assert hasattr(s, "chunking")
        assert hasattr(s, "database")
        assert hasattr(s, "search")
        assert hasattr(s, "hugging_face")

    def test_extra_ignore(self):
        """The extra='ignore' setting must be present to avoid rejecting
        prefixed env vars that belong to nested classes (regression test
        for Pydantic bug #1).
        """
        assert Settings.model_config.get("extra") == "ignore"


class TestSingleton:
    """get_settings() should return the same instance on repeated calls."""

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings_returns_new_instance(self):
        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2

    def test_reload_updates_global(self):
        """After reload, get_settings() should return the new instance."""
        s_old = get_settings()
        s_new = reload_settings()
        assert get_settings() is s_new
        assert get_settings() is not s_old
        # Restore original to avoid side effects on other tests
        settings_module._settings_instance = s_old


# -----------------------------------------------------------------------
# W5.1 — New configuration settings
# -----------------------------------------------------------------------


class TestEdgarSettingsW5:
    """EdgarSettings fields are now optional for web deployment scenarios."""

    def test_edgar_fields_optional(self, monkeypatch):
        """identity_name and identity_email default to None."""
        monkeypatch.delenv("EDGAR_IDENTITY_NAME", raising=False)
        monkeypatch.delenv("EDGAR_IDENTITY_EMAIL", raising=False)
        s = EdgarSettings()
        assert s.identity_name is None
        assert s.identity_email is None

    def test_edgar_fields_set(self, monkeypatch):
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
        s = EdgarSettings()
        assert s.identity_name == "Test User"
        assert s.identity_email == "test@example.com"


class TestDatabaseSettingsW5:
    """New privacy fields on DatabaseSettings."""

    def test_encryption_key_default_none(self, monkeypatch):
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        s = DatabaseSettings()
        assert s.encryption_key is None

    def test_encryption_key_set(self, monkeypatch):
        monkeypatch.setenv("DB_ENCRYPTION_KEY", "my-secret-key")
        s = DatabaseSettings()
        assert s.encryption_key == "my-secret-key"

    def test_task_history_retention_default(self, monkeypatch):
        monkeypatch.delenv("DB_TASK_HISTORY_RETENTION_DAYS", raising=False)
        s = DatabaseSettings()
        assert s.task_history_retention_days == 0

    def test_task_history_persist_tickers_default_false(self, monkeypatch):
        monkeypatch.delenv("DB_TASK_HISTORY_PERSIST_TICKERS", raising=False)
        s = DatabaseSettings()
        assert s.task_history_persist_tickers is False

    def test_task_history_persist_tickers_true(self, monkeypatch):
        monkeypatch.setenv("DB_TASK_HISTORY_PERSIST_TICKERS", "true")
        s = DatabaseSettings()
        assert s.task_history_persist_tickers is True

    # F5 mitigation: file-based encryption key loading
    def test_encryption_key_file_default_none(self, monkeypatch):
        monkeypatch.delenv("DB_ENCRYPTION_KEY_FILE", raising=False)
        s = DatabaseSettings()
        assert s.encryption_key_file is None

    def test_encryption_key_read_from_file(self, tmp_path, monkeypatch):
        """encryption_key_file is set → encryption_key resolved from file."""
        key_file = tmp_path / "secret.key"
        key_file.write_text("super-secret-key")
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", str(key_file))
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        s = DatabaseSettings()
        assert s.encryption_key == "super-secret-key"

    def test_encryption_key_file_strips_trailing_newline(self, tmp_path, monkeypatch):
        """Docker secrets append newline → should be stripped."""
        key_file = tmp_path / "secret.key"
        key_file.write_text("my-secret\n")
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", str(key_file))
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        s = DatabaseSettings()
        assert s.encryption_key == "my-secret"

    def test_encryption_key_file_and_direct_key_conflict(self, tmp_path, monkeypatch):
        """Both encryption_key and encryption_key_file set → ValueError."""
        key_file = tmp_path / "secret.key"
        key_file.write_text("secret")
        monkeypatch.setenv("DB_ENCRYPTION_KEY", "direct-key")
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", str(key_file))
        with pytest.raises(ValueError, match="mutually exclusive"):
            DatabaseSettings()

    def test_encryption_key_file_not_found(self, monkeypatch):
        """File path does not exist → ValueError."""
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", "/nonexistent/path/to/secret")
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="does not exist"):
            DatabaseSettings()

    def test_encryption_key_file_empty_content(self, tmp_path, monkeypatch):
        """File exists but is empty → ValueError."""
        key_file = tmp_path / "empty.key"
        key_file.write_text("")
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", str(key_file))
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="is empty"):
            DatabaseSettings()

    def test_encryption_key_file_empty_string_env_var(self, monkeypatch):
        """DB_ENCRYPTION_KEY_FILE='' (empty string) → no error, treated as unset."""
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", "")
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        s = DatabaseSettings()
        assert s.encryption_key is None
        assert s.encryption_key_file == ""


class TestLoggingSettings:
    """New LoggingSettings class for optional file logging."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LOG_FILE_PATH", raising=False)
        monkeypatch.delenv("LOG_FILE_MAX_BYTES", raising=False)
        monkeypatch.delenv("LOG_FILE_BACKUP_COUNT", raising=False)
        s = LoggingSettings()
        assert s.path is None
        assert s.max_bytes == 10_485_760
        assert s.backup_count == 3

    def test_file_path_set(self, monkeypatch):
        monkeypatch.setenv("LOG_FILE_PATH", "./logs/app.log")
        s = LoggingSettings()
        assert s.path == "./logs/app.log"

    def test_rotation_overrides(self, monkeypatch):
        monkeypatch.setenv("LOG_FILE_MAX_BYTES", "5242880")
        monkeypatch.setenv("LOG_FILE_BACKUP_COUNT", "5")
        s = LoggingSettings()
        assert s.max_bytes == 5_242_880
        assert s.backup_count == 5


class TestApiSettingsW5:
    """New W5 fields on ApiSettings."""

    def test_admin_key_default_none(self, monkeypatch):
        monkeypatch.delenv("API_ADMIN_KEY", raising=False)
        s = ApiSettings()
        assert s.admin_key is None

    def test_edgar_session_required_default_false(self, monkeypatch):
        monkeypatch.delenv("API_EDGAR_SESSION_REQUIRED", raising=False)
        s = ApiSettings()
        assert s.edgar_session_required is False

    def test_demo_mode_default_false(self, monkeypatch):
        monkeypatch.delenv("API_DEMO_MODE", raising=False)
        s = ApiSettings()
        assert s.demo_mode is False

    def test_demo_eviction_buffer_default(self, monkeypatch):
        monkeypatch.delenv("API_DEMO_EVICTION_BUFFER", raising=False)
        s = ApiSettings()
        assert s.demo_eviction_buffer == 500

    def test_max_task_queue_size_default(self, monkeypatch):
        monkeypatch.delenv("API_MAX_TASK_QUEUE_SIZE", raising=False)
        s = ApiSettings()
        assert s.max_task_queue_size == 5

    def test_max_task_queue_size_override(self, monkeypatch):
        monkeypatch.setenv("API_MAX_TASK_QUEUE_SIZE", "10")
        s = ApiSettings()
        assert s.max_task_queue_size == 10

    def test_abuse_prevention_defaults(self, monkeypatch):
        """All abuse prevention caps default to 0 (disabled)."""
        for var in (
            "API_MAX_TICKERS_PER_REQUEST",
            "API_MAX_FILINGS_PER_REQUEST",
            "API_INGEST_COOLDOWN_SECONDS",
            "API_MAX_TASK_DURATION_MINUTES",
        ):
            monkeypatch.delenv(var, raising=False)
        s = ApiSettings()
        assert s.max_tickers_per_request == 0
        assert s.max_filings_per_request == 0
        assert s.ingest_cooldown_seconds == 0
        assert s.max_task_duration_minutes == 0

    def test_abuse_prevention_overrides(self, monkeypatch):
        monkeypatch.setenv("API_MAX_TICKERS_PER_REQUEST", "100")
        monkeypatch.setenv("API_MAX_FILINGS_PER_REQUEST", "200")
        monkeypatch.setenv("API_INGEST_COOLDOWN_SECONDS", "60")
        monkeypatch.setenv("API_MAX_TASK_DURATION_MINUTES", "30")
        s = ApiSettings()
        assert s.max_tickers_per_request == 100
        assert s.max_filings_per_request == 200
        assert s.ingest_cooldown_seconds == 60
        assert s.max_task_duration_minutes == 30


class TestRootSettingsW5:
    """Root Settings now includes log_file section."""

    def test_has_log_file_section(self, monkeypatch):
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
        s = Settings()
        assert hasattr(s, "log_file")
        assert isinstance(s.log_file, LoggingSettings)
