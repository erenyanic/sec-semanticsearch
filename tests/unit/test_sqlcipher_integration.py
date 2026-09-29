"""
Tests for SQLCipher integration in MetadataRegistry.

Covers:
- Unencrypted fallback (default, no key set)
- Encrypted mode via mocked pysqlcipher3
- Warning when key is set but pysqlcipher3 is unavailable
- _get_sqlite_module helper logic
- encrypted property
- PRAGMA key execution on connect
- Full round-trip (register, query, delete) through encrypted mock
"""

import sqlite3
import types
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sec_semantic_search.core.types import FilingIdentifier
from sec_semantic_search.database.metadata import MetadataRegistry, _get_sqlite_module

# ---------------------------------------------------------------------------
# _get_sqlite_module helper
# ---------------------------------------------------------------------------


class TestGetSqliteModule:
    """Unit tests for the _get_sqlite_module() helper."""

    def test_returns_sqlite3_when_no_key(self):
        """No encryption key → standard sqlite3 module."""
        module = _get_sqlite_module(None)
        assert module is sqlite3

    def test_returns_sqlite3_when_empty_key(self):
        """Empty string encryption key → treated as unset."""
        module = _get_sqlite_module("")
        assert module is sqlite3

    def test_returns_sqlcipher_when_key_and_installed(self):
        """Key set + pysqlcipher3 installed → sqlcipher module."""
        fake_sqlcipher = types.ModuleType("pysqlcipher3.dbapi2")
        fake_sqlcipher.connect = MagicMock()
        fake_sqlcipher.Error = type("Error", (Exception,), {})
        fake_sqlcipher.IntegrityError = type("IntegrityError", (Exception,), {})

        fake_parent = types.ModuleType("pysqlcipher3")
        fake_parent.dbapi2 = fake_sqlcipher

        with patch.dict(
            "sys.modules",
            {
                "pysqlcipher3": fake_parent,
                "pysqlcipher3.dbapi2": fake_sqlcipher,
            },
        ):
            module = _get_sqlite_module("my-secret-key")
            assert module is fake_sqlcipher

    def test_falls_back_with_warning_when_key_but_not_installed(self):
        """Key set but pysqlcipher3 not installed → fallback to sqlite3 with warning."""
        with patch("sec_semantic_search.database.metadata.logger") as mock_logger:
            with patch.dict("sys.modules", {"pysqlcipher3": None, "pysqlcipher3.dbapi2": None}):
                module = _get_sqlite_module("my-secret-key")
                assert module is sqlite3
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "pysqlcipher3 is not installed" in warning_msg


# ---------------------------------------------------------------------------
# MetadataRegistry — unencrypted mode (default)
# ---------------------------------------------------------------------------


class TestUnencryptedFallback:
    """MetadataRegistry with no encryption key uses plain sqlite3."""

    @pytest.fixture
    def registry(self, tmp_db_path):
        # Explicit empty key overrides any DB_ENCRYPTION_KEY in .env,
        # ensuring this test exercises the unencrypted path regardless
        # of the local environment.
        return MetadataRegistry(db_path=tmp_db_path, encryption_key="")

    def test_encrypted_property_false(self, registry):
        assert registry.encrypted is False

    def test_uses_sqlite3_module(self, registry):
        assert registry._sqlite_module is sqlite3

    def test_full_roundtrip(self, registry, sample_filing_id):
        """Register, query, and delete a filing — standard unencrypted path."""
        registry.register_filing(sample_filing_id, chunk_count=42)
        assert registry.count() == 1
        record = registry.get_filing(sample_filing_id.accession_number)
        assert record is not None
        assert record.ticker == "AAPL"
        assert record.chunk_count == 42

        registry.remove_filing(sample_filing_id.accession_number)
        assert registry.count() == 0


# ---------------------------------------------------------------------------
# MetadataRegistry — encrypted mode (mocked pysqlcipher3)
# ---------------------------------------------------------------------------


class _FakeSqlCipherConnection:
    """Wrapper around a real sqlite3.Connection that silently accepts PRAGMA key.

    Python 3.13 makes ``sqlite3.Connection.execute`` read-only, so we
    cannot monkey-patch it.  This wrapper delegates everything to the real
    connection but intercepts ``PRAGMA key`` calls (which plain sqlite3
    does not understand).
    """

    def __init__(self, real_conn: sqlite3.Connection) -> None:
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, "_conn", real_conn)

    def execute(self, sql: str, *args, **kwargs):
        if isinstance(sql, str) and sql.strip().upper().startswith("PRAGMA KEY"):
            return self._conn.execute("SELECT 1")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        # Delegate attribute writes (e.g. row_factory) to the real connection
        if name == "_conn":
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)


def _make_fake_sqlcipher_module():
    """Create a fake pysqlcipher3.dbapi2 module backed by real sqlite3.

    Returns connections wrapped in ``_FakeSqlCipherConnection`` so that
    ``PRAGMA key`` is silently accepted.  This lets us test the encrypted
    code path without the native sqlcipher library.
    """
    mod = types.ModuleType("pysqlcipher3.dbapi2")

    def _fake_connect(*args, **kwargs):
        real_conn = sqlite3.connect(*args, **kwargs)
        return _FakeSqlCipherConnection(real_conn)

    mod.connect = _fake_connect
    mod.Error = sqlite3.Error
    mod.IntegrityError = sqlite3.IntegrityError
    mod.Row = sqlite3.Row
    return mod


class TestEncryptedMode:
    """MetadataRegistry with encryption key and mocked pysqlcipher3.

    Since pysqlcipher3 may not be installed in the dev environment, these
    tests use a real sqlite3 connection behind a mock module facade.  This
    verifies the code path (PRAGMA key, module selection, exception routing)
    without requiring the native library.
    """

    @pytest.fixture
    def fake_sqlcipher_module(self):
        return _make_fake_sqlcipher_module()

    @pytest.fixture
    def registry(self, tmp_db_path, fake_sqlcipher_module):
        """Registry using fake sqlcipher module."""
        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=fake_sqlcipher_module,
        ):
            reg = MetadataRegistry(
                db_path=tmp_db_path,
                encryption_key="test-encryption-key",
            )
        return reg

    def test_encrypted_property_true(self, registry):
        assert registry.encrypted is True

    def test_full_roundtrip_encrypted(self, registry):
        """Register, query, list, statistics, delete — encrypted mode."""
        fid = FilingIdentifier("MSFT", "10-Q", date(2024, 6, 1), "ACC-ENC-1")
        registry.register_filing(fid, chunk_count=100)

        assert registry.count() == 1
        record = registry.get_filing("ACC-ENC-1")
        assert record is not None
        assert record.ticker == "MSFT"
        assert record.chunk_count == 100

        filings = registry.list_filings(ticker="MSFT")
        assert len(filings) == 1

        stats = registry.get_statistics()
        assert stats.filing_count == 1
        assert stats.tickers == ["MSFT"]

        registry.remove_filing("ACC-ENC-1")
        assert registry.count() == 0

    def test_duplicate_detection_encrypted(self, registry):
        """Duplicate lookup and register_filing_if_new work in encrypted mode."""
        fid = FilingIdentifier("GOOGL", "10-K", date(2024, 1, 1), "ACC-ENC-DUP")
        assert registry.get_existing_accessions(["ACC-ENC-DUP"]) == set()
        registry.register_filing(fid, chunk_count=50)
        assert registry.get_existing_accessions(["ACC-ENC-DUP"]) == {"ACC-ENC-DUP"}

        result = registry.register_filing_if_new(fid, chunk_count=50)
        assert result is False
        assert registry.count() == 1

    def test_task_history_encrypted(self, registry):
        """Task history operations work through encrypted connection."""
        registry.save_task_history(
            "task-enc-1",
            status="completed",
            tickers=["AAPL"],
            form_types=["10-K"],
            results=[{"accession": "ACC-1", "status": "done"}],
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:01:00",
            filings_done=1,
        )

        task = registry.get_task_history("task-enc-1")
        assert task is not None
        assert task["status"] == "completed"
        # Tickers stripped by default (TASK_HISTORY_PERSIST_TICKERS=false).
        assert task["tickers"] == []

    def test_thread_safety_encrypted(self, registry):
        """Concurrent writes from multiple threads work in encrypted mode."""
        import threading

        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                fid = FilingIdentifier(
                    "AAPL",
                    "10-K",
                    date(2020 + i, 1, 1),
                    f"ACC-ENC-THR-{i}",
                )
                registry.register_filing(fid, chunk_count=i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert registry.count() == 10


# ---------------------------------------------------------------------------
# PRAGMA key execution
# ---------------------------------------------------------------------------


class TestPragmaKeyExecution:
    """Verify PRAGMA key is executed immediately on encrypted connections."""

    def test_pragma_key_called_on_connect(self, tmp_db_path):
        """PRAGMA key should be the first statement after connect."""
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=MagicMock(fetchone=MagicMock(return_value=None)))

        fake_module = types.ModuleType("pysqlcipher3.dbapi2")
        fake_module.connect = MagicMock(return_value=mock_conn)
        fake_module.Error = type("Error", (Exception,), {})
        fake_module.IntegrityError = type("IntegrityError", (Exception,), {})
        fake_module.Row = sqlite3.Row

        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=fake_module,
        ):
            try:
                MetadataRegistry(
                    db_path=tmp_db_path,
                    encryption_key="my-secret-key",
                )
            except Exception:
                pass  # May fail on table creation with mock — that's fine

        # The very first execute call should be PRAGMA key with hex-encoded blob
        first_call = mock_conn.execute.call_args_list[0]
        assert "PRAGMA key" in first_call.args[0]
        expected_hex = b"my-secret-key".hex()
        assert expected_hex in first_call.args[0]

    def test_pragma_key_not_called_without_encryption(self, tmp_db_path):
        """PRAGMA key should NOT be executed when no encryption key is set."""
        registry = MetadataRegistry(db_path=tmp_db_path, encryption_key="")
        # Check that no PRAGMA key was issued by verifying we can still
        # query the database normally (if PRAGMA key was issued on plain
        # sqlite3, it would be treated as a no-op or error)
        assert registry.count() == 0
        registry.close()


# ---------------------------------------------------------------------------
# Key set but pysqlcipher3 unavailable — graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """When DB_ENCRYPTION_KEY is set but pysqlcipher3 is not installed."""

    def test_falls_back_to_unencrypted_with_warning(self, tmp_db_path):
        """Registry should still work, using plain sqlite3, with a warning."""
        with patch("sec_semantic_search.database.metadata.logger") as mock_logger:
            with patch.dict(
                "sys.modules",
                {"pysqlcipher3": None, "pysqlcipher3.dbapi2": None},
            ):
                registry = MetadataRegistry(
                    db_path=tmp_db_path,
                    encryption_key="my-secret-key",
                )

        assert registry.encrypted is False
        assert registry._sqlite_module is sqlite3
        # Verify the warning was logged
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "pysqlcipher3 is not installed" in str(call)
        ]
        assert len(warning_calls) == 1

        # Should still work for basic operations
        fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-DEGRADE")
        registry.register_filing(fid, chunk_count=10)
        assert registry.count() == 1
        registry.close()


# ---------------------------------------------------------------------------
# Encryption key handling edge cases
# ---------------------------------------------------------------------------


class TestEncryptionKeyEdgeCases:
    """Edge cases around encryption key values."""

    def test_none_key_defers_to_settings(self, tmp_db_path):
        """encryption_key=None defers to settings.database.encryption_key.

        When DB_ENCRYPTION_KEY is set in .env, passing None still results
        in an encrypted database — but only when pysqlcipher3 can actually
        be imported.  If the extra is missing, or installed without the
        libsqlcipher shared library, ``_get_sqlite_module`` warns and falls
        back to plain sqlite3, so the expectation has to hold in both
        environments.  To explicitly disable encryption regardless of
        settings, pass encryption_key="".
        """
        from sec_semantic_search.config import get_settings

        # Probe the same way ``_get_sqlite_module`` does: importlib.find_spec
        # is not enough, because the package can be installed while its
        # compiled extension fails to load (missing libsqlcipher).
        try:
            from pysqlcipher3 import dbapi2 as _sqlcipher  # noqa: F401

            sqlcipher_usable = True
        except ImportError:
            sqlcipher_usable = False

        settings_key = get_settings().database.encryption_key
        registry = MetadataRegistry(db_path=tmp_db_path, encryption_key=None)
        expected_encrypted = bool(settings_key) and sqlcipher_usable
        assert registry.encrypted is expected_encrypted
        registry.close()

    def test_explicit_empty_string_key_means_unencrypted(self, tmp_db_path):
        """Empty string should be treated the same as None."""
        registry = MetadataRegistry(db_path=tmp_db_path, encryption_key="")
        assert registry.encrypted is False
        registry.close()

    def test_key_with_special_characters(self, tmp_db_path):
        """Keys with special characters should not cause SQL injection."""
        fake_module = _make_fake_sqlcipher_module()

        dangerous_key = "'; DROP TABLE filings; --"
        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=fake_module,
        ):
            registry = MetadataRegistry(
                db_path=tmp_db_path,
                encryption_key=dangerous_key,
            )
        # Key is hex-encoded, not interpolated — tables should exist
        assert registry.count() == 0
        registry.close()


# ---------------------------------------------------------------------------
# WAL mode with encrypted connection
# ---------------------------------------------------------------------------


class TestWalModeEncrypted:
    """WAL journal mode works with encrypted connections."""

    def test_wal_mode_active_in_encrypted_mode(self, tmp_db_path):
        """WAL mode should be enabled even when using sqlcipher."""
        fake_module = _make_fake_sqlcipher_module()

        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=fake_module,
        ):
            registry = MetadataRegistry(
                db_path=tmp_db_path,
                encryption_key="test-key",
            )

        with registry._lock:
            row = registry._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        registry.close()


# ---------------------------------------------------------------------------
# F5 mitigation: File-based encryption key loading
# ---------------------------------------------------------------------------


class TestFileBasedKeyLoading:
    """MetadataRegistry works with encryption key loaded from a file (Docker secrets)."""

    @pytest.fixture
    def fake_sqlcipher_module(self):
        return _make_fake_sqlcipher_module()

    def test_registry_uses_key_loaded_from_file(
        self,
        tmp_path,
        tmp_db_path,
        monkeypatch,
        fake_sqlcipher_module,
    ):
        """Write key to a file, load via settings, verify MetadataRegistry is encrypted.

        Simulates Docker secrets workflow:
        1. Key file exists at /run/secrets/db_encryption_key
        2. Environment has DB_ENCRYPTION_KEY_FILE=/run/secrets/db_encryption_key
        3. DatabaseSettings reads and resolves the key
        4. MetadataRegistry(encryption_key=None) picks it up from settings
        5. Verify MetadataRegistry recognises encryption is enabled
        """
        from sec_semantic_search.config import DatabaseSettings, reload_settings

        # Create a temporary key file (simulating Docker secrets)
        key_file = tmp_path / "db_encryption_key"
        key_file.write_text("file-based-secret-key\n")  # Trailing newline like Docker

        # Patch the environment
        monkeypatch.setenv("DB_ENCRYPTION_KEY_FILE", str(key_file))
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)

        # Reload settings to pick up the new env vars
        reload_settings()

        # Verify that DatabaseSettings resolved the key from the file
        settings = DatabaseSettings()
        assert settings.encryption_key == "file-based-secret-key"
        assert settings.encryption_key_file == str(key_file)

        # Create MetadataRegistry with encryption_key=None (use settings)
        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=fake_sqlcipher_module,
        ):
            registry = MetadataRegistry(
                db_path=tmp_db_path,
                encryption_key=None,  # Should use settings.database.encryption_key
            )

        # Verify it picked up the encrypted key — the encryption flag should be True
        # (This proves the key was loaded from the file and encryption is active)
        assert registry.encrypted is True
        registry.close()
