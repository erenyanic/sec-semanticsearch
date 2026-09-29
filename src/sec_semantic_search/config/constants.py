"""Application-wide constants."""

# Supported SEC filing forms (alphabetical order).
# Amendments (e.g. 10-K/A) are separate form types because edgartools
# marks them with a distinct ``form`` attribute and supports querying
# them directly via ``Company.get_filings(form='10-K/A')``.
SUPPORTED_FORMS = ("8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A")

# Base (non-amendment) form types — used by the amendment filter to
# decide whether to skip ``/A`` filings returned by edgartools.
BASE_FORMS = ("8-K", "10-K", "10-Q")

# Amendment form types — the ``/A`` variants.
AMENDMENT_FORMS = ("8-K/A", "10-K/A", "10-Q/A")

# Default form types for ingestion (both annual and quarterly)
DEFAULT_FORM_TYPES = "10-K,10-Q"


def parse_form_types(form_input: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated form type string.

    Accepts ``"10-K"``, ``"10-Q"``, ``"10-K,10-Q"``, ``"10-Q, 10-K"``, etc.
    Returns a sorted tuple of unique, validated form types.  Input order
    does not matter — ``"10-Q,10-K"`` produces the same result as
    ``"10-K,10-Q"``.

    Args:
        form_input: Comma-separated form type string.

    Returns:
        Sorted tuple of validated form types.

    Raises:
        ValueError: If any form type is not in ``SUPPORTED_FORMS``.
    """
    raw = [part.strip().upper() for part in form_input.split(",") if part.strip()]

    if not raw:
        raise ValueError(f"Empty form type. Supported: {', '.join(SUPPORTED_FORMS)}")

    invalid = [f for f in raw if f not in SUPPORTED_FORMS]
    if invalid:
        raise ValueError(
            f"Unsupported form type(s): {', '.join(invalid)}. "
            f"Supported: {', '.join(SUPPORTED_FORMS)}"
        )

    return tuple(sorted(set(raw)))


# Embedding model parameters. The model name and the chunking, storage and
# search defaults live on the settings classes (config/settings.py), the
# single source of truth.
EMBEDDING_DIMENSION = 768  # Dimension of google/embeddinggemma-300m

# Collection naming
COLLECTION_NAME = "sec_filings"
