"""
Pseudonymization of sensitive identifiers (e.g., Prolific IDs).

Why this exists
---------------
Public datasets must not contain identifiers that can be linked back to real
respondents (Prolific PIDs, session IDs, IP addresses, geolocation, etc.).
A naive `sha256(id)` is *not* safe for low-entropy identifiers like Prolific
PIDs (24-char hex): an attacker holding a list of candidate PIDs can simply
hash each one and look it up in the released data.

Approach
--------
We use **BLAKE2b in keyed mode** (built into Python's standard `hashlib`),
which is the current best-practice primitive for pseudonymizing identifiers:

- It is keyed (effectively a fast MAC), so without the secret key it is
  computationally infeasible to invert or precompute hashes.
- It is faster than HMAC-SHA256 and equally secure for this use case.
- It is in the standard library, so no extra dependencies are required.
- BLAKE3 is marginally faster but adds a third-party dependency; BLAKE2b
  is more than sufficient here.

The secret key is loaded from (in order of preference):

1. The ``FRAUDBENCH_HASH_KEY`` environment variable (hex-encoded).
2. A local ``.hash_key`` file in the repository root (hex-encoded, 64 chars).

If neither exists, a fresh 256-bit key is generated with ``secrets.token_bytes``
and written to ``.hash_key``. The user is warned to keep this file private and
to back it up — losing the key means hashes can never be re-derived from the
original IDs again (which is also the desired property if you want to
permanently sever the link).

The ``.hash_key`` file is added to ``.gitignore`` to prevent accidental commits.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Length of the BLAKE2b key (bytes). 32 = 256 bits, the recommended size.
_KEY_BYTES = 32

# Length of the hex digest returned by ``hash_id``. 16 hex chars = 64 bits,
# which gives a collision probability < 1 in 10^9 even for ~100k respondents
# (birthday bound), and keeps output IDs short and human-readable.
_DEFAULT_DIGEST_HEX_LEN = 16

# Default location of the secret key file. Resolved relative to the repo root.
_DEFAULT_KEY_FILENAME = ".hash_key"

# Cached key after the first lookup, to avoid repeated disk reads.
_cached_key: Optional[bytes] = None


def _repo_root() -> Path:
    """Return the repository root.

    This file lives at ``<repo_root>/src/fraudbench/utils/hashing.py``, so we
    need to walk up four levels to get back to the repo root.
    """
    return Path(__file__).resolve().parents[3]


def get_hash_key(key_path: Optional[str | os.PathLike] = None) -> bytes:
    """Return the secret hashing key, generating one on first use.

    Resolution order:

    1. ``FRAUDBENCH_HASH_KEY`` environment variable (hex string).
    2. ``key_path`` if provided, else ``<repo_root>/.hash_key``.
    3. A new random 256-bit key is generated and written to that file.

    The key is cached in-process after the first call.
    """
    global _cached_key
    if _cached_key is not None:
        return _cached_key

    env_key = os.environ.get("FRAUDBENCH_HASH_KEY")
    if env_key:
        try:
            key = bytes.fromhex(env_key.strip())
        except ValueError as e:
            raise ValueError(
                "FRAUDBENCH_HASH_KEY must be a hex-encoded string"
            ) from e
        if len(key) != _KEY_BYTES:
            raise ValueError(
                f"FRAUDBENCH_HASH_KEY must decode to {_KEY_BYTES} bytes "
                f"({_KEY_BYTES * 2} hex chars), got {len(key)} bytes"
            )
        _cached_key = key
        return key

    if key_path is None:
        key_file = _repo_root() / _DEFAULT_KEY_FILENAME
    else:
        key_file = Path(key_path)

    if key_file.exists():
        text = key_file.read_text(encoding="utf-8").strip()
        try:
            key = bytes.fromhex(text)
        except ValueError as e:
            raise ValueError(
                f"Hash key file {key_file} is not valid hex"
            ) from e
        if len(key) != _KEY_BYTES:
            raise ValueError(
                f"Hash key in {key_file} must decode to {_KEY_BYTES} bytes, "
                f"got {len(key)} bytes"
            )
        _cached_key = key
        return key

    # Generate a fresh key and persist it.
    key = secrets.token_bytes(_KEY_BYTES)
    key_file.write_text(key.hex(), encoding="utf-8")
    try:
        os.chmod(key_file, 0o600)  # POSIX best-effort; no-op on Windows.
    except (OSError, NotImplementedError):
        pass
    print(
        f"[utils.hashing] Generated a new hashing key at {key_file}.\n"
        f"[utils.hashing] KEEP THIS FILE PRIVATE and back it up — losing it "
        f"means hashes cannot be re-derived. It is gitignored by default."
    )
    _cached_key = key
    return key


def hash_id(
    value,
    key: Optional[bytes] = None,
    digest_hex_len: int = _DEFAULT_DIGEST_HEX_LEN,
) -> Optional[str]:
    """Pseudonymize a single identifier with BLAKE2b in keyed mode.

    ``None``, ``NaN`` and empty strings pass through unchanged so that missing
    values stay missing.

    Args:
        value: The identifier to hash. Coerced to ``str`` before hashing.
        key: Optional explicit key (bytes). Defaults to ``get_hash_key()``.
        digest_hex_len: Length of the returned hex string. Defaults to 16
            (64 bits).
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None

    if key is None:
        key = get_hash_key()

    # blake2b digest_size is in bytes; we want digest_hex_len hex chars.
    digest_bytes = max(1, (digest_hex_len + 1) // 2)
    h = hashlib.blake2b(s.encode("utf-8"), key=key, digest_size=digest_bytes)
    return h.hexdigest()[:digest_hex_len]


def _hash_series(
    series: pd.Series,
    key: Optional[bytes] = None,
    digest_hex_len: int = _DEFAULT_DIGEST_HEX_LEN,
) -> pd.Series:
    """Hash every value in a pandas Series, preserving NA."""
    if key is None:
        key = get_hash_key()
    return series.map(lambda v: hash_id(v, key=key, digest_hex_len=digest_hex_len))


def hash_dataframe_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    key: Optional[bytes] = None,
    digest_hex_len: int = _DEFAULT_DIGEST_HEX_LEN,
) -> pd.DataFrame:
    """Hash the given columns of ``df`` in place (returns the same DataFrame).

    Columns that do not exist in ``df`` are silently skipped, so the same call
    can be reused across schemas that drop or rename a column.
    """
    if key is None:
        key = get_hash_key()
    for col in columns:
        if col in df.columns:
            df[col] = _hash_series(df[col], key=key, digest_hex_len=digest_hex_len)
    return df
