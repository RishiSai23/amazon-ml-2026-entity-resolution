"""RAM-safe streaming inference for the Amazon ML Challenge test set.

The existing mock/train pipeline is intentionally untouched. This module provides
the test-only path: S2/S3 are normalized in chunks into SQLite, S1 is processed
in chunks, and only candidate records needed by the current S1 chunk are loaded.
"""

from __future__ import annotations

import gc
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.features import extract_pairwise_features
from src.matching import predict_matches
from src.normalize import normalize_dataframe
from src.postprocess import format_matching_results


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE records (
    source TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    business_name TEXT NOT NULL,
    business_address TEXT NOT NULL,
    country TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    address_normalized TEXT NOT NULL,
    country_normalized TEXT NOT NULL,
    name_tokens TEXT NOT NULL,
    address_tokens TEXT NOT NULL,
    name_core_tokens TEXT NOT NULL,
    name_suffix_tokens TEXT NOT NULL,
    PRIMARY KEY (source, entity_id)
);

CREATE TABLE blocking_index (
    country TEXT NOT NULL,
    block_type TEXT NOT NULL,
    block_value TEXT NOT NULL,
    source TEXT NOT NULL,
    entity_id TEXT NOT NULL
);

CREATE INDEX idx_blocking_lookup
    ON blocking_index(country, block_type, block_value);

CREATE TABLE candidates (
    source1_entity_id TEXT NOT NULL,
    candidate_entity_id TEXT NOT NULL,
    candidate_source TEXT NOT NULL,
    blocking_rules TEXT NOT NULL,
    PRIMARY KEY (
        source1_entity_id,
        candidate_entity_id,
        candidate_source
    )
);

CREATE INDEX idx_candidates_pair
    ON candidates(source1_entity_id, candidate_entity_id);

CREATE TABLE results (
    entity_id TEXT PRIMARY KEY,
    matched_entity_ids TEXT NOT NULL
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA.replace(
        "CREATE TABLE records (",
        "CREATE TABLE IF NOT EXISTS records ("
    ).replace(
        "CREATE TABLE blocking_index (",
        "CREATE TABLE IF NOT EXISTS blocking_index ("
    ).replace(
        "CREATE TABLE candidates (",
        "CREATE TABLE IF NOT EXISTS candidates ("
    ).replace(
        "CREATE TABLE results (",
        "CREATE TABLE IF NOT EXISTS results ("
    ).replace(
        "CREATE INDEX idx_blocking_lookup",
        "CREATE INDEX IF NOT EXISTS idx_blocking_lookup"
    ).replace(
        "CREATE INDEX idx_candidates_pair",
        "CREATE INDEX IF NOT EXISTS idx_candidates_pair"
    ))
    return conn


def _json_list(value) -> str:
    return json.dumps(value if isinstance(value, list) else [])


def _loads_list(value: str) -> List[str]:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []

    return result if isinstance(result, list) else []


def _clean_raw_chunk(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    return df


def _block_entries(row: pd.Series, config, source: str):
    """Yield the exact blocking keys used by the existing blocker."""

    country = row["country_normalized"]
    entity_id = str(row["entity_id"])

    name = row["name_normalized"]
    core_tokens = row["name_core_tokens"]
    name_tokens = row["name_tokens"]
    addr_tokens = row["address_tokens"]

    # Block A: Exact Name
    if config.enable_exact_name_block and name:
        yield (
            country,
            "exact_name",
            name,
            source,
            entity_id,
        )

    # Block B: Name Token
    if config.enable_name_prefix_block:
        tokens = core_tokens if core_tokens else name_tokens

        for token in tokens:
            if len(token) >= config.min_token_len:
                yield (
                    country,
                    "name_token",
                    token,
                    source,
                    entity_id,
                )

    # Block C: Address Token
    if config.enable_address_token_block and addr_tokens:
        min_addr_len = config.min_token_len + 1

        for token in addr_tokens:
            if token.isdigit() or len(token) >= min_addr_len:
                yield (
                    country,
                    "address_token",
                    token,
                    source,
                    entity_id,
                )

    # Block D: Character Signature
    if config.enable_char_signature_block and name:
        clean_chars = [c for c in name if c.isalnum()]

        if clean_chars:
            sig = "".join(
                sorted(clean_chars[: config.char_sig_len])
            )

            yield (
                country,
                "char_signature",
                sig,
                source,
                entity_id,
            )


def _insert_reference_chunk(
    conn: sqlite3.Connection,
    normalized: pd.DataFrame,
    source: str,
    config,
) -> None:
    record_rows = []
    block_rows = []

    for _, row in normalized.iterrows():
        record_rows.append(
            (
                source,
                str(row["entity_id"]),
                str(row.get("business_name", "")),
                str(row.get("business_address", "")),
                str(row.get("country", "")),
                str(row.get("name_normalized", "")),
                str(row.get("address_normalized", "")),
                str(row.get("country_normalized", "")),
                _json_list(row.get("name_tokens", [])),
                _json_list(row.get("address_tokens", [])),
                _json_list(row.get("name_core_tokens", [])),
                _json_list(row.get("name_suffix_tokens", [])),
            )
        )

        block_rows.extend(
            _block_entries(row, config, source)
        )

    conn.executemany(
        """
        INSERT INTO records (
            source,
            entity_id,
            business_name,
            business_address,
            country,
            name_normalized,
            address_normalized,
            country_normalized,
            name_tokens,
            address_tokens,
            name_core_tokens,
            name_suffix_tokens
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record_rows,
    )

    conn.executemany(
        """
        INSERT INTO blocking_index (
            country,
            block_type,
            block_value,
            source,
            entity_id
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        block_rows,
    )

    conn.commit()


def build_reference_store(
    s2_path: Path,
    s3_path: Path,
    db_path: Path,
    required_columns: Sequence[str],
    legal_suffixes,
    chunk_size: int,
    config,
) -> None:
    """Normalize/index S2 and S3 incrementally into SQLite."""

    # If the reference database already exists, reuse it.
    # This is important because S2/S3 indexing is expensive.
    if db_path.exists():
        print(f"  Reusing existing reference store: {db_path}")
        return

    conn = _connect(db_path)

    try:
        for source, path in (
            ("source2", s2_path),
            ("source3", s3_path),
        ):
            print(f"  Indexing {source}: {path.name}")

            for chunk_no, raw_chunk in enumerate(
                pd.read_csv(
                    path,
                    sep="\t",
                    dtype=str,
                    chunksize=chunk_size,
                    keep_default_na=False,
                ),
                start=1,
            ):
                raw_chunk = _clean_raw_chunk(
                    raw_chunk,
                    required_columns,
                )

                normalized = normalize_dataframe(
                    raw_chunk,
                    legal_suffixes,
                )

                _insert_reference_chunk(
                    conn,
                    normalized,
                    source,
                    config,
                )

                if chunk_no == 1 or chunk_no % 10 == 0:
                    print(
                        f"    {source} chunk "
                        f"{chunk_no:,}: "
                        f"{len(normalized):,} rows"
                    )

                del raw_chunk
                del normalized
                gc.collect()

    finally:
        conn.close()

def _lookup_block(
    conn: sqlite3.Connection,
    country: str,
    block_type: str,
    block_value: str,
) -> set:
    rows = conn.execute(
        """
        SELECT source, entity_id
        FROM blocking_index
        WHERE country = ?
          AND block_type = ?
          AND block_value = ?
        """,
        (
            country,
            block_type,
            block_value,
        ),
    ).fetchall()

    return {
        (source, entity_id)
        for source, entity_id in rows
    }


def _build_chunk_block_cache(
    conn: sqlite3.Connection,
    s1_chunk: pd.DataFrame,
    config,
) -> Dict[Tuple[str, str, str], set]:
    """Query each unique S1 block key once.

    Exact-name blocks intentionally have NO 500-row cap,
    matching blocking.py.

    Name-token, address-token, and character-signature blocks
    retain the existing 500-row cap.
    """

    keys = set()

    for _, row in s1_chunk.iterrows():
        country = row["country_normalized"]
        name = row["name_normalized"]

        core_tokens = row["name_core_tokens"]
        name_tokens = row["name_tokens"]
        addr_tokens = row["address_tokens"]

        # Exact name
        if config.enable_exact_name_block and name:
            keys.add(
                (
                    country,
                    "exact_name",
                    name,
                )
            )

        # Name token
        if config.enable_name_prefix_block:
            tokens = (
                core_tokens
                if core_tokens
                else name_tokens
            )

            for token in tokens:
                if len(token) >= config.min_token_len:
                    keys.add(
                        (
                            country,
                            "name_token",
                            token,
                        )
                    )

        # Address token
        if config.enable_address_token_block and addr_tokens:
            min_addr_len = config.min_token_len + 1

            for token in addr_tokens:
                if (
                    token.isdigit()
                    or len(token) >= min_addr_len
                ):
                    keys.add(
                        (
                            country,
                            "address_token",
                            token,
                        )
                    )

        # Character signature
        if config.enable_char_signature_block and name:
            clean_chars = [
                c for c in name
                if c.isalnum()
            ]

            if clean_chars:
                sig = "".join(
                    sorted(
                        clean_chars[
                            : config.char_sig_len
                        ]
                    )
                )

                keys.add(
                    (
                        country,
                        "char_signature",
                        sig,
                    )
                )

    cache = {}

    for country, block_type, value in keys:
        matches = _lookup_block(
            conn,
            country,
            block_type,
            value,
        )

        # Preserve blocking.py:
        #
        # exact_name -> unrestricted
        # all other blocks -> maximum 500
        if (
            block_type != "exact_name"
            and len(matches) > 500
        ):
            matches = set()

        cache[
            (
                country,
                block_type,
                value,
            )
        ] = matches

    return cache


def _candidate_frame_for_chunk(
    s1_chunk: pd.DataFrame,
    config,
    block_cache: Dict[Tuple[str, str, str], set],
) -> pd.DataFrame:
    """Generate candidates for one S1 chunk."""

    candidate_map = defaultdict(set)

    for _, row in s1_chunk.iterrows():
        s1_id = str(row["entity_id"])
        country = row["country_normalized"]

        name = row["name_normalized"]
        core_tokens = row["name_core_tokens"]
        name_tokens = row["name_tokens"]
        addr_tokens = row["address_tokens"]

        # Strategy 1: Exact Name
        if config.enable_exact_name_block and name:
            matches = block_cache.get(
                (
                    country,
                    "exact_name",
                    name,
                ),
                set(),
            )

            for cand_source, cand_id in matches:
                candidate_map[
                    (
                        s1_id,
                        cand_id,
                        cand_source,
                    )
                ].add("exact_name")

        # Strategy 2: Name Token
        if config.enable_name_prefix_block:
            tokens = (
                core_tokens
                if core_tokens
                else name_tokens
            )

            for token in tokens:
                if len(token) < config.min_token_len:
                    continue

                matches = block_cache.get(
                    (
                        country,
                        "name_token",
                        token,
                    ),
                    set(),
                )

                for cand_source, cand_id in matches:
                    candidate_map[
                        (
                            s1_id,
                            cand_id,
                            cand_source,
                        )
                    ].add("name_token")

        # Strategy 3: Address Token
        if (
            config.enable_address_token_block
            and addr_tokens
        ):
            min_addr_len = config.min_token_len + 1

            for token in addr_tokens:
                if not (
                    token.isdigit()
                    or len(token) >= min_addr_len
                ):
                    continue

                matches = block_cache.get(
                    (
                        country,
                        "address_token",
                        token,
                    ),
                    set(),
                )

                for cand_source, cand_id in matches:
                    candidate_map[
                        (
                            s1_id,
                            cand_id,
                            cand_source,
                        )
                    ].add("address_token")

        # Strategy 4: Character Signature
        if (
            config.enable_char_signature_block
            and name
        ):
            clean_chars = [
                c for c in name
                if c.isalnum()
            ]

            if clean_chars:
                sig = "".join(
                    sorted(
                        clean_chars[
                            : config.char_sig_len
                        ]
                    )
                )

                matches = block_cache.get(
                    (
                        country,
                        "char_signature",
                        sig,
                    ),
                    set(),
                )

                for cand_source, cand_id in matches:
                    candidate_map[
                        (
                            s1_id,
                            cand_id,
                            cand_source,
                        )
                    ].add(
                        "char_signature"
                    )

    rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "candidate_source": cand_source,
            "blocking_rules": "|".join(
                sorted(rules)
            ),
        }
        for (
            s1_id,
            cand_id,
            cand_source,
        ), rules in candidate_map.items()
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "blocking_rules",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
            ]
        )
        .reset_index(drop=True)
    )


def _load_candidate_records(
    conn: sqlite3.Connection,
    candidate_pairs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load only S2/S3 records referenced by this chunk."""

    if candidate_pairs.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_normalized",
        "address_normalized",
        "country_normalized",
        "name_tokens",
        "address_tokens",
        "name_core_tokens",
        "name_suffix_tokens",
    ]

    output = {}

    for source in (
        "source2",
        "source3",
    ):
        ids = (
            candidate_pairs.loc[
                candidate_pairs[
                    "candidate_source"
                ] == source,
                "candidate_entity_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if not ids:
            output[source] = pd.DataFrame(
                columns=columns
            )
            continue

        frames = []

        # Stay safely below SQLite's parameter limit.
        for start in range(
            0,
            len(ids),
            500,
        ):
            batch = ids[
                start : start + 500
            ]

            placeholders = ",".join(
                "?" for _ in batch
            )

            rows = conn.execute(
                f"""
                SELECT
                    entity_id,
                    business_name,
                    business_address,
                    country,
                    name_normalized,
                    address_normalized,
                    country_normalized,
                    name_tokens,
                    address_tokens,
                    name_core_tokens,
                    name_suffix_tokens
                FROM records
                WHERE source = ?
                  AND entity_id IN (
                      {placeholders}
                  )
                """,
                [source, *batch],
            ).fetchall()

            frames.append(
                pd.DataFrame(
                    rows,
                    columns=columns,
                )
            )

        df = (
            pd.concat(
                frames,
                ignore_index=True,
            )
            if frames
            else pd.DataFrame(
                columns=columns
            )
        )

        for col in [
            "name_tokens",
            "address_tokens",
            "name_core_tokens",
            "name_suffix_tokens",
        ]:
            if not df.empty:
                df[col] = df[col].apply(
                    _loads_list
                )

        output[source] = df

    return (
        output["source2"],
        output["source3"],
    )


def _write_candidates(
    conn: sqlite3.Connection,
    candidate_pairs: pd.DataFrame,
) -> None:
    if candidate_pairs.empty:
        return

    conn.executemany(
        """
        INSERT INTO candidates (
            source1_entity_id,
            candidate_entity_id,
            candidate_source,
            blocking_rules
        )
        VALUES (?, ?, ?, ?)
        """,
        candidate_pairs[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "blocking_rules",
            ]
        ].itertuples(
            index=False,
            name=None,
        ),
    )

    conn.commit()


def _write_results(
    conn: sqlite3.Connection,
    s1_chunk: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    formatted = format_matching_results(
        s1_chunk,
        predictions,
    )

    conn.executemany(
        """
        INSERT INTO results (
            entity_id,
            matched_entity_ids
        )
        VALUES (?, ?)
        ON CONFLICT(entity_id)
        DO UPDATE SET
            matched_entity_ids =
                excluded.matched_entity_ids
        """,
        formatted.itertuples(
            index=False,
            name=None,
        ),
    )

    conn.commit()


def _export_candidates(
    conn: sqlite3.Connection,
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        f.write(
            "source1_entity_id\t"
            "candidate_entity_id\t"
            "candidate_source\t"
            "blocking_rules\n"
        )

        for row in conn.execute(
            """
            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                blocking_rules
            FROM candidates
            ORDER BY
                source1_entity_id,
                candidate_entity_id,
                candidate_source
            """
        ):
            f.write(
                "\t".join(
                    str(v)
                    for v in row
                )
                + "\n"
            )


def _export_results(
    conn: sqlite3.Connection,
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        f.write(
            "entity_id\tmatched_entity_ids\n"
        )

        for entity_id, matched in conn.execute(
            """
            SELECT
                entity_id,
                matched_entity_ids
            FROM results
            ORDER BY entity_id
            """
        ):
            f.write(
                f"{entity_id}\t{matched}\n"
            )


def _validate_streaming_results(
    conn: sqlite3.Connection,
    expected_s1_count: int,
) -> None:
    """Validate the same critical invariants without loading full outputs."""

    result_count = conn.execute(
        "SELECT COUNT(*) FROM results"
    ).fetchone()[0]

    if result_count != expected_s1_count:
        raise ValueError(
            "Streaming validation failed: "
            f"expected {expected_s1_count} "
            f"S1 results, found {result_count}."
        )

    for s1_id, matched in conn.execute(
        """
        SELECT entity_id, matched_entity_ids
        FROM results
        """
    ):
        if not matched:
            continue

        for candidate_id in matched.split(","):
            candidate_id = candidate_id.strip()

            exists = conn.execute(
                """
                SELECT 1
                FROM candidates
                WHERE source1_entity_id = ?
                  AND candidate_entity_id = ?
                LIMIT 1
                """,
                (
                    s1_id,
                    candidate_id,
                ),
            ).fetchone()

            if exists is None:
                raise AssertionError(
                    "Streaming validation failed: "
                    f"({s1_id}, {candidate_id}) "
                    "was predicted but is absent "
                    "from candidate_pairs."
                )

    print(
        "SUCCESS: Streaming output validation passed."
    )


def run_streaming_test_pipeline(
    config,
) -> None:
    """Run RAM-safe production inference for test data."""

    chunk_size = getattr(
        config,
        "inference_chunk_size",
        5000,
    )

    db_name = getattr(
        config,
        "streaming_db_name",
        "test_inference.sqlite",
    )

    test_dir = config.test_dir
    output_dir = config.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    s1_path = (
        test_dir / "test_source1.tsv"
    )
    s2_path = (
        test_dir / "test_source2.tsv"
    )
    s3_path = (
        test_dir / "test_source3.tsv"
    )

    db_path = output_dir / db_name

    print(
        "\n[Streaming Test 1/4] "
        "Building S2/S3 disk-backed index..."
    )

    build_reference_store(
        s2_path,
        s3_path,
        db_path,
        config.required_columns,
        config.legal_suffixes,
        chunk_size,
        config,
    )

    conn = _connect(db_path)

    try:
        # Allows safe reruns using the same SQLite file.
        conn.execute(
            "DELETE FROM candidates"
        )
        conn.execute(
            "DELETE FROM results"
        )
        conn.commit()

        total_s1 = 0
        total_candidates = 0
        total_predictions = 0

        print(
            "\n[Streaming Test 2/4] "
            f"Processing S1 with "
            f"chunk_size={chunk_size:,}..."
        )

        for chunk_no, raw_s1 in enumerate(
            pd.read_csv(
                s1_path,
                sep="\t",
                dtype=str,
                chunksize=chunk_size,
                keep_default_na=False,
            ),
            start=1,
        ):
            raw_s1 = _clean_raw_chunk(
                raw_s1,
                config.required_columns,
            )

            s1_norm = normalize_dataframe(
                raw_s1,
                config.legal_suffixes,
            )

            block_cache = (
                _build_chunk_block_cache(
                    conn,
                    s1_norm,
                    config,
                )
            )

            candidate_pairs = (
                _candidate_frame_for_chunk(
                    s1_norm,
                    config,
                    block_cache,
                )
            )

            _write_candidates(
                conn,
                candidate_pairs,
            )

            (
                s2_candidates,
                s3_candidates,
            ) = _load_candidate_records(
                conn,
                candidate_pairs,
            )

            features = None

            if candidate_pairs.empty:
                predictions = pd.DataFrame(
                    columns=[
                        "source1_entity_id",
                        "candidate_entity_id",
                        "candidate_source",
                        "match_score",
                        "blocking_rules",
                    ]
                )
            else:
                features = extract_pairwise_features(
                    candidate_pairs,
                    s1_norm,
                    s2_candidates,
                    s3_candidates,
                )

                predictions = predict_matches(
                    features,
                    config,
                )

            _write_results(
                conn,
                s1_norm,
                predictions,
            )

            total_s1 += len(s1_norm)
            total_candidates += len(
                candidate_pairs
            )
            total_predictions += len(
                predictions
            )

            print(
                f"  chunk {chunk_no:>4}: "
                f"S1={len(s1_norm):>7,} | "
                f"candidates="
                f"{len(candidate_pairs):>9,} | "
                f"matches="
                f"{len(predictions):>7,}"
            )

            del raw_s1
            del s1_norm
            del block_cache
            del candidate_pairs
            del s2_candidates
            del s3_candidates
            del predictions

            if features is not None:
                del features

            gc.collect()

        print(
            "\n[Streaming Test 3/4] "
            "Validating disk-backed outputs..."
        )

        _validate_streaming_results(
            conn,
            expected_s1_count=total_s1,
        )

        candidate_path = (
            output_dir / "candidate_pairs.tsv"
        )

        matching_path = (
            output_dir / "matching_results.tsv"
        )

        print(
            "[Streaming Test 4/4] "
            "Exporting final TSVs..."
        )

        _export_candidates(
            conn,
            candidate_path,
        )

        _export_results(
            conn,
            matching_path,
        )

        print(
            "\nStreaming test pipeline "
            "finished successfully."
        )

        print(
            f"  S1 rows processed: "
            f"{total_s1:,}"
        )

        print(
            f"  Candidate pairs:   "
            f"{total_candidates:,}"
        )

        print(
            f"  Predicted matches: "
            f"{total_predictions:,}"
        )

        print(
            f"  Candidate output:  "
            f"{candidate_path}"
        )

        print(
            f"  Matching output:   "
            f"{matching_path}"
        )

        print(
            f"  SQLite work store: "
            f"{db_path}"
        )

    finally:
        conn.close()