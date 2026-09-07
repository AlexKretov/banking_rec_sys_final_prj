"""Bounded, read-only helpers for eda.ipynb (no model/service imports).

Parquet footers describe the complete *local* inventory. Row sampling is uniform
without replacement inside every domain/day/file stratum, never a file prefix.
Only additive event totals/ratios may use the resulting inverse sampling weights;
unique entities, overlaps, degrees and cold-start rates describe the sample.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ITEM_DOMAINS = ("marketplace", "retail", "offers")
ALL_DOMAINS = (*ITEM_DOMAINS, "reviews", "payments")
ID_COLUMNS = ("user_id", "item_id", "brand_id")
TIME_COLUMNS = ("timestamp", "event_timestamp", "event_time", "datetime")
INVENTORY_COLUMNS = (
    "path", "relative_path", "role", "domain", "day", "status", "rows",
    "bytes", "mtime_ns", "row_groups", "schema", "nested_columns", "error",
)


def report_filename(name: str) -> str:
    """Keep data-derived column/table labels from escaping the report directory."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        return f"{name}.csv"
    return f"table_{hashlib.sha256(name.encode()).hexdigest()[:16]}.csv"


def valid_id(series: pd.Series) -> pd.Series:
    """Do not invent sentinel IDs or cast UInt64 identifiers through float."""
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def find_eda_root(path: str | Path, variant: str | None = "small") -> Path | None:
    """Accept an event-only download too; never silently pick Small over Full."""
    if variant not in (None, "small", "full"):
        raise ValueError("variant must be small, full or None")
    base = Path(path)
    variants = ("small", "full") if variant is None else (variant,)
    candidates = [base, *(base / "dataset" / v for v in variants)]
    roots = []
    for candidate in candidates:
        present = any((candidate / name).is_file() for name in ("users.pq", "brands.pq"))
        present |= any((candidate / d / "events").is_dir() for d in ALL_DOMAINS)
        present |= (candidate / "reviews").is_dir()
        if present and candidate.resolve() not in roots:
            roots.append(candidate.resolve())
    if len(roots) > 1:
        raise ValueError(f"Ambiguous dataset roots: {roots}; set DATA_DIR explicitly")
    return roots[0] if roots else None


def _nested(dtype: pa.DataType) -> bool:
    return any(check(dtype) for check in (
        pa.types.is_list, pa.types.is_large_list, pa.types.is_fixed_size_list,
        pa.types.is_struct, pa.types.is_map, pa.types.is_union,
    ))


def scalar_columns(schema: pa.Schema) -> list[str]:
    return [f.name for f in schema if not _nested(f.type)
            and not pa.types.is_binary(f.type) and "embed" not in f.name.lower()]


def _inspect_file(root: Path, file: Path, role: str, domain: str, day) -> dict:
    row = dict.fromkeys(INVENTORY_COLUMNS)
    row.update(path=str(file), relative_path=file.relative_to(root).as_posix(),
               role=role, domain=domain, day=day, schema={}, nested_columns=[], error="")
    if not file.is_file():
        row["status"] = "missing"
        return row
    stat = file.stat()
    row.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    try:
        pf = pq.ParquetFile(file)
        row.update(rows=pf.metadata.num_rows, row_groups=pf.metadata.num_row_groups,
                   schema={f.name: str(f.type) for f in pf.schema_arrow},
                   nested_columns=[f.name for f in pf.schema_arrow if _nested(f.type)],
                   status="ok" if pf.metadata.num_rows else "empty")
    except (OSError, ValueError, pa.ArrowException) as exc:
        row.update(status="unreadable", error=f"{type(exc).__name__}: {exc}")
    return row


def dataset_inventory(root: Path | None, domains: Iterable[str] = ALL_DOMAINS,
                      day_begin: int | None = None, day_end: int | None = None) -> pd.DataFrame:
    """Read footers, not event tables. Missing partitions are not zero activity.

    Multiple filenames for one logical (role, domain, integer day) are rejected
    from analysis rather than double counted (e.g. 1.pq and 00001.pq).
    """
    if day_begin is not None and day_end is not None and day_begin > day_end:
        raise ValueError("day_begin must be <= day_end")
    domains = tuple(dict.fromkeys(domains))
    if set(domains) - set(ALL_DOMAINS):
        raise ValueError("Unknown domain")
    if root is None:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    root = Path(root)
    records = [_inspect_file(root, root / f"{name}.pq", name, "shared", None)
               for name in ("users", "brands")]
    for domain in domains:
        if domain in ITEM_DOMAINS:
            records.append(_inspect_file(root, root / domain / "items.pq", "items", domain, None))
        roles = ("events", "receipts") if domain == "payments" else ("events",)
        for role in roles:
            folder = root / domain if domain == "reviews" else root / domain / role
            by_day: dict[int, list[Path]] = {}
            for file in sorted([*folder.glob("*.pq"), *folder.glob("*.parquet")]):
                if not file.stem.isdigit():
                    bad = _inspect_file(root, file, role, domain, None)
                    bad["status"] = "invalid_day_name"
                    records.append(bad)
                    continue
                day = int(file.stem)
                if (day_begin is None or day >= day_begin) and (day_end is None or day <= day_end):
                    by_day.setdefault(day, []).append(file)
            lo = day_begin if day_begin is not None else min(by_day, default=None)
            hi = day_end if day_end is not None else max(by_day, default=None)
            expected = range(lo, hi + 1) if lo is not None and hi is not None else sorted(by_day)
            for day in expected:
                files = by_day.get(day, [folder / f"{day:05d}.pq"])
                for file in files:
                    row = _inspect_file(root, file, role, domain, day)
                    if len(files) > 1:
                        row["status"] = "duplicate_partition"
                    records.append(row)
    result = pd.DataFrame(records, columns=INVENTORY_COLUMNS)
    result["day"] = pd.array(result["day"], dtype="Int64")
    return result


def balanced_plan(inventory: pd.DataFrame, max_rows: int,
                  roles: Iterable[str] = ("events",)) -> pd.DataFrame:
    """Water-fill the row budget across all readable, nonempty strata."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    plan = inventory.loc[inventory.role.isin(roles) & inventory.status.eq("ok")].copy()
    plan = plan.sort_values("relative_path").reset_index(drop=True)
    if plan.empty:
        plan["sample_rows"] = pd.Series(dtype="int64")
        return plan
    if max_rows < len(plan):
        raise ValueError("Row budget must cover every stratum; increase max_rows or narrow days")
    capacities = plan.rows.to_numpy(dtype=np.int64)
    quota = np.zeros(len(plan), dtype=np.int64)
    remaining = min(int(capacities.sum()), int(max_rows))
    while remaining:
        active = np.flatnonzero(quota < capacities)
        share = remaining // len(active)
        if share == 0:
            quota[active[:remaining]] += 1
            break
        take = np.minimum(capacities[active] - quota[active], share)
        quota[active] += take
        remaining -= int(take.sum())
    plan["sample_rows"] = quota
    return plan


def _file_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest(), "little")


def sample_parquet(file: str | Path, n: int, seed: int = 42,
                   columns: list[str] | None = None, batch_size: int = 65_536) -> pd.DataFrame:
    """Uniform row indices from the whole file, streamed in bounded batches.

    Arrow nullable dtypes prevent large integer IDs with nulls being rounded by
    an intermediate float64 conversion. Nested columns are opt-in (embeddings).
    """
    if n < 0 or batch_size <= 0:
        raise ValueError("n must be nonnegative and batch_size positive")
    pf = pq.ParquetFile(file)
    columns = scalar_columns(pf.schema_arrow) if columns is None else columns
    total = pf.metadata.num_rows
    n = min(n, total)
    if n == 0:
        return pd.DataFrame(columns=[*columns, "_row_in_file"])
    rng = np.random.default_rng(seed)
    selected = np.arange(total) if n == total else np.sort(rng.choice(total, n, replace=False))
    frames, offset = [], 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        end = offset + batch.num_rows
        left, right = np.searchsorted(selected, [offset, end])
        if left < right:
            indices = selected[left:right]
            frame = batch.take(pa.array(indices - offset)).to_pandas(types_mapper=pd.ArrowDtype)
            frame["_row_in_file"] = indices
            frames.append(frame)
        offset = end
    if offset != total:
        raise ValueError(f"Footer/scan row mismatch: {file}")
    return pd.concat(frames, ignore_index=True)


def parse_event_time(frame: pd.DataFrame, column: str | None = None,
                     unit: str | None = None, origin: str | None = None) -> tuple[pd.Series, dict]:
    """Never guess seconds/ms/ns or the epoch of an anonymized numeric clock."""
    empty = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    column = column or next((c for c in TIME_COLUMNS if c in frame), None)
    report = {"column": column, "status": "absent", "raw_non_null": 0, "parsed": 0, "unparsed": 0}
    if column is None or column not in frame:
        return empty, report
    raw = frame[column]
    report["raw_non_null"] = int(raw.notna().sum())
    if not raw.notna().any():
        report["status"] = "all_null"
        return empty, report
    numeric_text = raw.astype("string").str.fullmatch(r"[+-]?\d+(?:\.\d+)?").fillna(False)
    numeric = pd.api.types.is_numeric_dtype(raw.dtype) or numeric_text[raw.notna()].all()
    if numeric:
        if unit is None or origin is None:
            report.update(status="numeric_needs_unit_and_origin", unparsed=report["raw_non_null"])
            return empty, report
        if unit not in ("s", "ms", "us", "ns", "D"):
            raise ValueError("Unsupported timestamp unit")
        # Nullable strings -> nullable numeric avoids float64 rounding of integer
        # nanoseconds when a source column also contains nulls.
        values = pd.to_numeric(raw.astype("string"), errors="coerce")
        origin_time = pd.Timestamp("1970-01-01") if origin == "unix" else pd.Timestamp(origin)
        if pd.isna(origin_time):
            raise ValueError("Numeric clock origin must be a valid timestamp")
        origin_ns = origin_time.value
        factor = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000, "D": 86_400_000_000_000}[unit]
        parsed = empty.copy()
        if pd.api.types.is_integer_dtype(values.dtype):
            lo = (pd.Timestamp.min.value - origin_ns + factor - 1) // factor
            hi = (pd.Timestamp.max.value - origin_ns) // factor
            usable = values.between(lo, hi).fillna(False)
            # Python integer arithmetic also prevents unsigned offset overflow.
            nanoseconds = np.fromiter((int(v) * factor + origin_ns for v in values[usable]),
                                     dtype=np.int64, count=int(usable.sum()))
            parsed.loc[usable] = pd.to_datetime(nanoseconds, utc=True, unit="ns")
        else:
            usable = values.notna()
            utc_origin = origin_time.tz_convert("UTC").tz_localize(None) if origin_time.tzinfo else origin_time
            parsed.loc[usable] = pd.to_datetime(values[usable].to_numpy(dtype=float), unit=unit,
                                                origin=utc_origin, utc=True, errors="coerce")
        status = "explicit_numeric_clock"
    else:
        # Numeric strings mixed with ISO dates are invalid unless explicitly mapped.
        values = raw.astype("string").mask(numeric_text)
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
        status = "parsed_datetime" if not numeric_text.any() else "mixed_numeric_values_unparsed"
    report.update(status=status, parsed=int(parsed.notna().sum()),
                  unparsed=int((raw.notna() & parsed.isna()).sum()))
    return parsed, report


def empty_events() -> pd.DataFrame:
    result = pd.DataFrame({c: pd.Series(dtype="string") for c in (*ID_COLUMNS, "domain", "_action", "_source", "_item_key")})
    result["day"] = pd.Series(dtype="int64")
    result["_weight"] = pd.Series(dtype="float64")
    result["_event_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    return result


def load_eda_sample(inventory: pd.DataFrame, max_rows: int = 1_000_000, seed: int = 42,
                    time_config: dict | None = None, roles: Iterable[str] = ("events",)
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep raw nullable IDs/fields, including bad keys; do not clean before EDA."""
    plan = balanced_plan(inventory, max_rows, roles)
    frames, clocks = [], []
    for row in plan.itertuples():
        frame = sample_parquet(row.path, int(row.sample_rows), _file_seed(seed, row.relative_path))
        for column in ID_COLUMNS:
            if column in frame:
                frame[column] = frame[column].astype("string")
        for column in ("domain", "day"):
            if column in frame:
                frame[f"_raw_{column}"] = frame[column]
        frame["domain"], frame["day"] = row.domain, int(row.day)
        frame["_source"], frame["_weight"] = row.relative_path, int(row.rows) / len(frame)
        frame["_role"] = row.role
        if "action_type" in frame:
            frame["_action"] = frame.action_type.astype("string").str.strip().str.lower().fillna("__missing__")
        else:
            frame["_action"] = {"reviews": "review", "payments": "payment"}.get(row.domain, "__missing__")
        if "item_id" in frame:
            frame["_item_key"] = (row.domain + "::" + frame.item_id).where(valid_id(frame.item_id))
        else:
            frame["_item_key"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        clock, report = parse_event_time(frame, **(time_config or {}).get(row.domain, {}))
        frame["_event_time"] = clock
        clocks.append({"source": row.relative_path, "domain": row.domain, **report})
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True) if frames else empty_events()
    # Structural absence of IDs is visible through inventory.schema, not confused
    # with missing cells in a source column that actually exists.
    for column in ID_COLUMNS:
        if column not in events:
            events[column] = pd.Series(pd.NA, index=events.index, dtype="string")
    return events, plan, pd.DataFrame(clocks)


def schema_missingness(events: pd.DataFrame, inventory: pd.DataFrame,
                       fields: Iterable[str]) -> pd.DataFrame:
    rows = []
    files = inventory[inventory.role.eq("events") & inventory.status.isin(["ok", "empty"])]
    for domain, group in events.groupby("domain", observed=True):
        source_files = files[files.domain.eq(domain)]
        for field in fields:
            present_files = set(source_files.loc[source_files.schema.map(lambda s: field in s), "relative_path"])
            present = group._source.isin(present_files)
            nulls = group[field].isna() if field in group else pd.Series(True, index=group.index)
            rows.append({"domain": domain, "field": field, "files_total": len(source_files),
                         "files_present": len(present_files), "sample_present_rows": int(present.sum()),
                         "null_share_when_present": float(nulls[present].mean()) if present.any() else np.nan,
                         "structural_absence_share": float((~present).mean()),
                         "pooled_null_share": float(nulls.mean())})
    return pd.DataFrame(rows)


def robust_scores(values, min_points: int = 7) -> np.ndarray:
    """Descriptive MAD scores; constant baseline != division by zero/NaN.

    A deviation from a zero-MAD baseline receives signed infinity: a candidate
    to inspect, not proof of a bot, logging incident or statistical significance.
    """
    values = np.asarray(values, dtype=float)
    scores = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.sum() < min_points:
        return scores
    median = np.median(values[finite])
    delta = values[finite] - median
    mad = np.median(np.abs(delta))
    if mad == 0:
        result = np.zeros(delta.shape)
        result[delta != 0] = np.sign(delta[delta != 0]) * np.inf
        scores[finite] = result
    else:
        scores[finite] = 0.67448975 * delta / mad
    return scores


def numeric_profile(events: pd.DataFrame, fields=("price", "count", "rating", "amount", "quantity")) -> pd.DataFrame:
    rows = []
    for (domain, action), group in events.groupby(["domain", "_action"], observed=True, dropna=False):
        for field in fields:
            if field not in group or not group[field].notna().any():
                continue
            raw = group[field]
            values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            numeric_nan = (raw.astype("string").str.lower().isin(["nan", "+nan", "-nan"])
                           & raw.notna()).to_numpy(dtype=bool)
            parse_failed = raw.notna().to_numpy() & np.isnan(values) & ~numeric_nan
            positive = values[finite & (values > 0)]
            z = robust_scores(np.log1p(positive), min_points=30)
            quantiles = np.quantile(values[finite], [.01, .5, .99]) if finite.any() else [np.nan] * 3
            rows.append({"domain": domain, "action": action, "field": field, "sample_rows": len(group),
                         "missing": int(raw.isna().sum()), "nonnumeric": int(parse_failed.sum()),
                         "nan_values": int(numeric_nan.sum()), "infinite": int(np.isinf(values).sum()),
                         "negative": int((values[finite] < 0).sum()),
                         "zero": int((values[finite] == 0).sum()),
                         "fractional": int((~np.isclose(values[finite], np.round(values[finite]), rtol=0, atol=1e-9)).sum()),
                         "min": values[finite].min() if finite.any() else np.nan,
                         "p01": quantiles[0], "p50": quantiles[1], "p99": quantiles[2],
                         "max": values[finite].max() if finite.any() else np.nan,
                         "positive_log_tail_candidates": int((z > 5).sum()),
                         "tail_tested": int(np.count_nonzero(~np.isnan(z)))})
    return pd.DataFrame(rows)


def gini(counts) -> float:
    values = np.sort(np.asarray(counts, dtype=float))
    if not len(values) or values.sum() <= 0 or (values < 0).any():
        return np.nan
    n = len(values)
    return float(2 * np.dot(np.arange(1, n + 1), values) / (n * values.sum()) - (n + 1) / n)


def degree_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in events[events.domain.isin(ITEM_DOMAINS)].groupby("domain", observed=True):
        group = group[valid_id(group.user_id) & valid_id(group._item_key)]
        if group.empty:
            continue
        user_degree = group.groupby("user_id", observed=True).size()
        item_degree = group.groupby("_item_key", observed=True).size()
        pairs = len(group[["user_id", "_item_key"]].drop_duplicates())
        head = max(1, int(np.ceil(.01 * len(item_degree))))
        rows.append({"domain": domain, "sample_events": len(group), "users": len(user_degree),
                     "items": len(item_degree), "unique_pairs": pairs,
                     "density": pairs / (len(user_degree) * len(item_degree)),
                     "repeated_pair_event_share": 1 - pairs / len(group),
                     "user_events_p50": user_degree.quantile(.5), "user_events_p99": user_degree.quantile(.99),
                     "user_events_max": int(user_degree.max()),
                     "singleton_user_share": float(user_degree.eq(1).mean()),
                     "singleton_item_share": float(item_degree.eq(1).mean()),
                     "item_gini": gini(item_degree), "user_gini": gini(user_degree),
                     "top_1pct_item_event_share": item_degree.nlargest(head).sum() / len(group)})
    return pd.DataFrame(rows)


def overlap_table(events: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in events:
        return pd.DataFrame()
    sets = {domain: set(group.loc[valid_id(group[key]), key])
            for domain, group in events.groupby("domain", observed=True)}
    rows = []
    for source, a in sorted(sets.items()):
        for target, b in sorted(sets.items()):
            intersection, union = len(a & b), len(a | b)
            rows.append({"source": source, "target": target, "source_entities": len(a),
                         "target_entities": len(b), "intersection": intersection,
                         "jaccard": intersection / union if union else np.nan,
                         "p_target_given_source": intersection / len(a) if a else np.nan})
    return pd.DataFrame(rows)


def duplicate_profile(events: pd.DataFrame) -> pd.DataFrame:
    """Repeated scalar projections are candidates, not necessarily duplicate events."""
    rows = []
    raw_columns = [c for c in events if not c.startswith("_") or c.startswith("_raw_")]
    for domain, group in events.groupby("domain", observed=True):
        key = [c for c in ("user_id", "item_id", "brand_id", "_action", "day") if c in group]
        # The loader intentionally excludes nested fields; state this scope in EDA.
        repeated = group.duplicated(raw_columns, keep="first")
        rows.append({"domain": domain, "sample_rows": len(group),
                     "scalar_repeat_excess": int(repeated.sum()), "scalar_repeat_share": float(repeated.mean()),
                     "user_entity_action_day_repeat_share": float(group.duplicated(key).mean()),
                     "parsed_time_share": float(group._event_time.notna().mean())})
    return pd.DataFrame(rows)


def purged_day_split(events: pd.DataFrame, val_days: int = 2, test_days: int = 2,
                     gap_days: int = 1, day_begin: int | None = None,
                     day_end: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar-day train | gap | validation | gap | test, not row/unique-day split.

    Pass the requested window explicitly (as the notebook does) so missing tail
    files do not silently move the prediction cutoff into the past. Without an
    explicit window, the observed min/max days define the diagnostic window.
    One whole unused partition is conservative for the 12h rule *if* partitions
    correspond to the published clock. Check this against raw timestamps when
    their unit/origin are known; this function does not establish that mapping.
    """
    if min(val_days, test_days, gap_days) < 1:
        raise ValueError("val_days, test_days and gap_days must all be >= 1")
    if events.empty or events.day.isna().any():
        raise ValueError("Nonempty events with known partition days are required")
    start = int(events.day.min()) if day_begin is None else int(day_begin)
    end = int(events.day.max()) if day_end is None else int(day_end)
    if start > end or not events.day.between(start, end).all():
        raise ValueError("Events must lie inside the specified calendar window")
    test_start = end - test_days + 1
    val_end = test_start - gap_days - 1
    val_start = val_end - val_days + 1
    train_end = val_start - gap_days - 1
    if train_end < start:
        raise ValueError("Insufficient calendar history for train/gap/validation/gap/test")
    intervals = [
        ("train", start, train_end), ("gap_validation", train_end + 1, val_start - 1),
        ("validation", val_start, val_end), ("gap_test", val_end + 1, test_start - 1),
        ("test", test_start, end),
    ]
    result = events.copy()
    result["_split"] = "unassigned"
    rows = []
    for split, lo, hi in intervals:
        mask = result.day.between(lo, hi)
        result.loc[mask, "_split"] = split
        rows.append({"split": split, "day_from": lo, "day_to": hi,
                     "calendar_days": hi - lo + 1, "observed_days": result.loc[mask, "day"].nunique(),
                     "sample_events": int(mask.sum())})
    for split in ("train", "validation", "test"):
        if not result._split.eq(split).any():
            raise ValueError(f"No observed events in {split}; inspect coverage rather than shift the boundary")
    assert not result._split.eq("unassigned").any()
    assert train_end + gap_days < val_start and val_end + gap_days < test_start
    return result, pd.DataFrame(rows)


def purged_history(events: pd.DataFrame, prediction_time, gap_hours: float = 12) -> pd.DataFrame:
    """A single point-in-time history; unknown times are excluded, never imputed."""
    cutoff = pd.Timestamp(prediction_time)
    if cutoff.tzinfo is None:
        raise ValueError("prediction_time must have an explicit timezone")
    if gap_hours < 12:
        raise ValueError("T-ECD requires a minimum 12-hour gap")
    limit = cutoff.tz_convert("UTC") - pd.Timedelta(hours=gap_hours)
    history = events.loc[events._event_time.notna() & events._event_time.lt(limit)].copy()
    assert history.empty or (cutoff - history._event_time.max()) >= pd.Timedelta(hours=12)
    return history


def cold_start_table(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Observed-history coldness, not a claim about account/product creation."""
    rows = []
    all_users = set(history.loc[valid_id(history.user_id), "user_id"])
    for domain, target in targets[targets.domain.isin(ITEM_DOMAINS)].groupby("domain", observed=True):
        target = target[valid_id(target.user_id) & valid_id(target._item_key)]
        past = history[history.domain.eq(domain)]
        users = set(past.loc[valid_id(past.user_id), "user_id"])
        items = set(past.loc[valid_id(past._item_key), "_item_key"])
        if target.empty:
            continue
        cold_user, cold_item = ~target.user_id.isin(users), ~target._item_key.isin(items)
        rows.append({"domain": domain, "target_events": len(target),
                     "target_users": target.user_id.nunique(), "target_items": target._item_key.nunique(),
                     "cold_user_event_share": float(cold_user.mean()),
                     "cold_item_event_share": float(cold_item.mean()),
                     "cold_both_event_share": float((cold_user & cold_item).mean()),
                     "warm_both_event_share": float((~cold_user & ~cold_item).mean()),
                     "cold_target_user_share": len(set(target.user_id) - users) / target.user_id.nunique(),
                     "cold_target_item_share": len(set(target._item_key) - items) / target._item_key.nunique(),
                     "cold_domain_user_known_elsewhere_event_share": float((cold_user & target.user_id.isin(all_users)).mean())})
    return pd.DataFrame(rows)


def jensen_shannon(a: pd.Series, b: pd.Series) -> float:
    """Base-2 JS divergence [0, 1]; no observations => NaN, not 'no drift'."""
    a, b = a.align(b, join="outer", fill_value=0)
    p, q = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    if not len(p) or p.sum() == 0 or q.sum() == 0:
        return np.nan
    if (p < 0).any() or (q < 0).any():
        raise ValueError("Counts must be nonnegative")
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    def kl(x):
        mask = x > 0
        return np.sum(x[mask] * np.log2(x[mask] / m[mask]))
    return float((kl(p) + kl(q)) / 2)


@dataclass
class CatalogResult:
    frame: pd.DataFrame
    stats: dict
    ambiguous_keys: set[str]


def scan_catalog(file: str | Path, key: str, wanted_keys: Iterable,
                 batch_size: int = 65_536, max_tracked_keys: int = 1_000_000,
                 max_matched_rows: int = 2_000_000) -> CatalogResult:
    """Full streaming scan, retaining only rows relevant to the event sample.

    Never sample the catalog then label unseen keys 'orphans'. Global exact
    uniqueness is budgeted; after exceeding that budget it is explicitly unknown.
    Duplicate keys relevant to the sample are always found across batch borders
    and excluded from enrichment (no arbitrary first-row wins, no join explosion).
    """
    file = Path(file)
    empty = pd.DataFrame({key: pd.Series(dtype="string")})
    stats = {"file": file.as_posix(), "key": key, "status": "missing_file", "catalog_rows": np.nan,
             "scanned_rows": 0, "null_or_blank_keys": 0, "matched_rows": 0,
             "matched_duplicate_keys": 0, "unique_keys": np.nan, "duplicate_keys": np.nan}
    if not file.is_file():
        return CatalogResult(empty, stats, set())
    pf = pq.ParquetFile(file)
    columns = scalar_columns(pf.schema_arrow)
    stats["catalog_rows"] = pf.metadata.num_rows
    if key not in columns:
        stats["status"] = "missing_key_column"
        return CatalogResult(empty, stats, set())
    wanted = set(pd.Series(list(wanted_keys), dtype="string").dropna())
    counts: Counter | None = Counter()
    matches = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas(types_mapper=pd.ArrowDtype)
        for column in set(ID_COLUMNS) | {key}:
            if column in frame:
                frame[column] = frame[column].astype("string")
        valid = valid_id(frame[key])
        stats["scanned_rows"] += len(frame)
        stats["null_or_blank_keys"] += int((~valid).sum())
        if counts is not None:
            counts.update(frame.loc[valid, key])
            if len(counts) > max_tracked_keys:
                counts = None
        match = frame.loc[valid & frame[key].isin(wanted)]
        stats["matched_rows"] += len(match)
        if stats["matched_rows"] > max_matched_rows:
            raise MemoryError("Catalog lookup budget exceeded; reduce event sample, not the catalog scan")
        if not match.empty:
            matches.append(match)
    if stats["scanned_rows"] != stats["catalog_rows"]:
        raise ValueError("Catalog footer/scan mismatch")
    matched = pd.concat(matches, ignore_index=True) if matches else empty
    duplicate = matched.duplicated(key, keep=False)
    ambiguous = set(matched.loc[duplicate, key])
    lookup = matched.loc[~duplicate].copy()
    stats.update(status="ok", matched_duplicate_keys=len(ambiguous),
                 uniqueness_scope="entire_catalog" if counts is not None else "not_computed_key_budget",
                 unique_keys=len(counts) if counts is not None else np.nan,
                 duplicate_keys=sum(v > 1 for v in counts.values()) if counts is not None else np.nan)
    return CatalogResult(lookup, stats, ambiguous)


def join_catalog(events: pd.DataFrame, result: CatalogResult, left_key: str,
                 catalog_key: str, prefix: str) -> pd.DataFrame:
    """Validated many-to-one left join with explicit unresolved-key reasons."""
    enriched = events.copy()
    status_col = f"{prefix}match_status"
    if left_key not in events or result.stats["status"] != "ok":
        enriched[status_col] = "unavailable"
        return enriched
    right = result.frame.set_index(catalog_key).add_prefix(prefix)
    enriched = enriched.join(right, on=left_key, how="left", validate="many_to_one")
    enriched[status_col] = "unmatched"
    enriched.loc[events[left_key].isin(result.frame[catalog_key]), status_col] = "matched"
    enriched.loc[events[left_key].isin(result.ambiguous_keys), status_col] = "ambiguous_catalog_key"
    enriched.loc[~valid_id(events[left_key]), status_col] = "missing_event_key"
    assert len(enriched) == len(events)
    return enriched


def embedding_profile(values: pd.Series) -> dict:
    """Do not stack ragged/missing vectors; report each failure mode separately."""
    missing = invalid = empty = nonfinite = zero = norm_overflow = 0
    dimensions: Counter = Counter()
    norms = []
    for value in values:
        if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
            missing += 1
            continue
        try:
            vector = np.asarray(value, dtype=float)
        except (ValueError, TypeError):
            invalid += 1
            continue
        if vector.ndim != 1:
            invalid += 1
            continue
        dimensions[len(vector)] += 1
        if len(vector) == 0:
            empty += 1
        elif not np.isfinite(vector).all():
            nonfinite += 1
        else:
            scale = float(np.max(np.abs(vector)))
            with np.errstate(over="ignore"):
                norm = float(scale * np.linalg.norm(vector / scale)) if scale else 0.
            if np.isfinite(norm):
                norms.append(norm)
                zero += int(norm == 0)
            else:
                norm_overflow += 1
    return {"sample_rows": len(values), "missing": missing, "invalid_shape_or_type": invalid,
            "empty_vectors": empty, "nonfinite_vectors": nonfinite, "zero_vectors": zero,
            "norm_overflow_vectors": norm_overflow,
            "dimensions": dict(sorted(dimensions.items())),
            "mixed_nonempty_dimensions": len([d for d in dimensions if d > 0]) > 1,
            "norm_p50": float(np.quantile(norms, .5)) if norms else np.nan,
            "norm_p99": float(np.quantile(norms, .99)) if norms else np.nan}
