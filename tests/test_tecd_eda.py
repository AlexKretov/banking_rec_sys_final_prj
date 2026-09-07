"""Synthetic fixtures validate mechanics, never serve as T-ECD findings."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tecd_eda import (
    balanced_plan, cold_start_table, dataset_inventory, degree_summary,
    duplicate_profile, embedding_profile, find_eda_root, gini, jensen_shannon,
    join_catalog, load_eda_sample, overlap_table, parse_event_time,
    purged_day_split, purged_history, robust_scores, sample_parquet,
    scan_catalog, schema_missingness,
)


def write_table(file, data, row_group_size=7):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    table = data if isinstance(data, pa.Table) else pa.table(data)
    pq.write_table(table, file, row_group_size=row_group_size)
    return file


def event_file(root, domain, day, n=50, **extra):
    return write_table(root / domain / "events" / f"{day:05d}.pq", {
        "user_id": list(range(n)), "item_id": [f"item_{i % 8}" for i in range(n)],
        "action_type": ["view" if i % 2 else "click" for i in range(n)], **extra,
    })


def test_root_missing_and_event_only(tmp_path):
    assert find_eda_root(tmp_path) is None
    event_file(tmp_path / "dataset" / "small", "marketplace", 1)
    assert find_eda_root(tmp_path) == (tmp_path / "dataset" / "small").resolve()
    event_file(tmp_path / "dataset" / "full", "retail", 1)
    with pytest.raises(ValueError, match="Ambiguous"):
        find_eda_root(tmp_path, variant=None)
    assert find_eda_root(tmp_path, "full").name == "full"


def test_inventory_missing_empty_corrupt_and_integer_days(tmp_path):
    event_file(tmp_path, "marketplace", 2)
    write_table(tmp_path / "marketplace/events/00004.pq", {
        "user_id": pa.array([], type=pa.uint64()), "item_id": pa.array([], type=pa.string()),
    })
    (tmp_path / "marketplace/events/00005.pq").write_bytes(b"partial download")
    event_file(tmp_path, "marketplace", 10)
    inv = dataset_inventory(tmp_path, ["marketplace"], 2, 10)
    days = inv[inv.role.eq("events")].set_index("day")
    assert days.loc[2, "rows"] == 50
    assert days.loc[3, "status"] == "missing"
    assert pd.isna(days.loc[3, "rows"])
    assert days.loc[4, "status"] == "empty" and days.loc[4, "rows"] == 0
    assert days.loc[5, "status"] == "unreadable"
    assert list(days.index) == list(range(2, 11))
    assert "user_id" in days.loc[2, "schema"]


def test_duplicate_partition_names_are_not_double_counted(tmp_path):
    f = event_file(tmp_path, "offers", 1)
    (f.parent / "1.pq").write_bytes(f.read_bytes())
    (f.parent / "not-a-day.pq").write_bytes(f.read_bytes())
    inv = dataset_inventory(tmp_path, ["offers"], 1, 1)
    assert inv.status.eq("duplicate_partition").sum() == 2
    assert inv.status.eq("invalid_day_name").sum() == 1
    assert balanced_plan(inv, 100).empty


def test_all_days_domains_covered_no_prefix_weighted_counts_exact(tmp_path):
    for domain in ("marketplace", "offers", "retail"):
        for day in (1, 2, 3):
            event_file(tmp_path, domain, day, n=100)
    inv = dataset_inventory(tmp_path, ["marketplace", "offers", "retail"], 1, 3)
    events, plan, _ = load_eda_sample(inv, max_rows=180, seed=42)
    again, _, _ = load_eda_sample(inv, max_rows=180, seed=42)
    pd.testing.assert_frame_equal(events, again)
    assert len(events) == 180
    assert len(events.groupby(["day", "domain"])) == 9
    assert plan.sample_rows.eq(20).all()
    assert events.groupby("_source")._weight.sum().eq(100).all()
    assert events.groupby("_source")._row_in_file.max().gt(20).all()
    assert not events.duplicated(["_source", "_row_in_file"]).any()
    assert set(events.domain) == {"marketplace", "retail", "offers"}
    with pytest.raises(ValueError, match="every stratum"):
        load_eda_sample(inv, max_rows=8)


def test_water_fill_uses_small_partitions_and_total_budget(tmp_path):
    event_file(tmp_path, "marketplace", 1, n=1)
    event_file(tmp_path, "offers", 1, n=100)
    event_file(tmp_path, "retail", 1, n=100)
    inv = dataset_inventory(tmp_path, ["marketplace", "retail", "offers"], 1, 1)
    plan = balanced_plan(inv, 60)
    assert plan.sample_rows.sum() == 60
    assert sorted(plan.sample_rows) == [1, 29, 30]
    assert balanced_plan(inv, 500).sample_rows.sum() == 201
    with pytest.raises(ValueError):
        balanced_plan(inv, 0)


def test_large_nullable_uint64_identifiers_are_not_rounded(tmp_path):
    values = [2**64 - 2, None, 2**64 - 1]
    write_table(tmp_path / "marketplace/events/00001.pq", {
        "user_id": pa.array(values, type=pa.uint64()), "item_id": ["x", "y", None],
        "action_type": ["click", "view", "view"],
        "embedding": pa.array([[1., 2.], [3., 4.], None]),
    })
    inv = dataset_inventory(tmp_path, ["marketplace"], 1, 1)
    e, _, _ = load_eda_sample(inv, 20)
    assert list(e.user_id.dropna()) == [str(2**64 - 2), str(2**64 - 1)]
    assert e.user_id.isna().sum() == 1
    assert len(e) == 3  # missing required keys are not silently dropped before EDA
    assert e._item_key.isna().sum() == 1
    assert "embedding" not in e


def test_sample_size_zero_and_all(tmp_path):
    file = event_file(tmp_path, "retail", 1, n=13)
    assert sample_parquet(file, 0).empty
    assert list(sample_parquet(file, 100, batch_size=3)._row_in_file) == list(range(13))


def test_schema_absence_distinguished_from_null_cells(tmp_path):
    event_file(tmp_path, "marketplace", 1, n=2, os=["ios", None])
    event_file(tmp_path, "marketplace", 2, n=2)
    inv = dataset_inventory(tmp_path, ["marketplace"], 1, 2)
    e, _, _ = load_eda_sample(inv, 20)
    missing = schema_missingness(e, inv, ["os", "price"]).set_index("field")
    assert missing.loc["os", "null_share_when_present"] == .5
    assert missing.loc["os", "structural_absence_share"] == .5
    assert missing.loc["os", "pooled_null_share"] == .75
    assert missing.loc["price", "files_present"] == 0
    assert np.isnan(missing.loc["price", "null_share_when_present"])


@pytest.mark.parametrize("values", [[1234, 2345], ["1234", "2345"]])
def test_numeric_time_unit_is_never_guessed(values):
    t, audit = parse_event_time(pd.DataFrame({"timestamp": values}))
    assert t.isna().all()
    assert audit["status"] == "numeric_needs_unit_and_origin"
    assert audit["unparsed"] == 2


def test_explicit_numeric_time_and_datetime_parsing():
    t, report = parse_event_time(pd.DataFrame({"timestamp": [0, 3600]}), unit="s", origin="unix")
    assert t.iloc[1] == pd.Timestamp("1970-01-01T01:00:00Z")
    assert report["parsed"] == 2
    t, report = parse_event_time(pd.DataFrame({"timestamp": ["2026-01-01T03:00:00+03:00", "bad", None]}))
    assert t.iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert report["unparsed"] == 1
    arrow_dates = pa.table({"timestamp": pa.array([0, None], type=pa.timestamp("s", tz="UTC"))}).to_pandas(types_mapper=pd.ArrowDtype)
    t, _ = parse_event_time(arrow_dates)
    assert t.iloc[0] == pd.Timestamp("1970-01-01T00:00:00Z")


def test_unknown_optional_fields_do_not_prevent_loading(tmp_path):
    write_table(tmp_path / "reviews/00001.pq", {"user_id": [1], "brand_id": [7], "rating": [5]})
    inv = dataset_inventory(tmp_path, ["reviews"], 1, 1)
    e, _, clock = load_eda_sample(inv, 10)
    assert e._action.tolist() == ["review"]
    assert e._item_key.isna().all()
    assert clock.status.tolist() == ["absent"]
    assert degree_summary(e).empty


def test_robust_scores_small_constant_and_spike():
    assert np.isnan(robust_scores([1, 2, 3])).all()
    assert (robust_scores(np.ones(9)) == 0).all()
    assert np.isposinf(robust_scores([1] * 8 + [100])[-1])
    assert np.isnan(robust_scores([1] * 8 + [np.nan])[-1])


def sample_frame():
    return pd.DataFrame({
        "domain": ["marketplace"] * 4 + ["offers"] * 2,
        "user_id": ["u1", "u1", "u1", "u2", "u1", "u3"],
        "item_id": ["x", "x", "x", "y", "x", "z"],
        "_item_key": ["marketplace::x"] * 3 + ["marketplace::y", "offers::x", "offers::z"],
        "_action": ["click"] * 6, "day": [1] * 6,
        "_event_time": pd.to_datetime([None] * 6, utc=True),
        "_row_in_file": range(6), "_weight": [1, 2, 3, 4, 5, 6],
    })


def test_density_uses_unique_pairs_and_namespace():
    summary = degree_summary(sample_frame()).set_index("domain")
    assert summary.loc["marketplace", "density"] == .5  # 2 pairs / (2 users * 2 items), not 4/4
    assert summary.loc["marketplace", "repeated_pair_event_share"] == .5
    assert summary.loc["offers", "density"] == .5
    assert gini([1, 1, 1]) == 0
    assert np.isnan(gini([]))


def test_overlap_denominators_and_duplicate_scope():
    e = sample_frame()
    overlap = overlap_table(e, "user_id").set_index(["source", "target"])
    assert overlap.loc[("marketplace", "offers"), "jaccard"] == 1 / 3
    assert overlap.loc[("marketplace", "offers"), "p_target_given_source"] == 1 / 2
    repeats = duplicate_profile(e).set_index("domain")
    assert repeats.loc["marketplace", "scalar_repeat_excess"] == 2


def test_catalog_complete_scan_late_keys_and_duplicates_across_batches(tmp_path):
    file = write_table(tmp_path / "items.pq", {
        "item_id": ["x", "other", "late", None, "x"], "brand_id": [1, 2, 3, 4, 8],
        "embedding": [[1., 2.]] * 5,
    })
    result = scan_catalog(file, "item_id", ["x", "late", "absent"], batch_size=2)
    assert result.stats["scanned_rows"] == 5
    assert result.stats["null_or_blank_keys"] == 1
    assert result.stats["duplicate_keys"] == 1
    assert result.stats["matched_duplicate_keys"] == 1
    assert result.ambiguous_keys == {"x"}
    assert result.frame.item_id.tolist() == ["late"]
    e = pd.DataFrame({"item_id": ["x", "late", "late", "absent", None]})
    joined = join_catalog(e, result, "item_id", "item_id", "item__")
    assert len(joined) == len(e)
    assert joined.item__match_status.tolist() == ["ambiguous_catalog_key", "matched", "matched", "unmatched", "missing_event_key"]
    assert joined.loc[1, "item__brand_id"] == "3"


def test_catalog_absence_is_not_an_orphan(tmp_path):
    result = scan_catalog(tmp_path / "absent.pq", "item_id", ["x"])
    joined = join_catalog(pd.DataFrame({"item_id": ["x"]}), result, "item_id", "item_id", "item__")
    assert joined.item__match_status.tolist() == ["unavailable"]


def test_catalog_global_uniqueness_budget_is_explicit(tmp_path):
    file = write_table(tmp_path / "items.pq", {"item_id": ["a", "b", "c", "a"]})
    result = scan_catalog(file, "item_id", ["a"], batch_size=2, max_tracked_keys=1)
    assert np.isnan(result.stats["unique_keys"])
    assert result.stats["uniqueness_scope"] == "not_computed_key_budget"
    assert result.stats["matched_duplicate_keys"] == 1
    with pytest.raises(MemoryError):
        scan_catalog(file, "item_id", ["a"], max_matched_rows=1)


def test_purged_day_split_calendar_gaps_and_short_history():
    e = pd.DataFrame({"day": list(range(1, 10))})
    split, intervals = purged_day_split(e)
    assert split.loc[split._split.eq("train"), "day"].tolist() == [1, 2, 3]
    assert split.loc[split._split.eq("validation"), "day"].tolist() == [5, 6]
    assert split.loc[split._split.eq("test"), "day"].tolist() == [8, 9]
    assert split.loc[split._split.str.startswith("gap"), "day"].tolist() == [4, 7]
    assert intervals.sample_events.sum() == len(e)
    with pytest.raises(ValueError, match="Insufficient"):
        purged_day_split(e.head(1))
    with pytest.raises(ValueError, match="validation"):
        purged_day_split(e[~e.day.isin([5, 6])])
    with pytest.raises(ValueError):
        purged_day_split(e, gap_days=0)


def test_twelve_hour_rule_and_unknown_time_exclusion():
    e = pd.DataFrame({"_event_time": pd.to_datetime([
        "2026-01-01T11:59:00Z", "2026-01-01T12:00:00Z", "2026-01-01T23:59:00Z", None,
    ], utc=True)})
    assert len(purged_history(e, "2026-01-02T00:00:00Z")) == 1
    with pytest.raises(ValueError):
        purged_history(e, "2026-01-02")
    with pytest.raises(ValueError):
        purged_history(e, "2026-01-02T00:00:00Z", gap_hours=11)


def test_cold_start_uses_target_domain_and_namespaced_items():
    history = sample_frame()
    targets = pd.DataFrame({"domain": ["retail", "offers"], "user_id": ["u1", "u3"],
                            "_item_key": ["retail::x", "offers::z"]})
    cold = cold_start_table(history, targets).set_index("domain")
    assert cold.loc["retail", "cold_both_event_share"] == 1
    assert cold.loc["retail", "cold_domain_user_known_elsewhere_event_share"] == 1
    assert cold.loc["offers", "warm_both_event_share"] == 1


def test_js_divergence_empty_equal_disjoint():
    a, b = pd.Series({"view": 2}), pd.Series({"click": 2})
    assert jensen_shannon(a, a) == 0
    assert jensen_shannon(a, b) == 1
    assert np.isnan(jensen_shannon(pd.Series(dtype=float), a))


def test_embedding_failures_do_not_require_stacking():
    report = embedding_profile(pd.Series([
        None, [0., 0.], [1., np.inf], [1., 2., 3.], [], [[1., 2.]], "bad", [3., 4.],
    ]))
    assert report["sample_rows"] == 8
    assert report["missing"] == 1
    assert report["zero_vectors"] == 1
    assert report["nonfinite_vectors"] == 1
    assert report["empty_vectors"] == 1
    assert report["invalid_shape_or_type"] == 2
    assert report["mixed_nonempty_dimensions"]
    assert report["dimensions"] == {0: 1, 2: 3, 3: 1}


def test_numeric_clock_keeps_nanoseconds_with_nulls_and_offset():
    values = pd.Series([10**18, None, 10**18 + 1], dtype="UInt64")
    for origin in ("unix", "2000-01-01"):
        t, report = parse_event_time(pd.DataFrame({"timestamp": values}), unit="ns", origin=origin)
        assert t.iloc[2] - t.iloc[0] == pd.Timedelta(1, unit="ns")
        assert report["parsed"] == 2
        assert pd.isna(t.iloc[1])
    t, report = parse_event_time(pd.DataFrame({"timestamp": pd.Series([2**64 - 1], dtype="UInt64")}), unit="ns", origin="unix")
    assert t.isna().all()  # No unsigned overflow into a believable historical date.
    assert report["unparsed"] == 1


def test_numeric_profile_separates_parse_failure_nan_inf_and_fractional_count():
    from tecd_eda import numeric_profile
    e = pd.DataFrame({"domain": ["retail"] * 8, "_action": ["order"] * 8,
                      "count": ["1", "1.5", None, "bad", "inf", "-3", "0", "NaN"]})
    report = numeric_profile(e).iloc[0]
    assert report["missing"] == report["nonnumeric"] == report["infinite"] == report["nan_values"] == 1
    assert report["negative"] == report["zero"] == report["fractional"] == 1
    assert report["tail_tested"] == 0
    assert report["sample_rows"] == 8


def test_split_does_not_shift_requested_cutoff_when_tail_is_missing():
    e = pd.DataFrame({"day": list(range(1, 10))})
    with pytest.raises(ValueError, match="test"):
        purged_day_split(e, day_begin=1, day_end=12)
    split, intervals = purged_day_split(e, day_begin=1, day_end=10)
    test = intervals.set_index("split").loc["test"]
    assert test.day_from == 9 and test.day_to == 10
    assert test.calendar_days == 2 and test.observed_days == 1
    with pytest.raises(ValueError, match="inside"):
        purged_day_split(e, day_begin=2, day_end=9)


def test_raw_partition_fields_are_retained_for_mismatch_audit(tmp_path):
    write_table(tmp_path / "marketplace/events/00001.pq", {
        "user_id": [1, 2], "item_id": ["a", "b"], "action_type": ["view", "view"],
        "day": [99, 98], "domain": ["wrong", "wrong"],
    })
    inv = dataset_inventory(tmp_path, ["marketplace"], 1, 1)
    e, _, _ = load_eda_sample(inv, 10)
    assert e.day.tolist() == [1, 1]
    assert e.domain.tolist() == ["marketplace", "marketplace"]
    assert e._raw_day.tolist() == [99, 98]
    assert e._raw_domain.tolist() == ["wrong", "wrong"]


def test_embedding_norm_is_stable_and_overflow_is_reported():
    report = embedding_profile(pd.Series([[1e308, 1e308], [1e308] * 4]))
    assert report["nonfinite_vectors"] == 0
    assert report["norm_overflow_vectors"] == 1
    assert np.isfinite(report["norm_p50"])
    assert report["norm_p50"] / 1e308 == pytest.approx(np.sqrt(2))


def test_report_filenames_never_use_data_labels_as_paths():
    from tecd_eda import report_filename
    for label in ("../outside", "/absolute/path", "..", "users_/../../file", "Поле/регион", "x" * 300):
        filename = report_filename(label)
        assert Path(filename).name == filename
        assert "/" not in filename and "\\" not in filename
        assert filename == report_filename(label)
    assert report_filename("numeric_quality") == "numeric_quality.csv"
