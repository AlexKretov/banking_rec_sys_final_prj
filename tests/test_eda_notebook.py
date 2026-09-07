"""Execute all notebook branches on explicit synthetic fixtures, not real T-ECD.

Executed fixture notebooks are deliberately not written over eda.ipynb. Only
aggregate reports are written into pytest's temporary directories.
"""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def write(file, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), file, row_group_size=17)


def rich_fixture(root):
    users = [*range(1, 101), 2**64 - 1]
    write(root / "users.pq", {"user_id": pa.array(users, type=pa.uint64()),
                             "region": [i % 3 for i in range(len(users))],
                             "cluster": [i % 2 for i in range(len(users))]})
    write(root / "brands.pq", {"brand_id": [1, 2, 3, 4, None],
                              "embedding": [[1., 0.], [0., 0.], [np.inf, 1.], [1., 2., 3.], None]})
    for domain in ("marketplace", "retail", "offers"):
        write(root / domain / "items.pq", {
            "item_id": ["shared", "a", "b", "duplicate", "duplicate", None],
            "brand_id": [1, 2, 3, 2, 4, 1], "category_id": [1, 2, 2, 1, 2, 1],
            "embedding": [[1., 2.], [], [0., 0.], [1., 2., 3.], [1., np.nan], None],
        })
        for day in range(1, 10):
            n = 180
            ids = [1 if i < 65 else users[(i + day * 11) % len(users)] for i in range(n)]
            ids[3] = None
            actions = (["view", "click", "order", "cart", "unknown_action"] if domain == "retail"
                       else ["view", "click", "clickout", "like", "unknown_action"])
            data = {
                "user_id": pa.array(ids, type=pa.uint64()),
                "item_id": [["shared", "a", "b", "duplicate", "out_of_catalog", None][i % 6] for i in range(n)],
                "action_type": [actions[i % len(actions)] for i in range(n)],
                "timestamp": [f"2026-01-{day:02d}T{i % 24:02d}:00:00Z" if i % 17 else "bad_time" for i in range(n)],
                "price": [["100", "500", "0", "-5", "bad", "inf", None][i % 7] for i in range(n)],
                "count": [[1., 2., 1.5, 0., -1., np.inf, None][i % 7] for i in range(n)],
            }
            if domain == "marketplace" and day == 2:
                data["domain"] = ["wrong_source_domain"] * n
                data["day"] = [day] * n
            if domain != "offers":
                data["os"] = ["ios" if i % 2 else "android" for i in range(n)]
                if day != 3:
                    data["subdomain"] = ["catalog" if i % 3 else None for i in range(n)]
            if domain == "retail":
                # Catalog conflicts are deliberate test cases, not actual T-ECD findings.
                data["brand_id"] = [4 if i % 4 else None for i in range(n)]
            write(root / domain / "events" / f"{day:05d}.pq", data)
    for day in range(1, 10):
        write(root / "reviews" / f"{day:05d}.pq", {
            "user_id": list(range(1, 61)), "brand_id": [i % 4 + 1 for i in range(60)],
            "rating": [i % 7 if i % 13 else None for i in range(60)],
            "timestamp": [day * 86400 + i for i in range(60)],  # unit/origin deliberately unspecified
            "embedding": [[1., 2.] if i % 2 else [1., 2., 3.] for i in range(60)],
        })
        write(root / "payments/events" / f"{day:05d}.pq", {
            "user_id": [1, 2, 3, 4], "brand_id": [1, 2, 3, None],
            "transaction_id": [f"tx{day}_{i}" for i in range(4)],
            "amount": [10., 0., -3., np.inf],
        })
        write(root / "payments/receipts" / f"{day:05d}.pq", {
            "transaction_id": [f"tx{day}_0", f"tx{day}_1"],
            "lines": [[{"item_id": "shared", "count": 1}], [{"item_id": "a", "count": 2}]],
        })


@pytest.mark.parametrize("mode", ["no_data", "partial", "rich_full", "rich_sample"])
def test_notebook_executes_without_fabricating_data(tmp_path, monkeypatch, mode):
    root = tmp_path / "synthetic_t_ecd"
    report = tmp_path / "reports"
    if mode.startswith("rich"):
        rich_fixture(root)
    elif mode == "partial":
        write(root / "marketplace/events/00005.pq", {
            "user_id": [1, 1, None], "item_id": ["x", "x", None], "action_type": ["view"] * 3,
        })
        bad = root / "offers/events/00001.pq"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"incomplete parquet")
        write(root / "retail/events/00002.pq", {"user_id": pa.array([], type=pa.uint64()),
                                               "item_id": pa.array([], type=pa.string())})
    monkeypatch.setenv("TECD_EDA_DATA_DIR", str(root))
    monkeypatch.setenv("TECD_EDA_REPORT_DIR", str(report))
    monkeypatch.setenv("TECD_EDA_DAY_BEGIN", "1")
    monkeypatch.setenv("TECD_EDA_DAY_END", "9")
    monkeypatch.setenv("TECD_EDA_MAX_ROWS", "800" if mode == "rich_sample" else "100000")
    monkeypatch.setenv("TECD_EDA_VARIANT", "full" if mode.startswith("rich") else "small")
    monkeypatch.delenv("TECD_EDA_DOMAINS", raising=False)
    if mode == "no_data":
        report.mkdir()
        (report / "old_owned.csv").write_text("stale previous result")
        (report / "notes.csv").write_text("user notes")
        (tmp_path / "keep_me.csv").write_text("outside report directory")
        (report / "run_info.json").write_text(json.dumps({"artifact_files": ["old_owned.csv", "../keep_me.csv"]}))
    original = (PROJECT / "eda.ipynb").read_bytes()
    nb = nbformat.reads(original.decode(), as_version=4)
    if mode == "rich_full":
        parameters = next(c for c in nb.cells if c.id == "parameters")
        parameters.source = parameters.source.replace('TIME_CONFIG = {}', 'TIME_CONFIG = {"reviews": {"unit": "s", "origin": "2025-12-31"}}')
        parameters.source = parameters.source.replace('DAY_ZERO = None', 'DAY_ZERO = pd.Timestamp("2025-12-31", tz="UTC")')
        parameters.source = parameters.source.replace('PREDICTION_TIME = None', 'PREDICTION_TIME = pd.Timestamp("2026-01-09T00:00:00Z")')
        parameters.source = parameters.source.replace('RATING_RANGE = None', 'RATING_RANGE = (1, 5)')
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(PROJECT)}}).execute()
    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
    info = json.loads((report / "run_info.json").read_text())
    findings = json.loads((report / "findings.json").read_text())
    assert info["raw_analysis_executed"] is (mode != "no_data")
    assert info["reference_provenance"]["kind"] == "saved_notebook_outputs_not_a_new_raw_data_run"
    if mode.startswith("rich"):
        for artifact in ("observed_cold_start.csv", "numeric_quality.csv", "embedding_quality.csv",
                         "catalog_match_status.csv", "distribution_drift.csv", "review_rating_distribution.csv",
                         "receipt_sample_plan.csv", "embedding_dimensions_across_files.csv", "source_vs_partition_metadata.csv"):
            assert (report / artifact).exists(), artifact
        assert info["sample_is_census_of_readable_strata"] is (mode == "rich_full")
        if mode == "rich_full":
            assert (report / "partition_clock_delta.csv").exists()
            assert any(f["check"] == "L17_rating_range" for f in findings)
        else:
            assert info["sample_rows"] == 800
        assert any(f["check"] == "L04_source_partition" and f["status"] == "кандидат" for f in findings)
        assert any(f["check"] == "L05_unknown_actions" for f in findings)
        assert any(f["check"] == "L12_catalog_integrity" and f["status"] == "кандидат" for f in findings)
    elif mode == "partial":
        assert any(f["check"] == "L19_temporal_split" and f["status"] == "не выполнено" for f in findings)
        assert any(f["check"] == "L12_catalog_integrity" and f["status"] == "не выполнено" for f in findings)
    else:
        assert not (report / "old_owned.csv").exists()
        assert (report / "notes.csv").read_text() == "user notes"
        assert (tmp_path / "keep_me.csv").exists()
        assert info["sample_rows"] == 0
        assert info["manifest_sha256"] is None
        assert any(f["check"] == "L00_raw_data" and f["status"] == "не выполнено" for f in findings)
    # Prevent accidental replacement of honest checked-in outputs by fixture outputs.
    assert (PROJECT / "eda.ipynb").read_bytes() == original
