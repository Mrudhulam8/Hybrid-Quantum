import numpy as np
import pandas as pd

from src import data_loader as dl
from src.data_loader import (CIC_FEATURES, BENIGN, generate_synthetic_cic, load_cic_ids2017,
                             load_dataset, normalise_label, standardise_cic_frame)
from src.preprocessing import (Preprocessor, clean_frame, load_prepared, prepare_data,
                               save_prepared, stratified_subsample)
from src.utils import load_config


def _cfg(tmp_path=None, **data):
    d = {"source": "synthetic-cic-schema", "synthetic": {"n_rows": 8000, "data_seed": 0}}
    d.update(data)
    cfg = load_config(overrides={"data": d})
    if tmp_path is not None:
        cfg["data"]["raw_dir"] = str(tmp_path)
    return cfg


def test_cic_feature_list_is_78_columns():
    assert len(CIC_FEATURES) == 78 and len(set(CIC_FEATURES)) == 78


def test_label_normalisation_keeps_real_hyphens_and_repairs_mojibake():
    assert normalise_label("FTP-Patator") == "FTP-Patator"
    assert normalise_label("Web Attack \x96 Brute Force") == "Web Attack - Brute Force"
    assert normalise_label("Web Attack \ufffd XSS") == "Web Attack - XSS"
    assert normalise_label("benign") == BENIGN


def test_synthetic_has_cic_layout_and_quirks_and_is_deterministic():
    a, b = generate_synthetic_cic(4000, seed=3), generate_synthetic_cic(4000, seed=3)
    assert list(a.columns) == CIC_FEATURES + ["Label"]
    pd.testing.assert_frame_equal(a, b)
    num = a[CIC_FEATURES].to_numpy()
    assert np.isinf(num).any() and np.isnan(num).any()
    assert a["Label"].nunique() == 8


def test_real_csv_path_handles_cic_quirks(tmp_path):
    raw = generate_synthetic_cic(3000, seed=1)
    raw["Label"] = raw["Label"].replace("Web Attack - Brute Force", "Web Attack \x96 Brute Force")
    cols = [" " + c for c in CIC_FEATURES]
    cols[CIC_FEATURES.index("Fwd Header Length.1")] = " Fwd Header Length"   # true duplicate header
    raw.columns = cols + [" Label"]
    d = tmp_path / "cic-ids2017"
    d.mkdir()
    raw.to_csv(d / "Fake-Monday.csv", index=False, encoding="latin-1")
    ds = load_cic_ids2017({"data": {"raw_dir": str(tmp_path), "cic_subdir": "cic-ids2017"}})
    assert ds.X.shape[1] == 77 and ds.provenance["schema_compatible"]
    assert ds.provenance["dropped_duplicate_columns"] == ["Fwd Header Length.1"]
    assert "Web Attack - Brute Force" in set(ds.category)
    assert len(ds.provenance["files"]["Fake-Monday.csv"]["sha256"]) == 64
    assert not ds.provenance["is_synthetic"] and ds.provenance["paper_eligible"]


def test_missing_cic_raises_helpful_error(tmp_path):
    try:
        load_cic_ids2017({"data": {"raw_dir": str(tmp_path), "cic_subdir": "cic-ids2017"}})
        assert False
    except FileNotFoundError as e:
        assert "MachineLearningCSV" in str(e)


def test_auto_falls_back_to_labelled_synthetic(tmp_path, monkeypatch):
    def boom(cfg):
        raise RuntimeError("offline")
    monkeypatch.setitem(dl._LOADERS, "kdd99", boom)
    cfg = _cfg(tmp_path, source="auto")
    raw = load_dataset(cfg)
    p = raw.provenance
    assert p["source"] == "synthetic-cic-schema" and p["is_synthetic"] and not p["paper_eligible"]
    assert set(p["load_errors_before_success"]) == {"cic-ids2017", "kdd99"}


def test_clean_frame_removes_inf_nan_dupes():
    X = pd.DataFrame({"a": [1, 1, np.inf, 2.0], "b": [5, 5, 6, np.nan], "c": [np.nan] * 4})
    Xc, cat, rep = clean_frame(X, np.array(["x", "x", "y", "y"]))
    assert "c" not in Xc and rep["duplicates_removed"] == 1 and rep["inf_replaced"] == 1
    assert np.isfinite(Xc.to_numpy()).sum() + np.isnan(Xc.to_numpy()).sum() == Xc.size


def test_subsample_keeps_rare_classes_and_respects_cap():
    cat = np.array(["A"] * 9000 + ["B"] * 900 + ["R"] * 12)
    idx = stratified_subsample(cat, 2000, np.random.default_rng(0), min_per_class=50)
    sub = cat[idx]
    assert (sub == "R").sum() == 12 and len(idx) <= 2000 and len(set(idx)) == len(idx)


def test_preprocessor_train_only_stats_and_inverse_roundtrip():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame(rng.lognormal(2, 1, (500, 4)), columns=list("abcd"))
    tr["const"] = 7.0
    tr.iloc[0, 0] = np.nan
    te = pd.DataFrame(rng.lognormal(5, 1, (50, 4)), columns=list("abcd"))
    te["const"] = 7.0
    te.iloc[0, 0] = np.nan
    pre = Preprocessor("standard", True).fit(tr)
    assert "const" not in pre.feature_names_
    assert np.isclose(pre.medians_[0], np.nanmedian(tr["a"]))          # train median, not test
    Z = pre.transform(tr)
    assert np.allclose(Z.mean(0), 0, atol=1e-6) and np.allclose(Z.std(0), 1, atol=1e-6)
    back = pre.inverse_transform(pre.transform(te))
    ok = te[list("abcd")].to_numpy().copy()
    ok[0, 0] = pre.medians_[0]                                         # imputed cell
    assert np.allclose(back, ok, rtol=1e-6)


def test_prepare_data_split_integrity_and_determinism():
    cfg = _cfg()
    p1, p2 = prepare_data(cfg), prepare_data(cfg)
    assert np.array_equal(p1.X_train, p2.X_train) and np.array_equal(p1.cat_test, p2.cat_test)
    assert np.isfinite(p1.X_train).all() and np.isfinite(p1.X_test).all()
    assert set(p1.cat_train) == set(p1.cat_test)                        # stratified: rare class in both
    assert np.array_equal(p1.y_train, p1.y_train_binary)
    h = lambda A: {hash(r.tobytes()) for r in np.round(A, 9)}
    assert not (h(p1.X_train) & h(p1.X_test))                           # no row in both splits
    assert p1.provenance["is_synthetic"] and not p1.provenance["paper_eligible"]


def test_multiclass_task_and_max_rows():
    cfg = _cfg(task="multiclass", max_rows=3000, min_rows_per_class=40)
    p = prepare_data(cfg)
    assert len(p.X_train) + len(p.X_test) <= 3000 + 8 * 40
    assert len(p.class_names) == 8 and p.y_train.max() == 7


def test_save_load_roundtrip(tmp_path):
    p = prepare_data(_cfg())
    save_prepared(p, tmp_path, "t")
    q = load_prepared(tmp_path, "t")
    assert np.array_equal(p.X_test, q.X_test) and q.feature_names == p.feature_names
    assert np.allclose(q.preprocessor.inverse_transform(q.X_test[:5]),
                       p.preprocessor.inverse_transform(p.X_test[:5]))
