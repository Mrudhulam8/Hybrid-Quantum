"""Dataset loading with ONE canonical schema (CIC-IDS2017 column names).

Sources
-------
cic-ids2017          PRIMARY. Real CSVs placed in data/raw/cic-ids2017/ (MachineLearningCSV).
kdd99                FALLBACK 1. Real public intrusion data (sklearn fetch_kddcup99; needs
                     internet on first use). Different features, same RawDataset contract.
synthetic-cic-schema FALLBACK 2. Offline generator that reproduces the CIC-IDS2017 column
                     layout *including its data-quality problems* (inf, NaN, duplicate
                     column, duplicate rows). FOR PIPELINE DEVELOPMENT AND TESTS ONLY:
                     provenance['paper_eligible'] is False and results must not be reported.

Every loader returns a `RawDataset` (features DataFrame in original units, string
category per row, provenance dict), so downstream code never knows which source it got.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger, resolve_path

log = get_logger("data_loader")

BENIGN = "BENIGN"

# 78 feature columns of CIC-IDS2017 MachineLearningCVE files (pandas renames the
# second 'Fwd Header Length' to 'Fwd Header Length.1'; it is an exact duplicate).
CIC_FEATURES: List[str] = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
CIC_LABEL = "Label"
# Identifier columns present in the 'GeneratedLabelledFlows' variant; never features.
ID_COLUMNS = {"Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp",
              "SimillarHTTP", "Inbound"}


@dataclass
class RawDataset:
    X: pd.DataFrame            # features, ORIGINAL units (may contain inf/NaN/duplicates)
    category: np.ndarray       # str per row; 'BENIGN' or attack name/family
    provenance: Dict[str, Any]

    @property
    def y_binary(self) -> np.ndarray:
        return (self.category != BENIGN).astype(np.int8)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalise_label(s: Any) -> str:
    """Map raw CIC labels to stable ASCII names ('Web Attack \x96 XSS' -> 'Web Attack - XSS')."""
    s = str(s)
    s = re.sub(r"[^\x20-\x7E]+", " - ", s)        # damaged non-ASCII dash -> ' - ' (ASCII hyphens untouched)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return BENIGN if s.upper() == "BENIGN" else s


def standardise_cic_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Common path for ALL CIC-layout data (real CSVs and the synthetic generator)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if CIC_LABEL not in df.columns:
        lab = [c for c in df.columns if c.lower() == "label"]
        if not lab:
            raise ValueError("No 'Label' column found; is this a CIC-IDS2017 CSV?")
        df = df.rename(columns={lab[0]: CIC_LABEL})
    category = np.array([normalise_label(v) for v in df[CIC_LABEL].to_numpy()], dtype=object)
    df = df.drop(columns=[CIC_LABEL])
    df = df.drop(columns=[c for c in df.columns if c in ID_COLUMNS])

    dropped_dupe_cols = []
    for c in list(df.columns):
        m = re.fullmatch(r"(.+)\.\d+", c)
        if m and m.group(1) in df.columns:
            a, b = df[c].to_numpy(), df[m.group(1)].to_numpy()
            if np.array_equal(a, b, equal_nan=True):
                df = df.drop(columns=[c])
                dropped_dupe_cols.append(c)

    X = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    expected = [c for c in CIC_FEATURES if c not in dropped_dupe_cols]
    known = [c for c in expected if c in X.columns]
    extra = [c for c in X.columns if c not in expected]
    X = X[known + extra]
    info = {
        "schema": "cic-ids2017",
        "schema_missing": [c for c in expected if c not in X.columns],
        "schema_extra": extra,
        "dropped_duplicate_columns": dropped_dupe_cols,
        "schema_compatible": len([c for c in expected if c not in X.columns]) == 0,
    }
    return X.reset_index(drop=True), category.astype(str), info


# --------------------------------------------------------------------------
# Source 1: real CIC-IDS2017
# --------------------------------------------------------------------------
def load_cic_ids2017(cfg: Dict[str, Any]) -> RawDataset:
    d = resolve_path(cfg["data"]["raw_dir"]) / cfg["data"].get("cic_subdir", "cic-ids2017")
    files = sorted(d.rglob("*.csv")) if d.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No CIC-IDS2017 CSVs found in {d}. Download 'MachineLearningCSV.zip' from the UNB "
            "CIC-IDS2017 page and extract the *.csv files there.")
    frames, cats, hashes, info = [], [], {}, {}
    for f in files:
        log.info("Reading %s", f.name)
        raw = pd.read_csv(f, encoding="latin-1", skipinitialspace=True, low_memory=False)
        X, c, info = standardise_cic_frame(raw)
        frames.append(X)
        cats.append(c)
        hashes[f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size, "rows": len(X)}
    X = pd.concat(frames, ignore_index=True)
    prov = {"source": "cic-ids2017", "role": "primary", "is_synthetic": False,
            "paper_eligible": True, "files": hashes, "n_rows_raw": len(X), **info,
            "notes": "Day-wise CSVs concatenated; random (non-temporal) split is applied later."}
    return RawDataset(X, np.concatenate(cats), prov)


# --------------------------------------------------------------------------
# Source 2: KDD Cup 99 (real, public fallback)
# --------------------------------------------------------------------------
KDD_FAMILY = {
    "normal": BENIGN,
    **{k: "DoS" for k in ["back", "land", "neptune", "pod", "smurf", "teardrop"]},
    **{k: "Probe" for k in ["ipsweep", "nmap", "portsweep", "satan"]},
    **{k: "R2L" for k in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                          "warezclient", "warezmaster"]},
    **{k: "U2R" for k in ["buffer_overflow", "loadmodule", "perl", "rootkit"]},
}
KDD_CATEGORICAL = ["protocol_type", "service", "flag"]


def load_kdd99(cfg: Dict[str, Any]) -> RawDataset:
    """KDD Cup 99 (10% subset). Categorical columns are dropped: a continuous search space
    cannot meaningfully perturb 'service' or 'flag'. 38 numeric features remain."""
    from sklearn.datasets import fetch_kddcup99
    home = resolve_path(cfg["data"]["raw_dir"]) / "kdd99"
    home.mkdir(parents=True, exist_ok=True)
    try:
        b = fetch_kddcup99(subset=None, data_home=str(home), percent10=True, as_frame=True,
                           download_if_missing=True)
    except Exception as e:  # network / cache problems
        raise RuntimeError(f"Could not obtain KDD99 (needs internet on first run): {e}") from e
    df = b.frame.copy()
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in df["labels"]]
    labels = [l.rstrip(".") for l in labels]
    category = np.array([KDD_FAMILY.get(l, "Other-Attack") for l in labels], dtype=str)
    df = df.drop(columns=["labels"] + [c for c in KDD_CATEGORICAL if c in df.columns])
    X = df.apply(pd.to_numeric, errors="coerce").astype(np.float32).reset_index(drop=True)
    prov = {"source": "kdd99", "role": "fallback", "is_synthetic": False, "paper_eligible": True,
            "schema": "kdd99-numeric", "schema_compatible": False,
            "n_rows_raw": len(X), "n_features": X.shape[1],
            "notes": "FALLBACK dataset, NOT CIC-IDS2017. Categorical columns dropped. Known to be "
                     "easy (>99.9% accuracy) with many duplicates; failures will be very rare."}
    return RawDataset(X, category, prov)


# --------------------------------------------------------------------------
# Source 3: synthetic data in CIC-IDS2017 layout (offline; tests only)
# --------------------------------------------------------------------------
SYNTH_CLASSES = {BENIGN: 0.80, "DoS Hulk": 0.06, "PortScan": 0.05, "DDoS": 0.04,
                 "FTP-Patator": 0.02, "Bot": 0.015, "Web Attack - Brute Force": 0.01,
                 "Infiltration": 0.005}


def generate_synthetic_cic(n_rows: int = 60000, seed: int = 0,
                           structure_seed: int = 12345) -> pd.DataFrame:
    """Raw-format frame (78 feature columns + 'Label') mimicking CIC-IDS2017 layout and quirks.

    `structure_seed` fixes the generative model; `seed` fixes the sample. The data is built so
    that some attack flows resemble benign traffic (and vice versa), producing genuine
    classifier errors rather than a trivially separable toy set. It is still synthetic:
    NOT valid evidence about real intrusion detectors.
    """
    rng, srng = np.random.default_rng(seed), np.random.default_rng(structure_seed)
    names = [c for c in CIC_FEATURES if c != "Fwd Header Length.1"]
    d, k = len(names), 10
    W = srng.normal(0, 0.5, (d, k))
    base = srng.uniform(2.0, 6.0, d)
    classes = list(SYNTH_CLASSES)
    attack_off = {}
    for cls in classes[1:]:
        off = np.zeros(d)
        sel = srng.choice(d, 14, replace=False)
        off[sel] = srng.choice([-1.0, 1.0], 14) * srng.uniform(1.2, 3.0, 14)
        attack_off[cls] = off
    O = np.stack(list(attack_off.values()))
    probs = np.array(list(SYNTH_CLASSES.values()))
    labels = rng.choice(classes, size=n_rows, p=probs / probs.sum())

    z = np.empty((n_rows, d))
    for cls in classes:
        m = np.flatnonzero(labels == cls)
        n = len(m)
        if n == 0:
            continue
        if cls == BENIGN:
            shift = np.zeros((n, d))
            odd = rng.random(n) < 0.01                      # benign flows that look malicious
            shift[odd] = 0.5 * O[rng.integers(0, len(O), odd.sum())]
        else:
            strength = np.where(rng.random(n) < 0.15, 0.25, 1.0)   # stealthy attack variants
            shift = strength[:, None] * attack_off[cls][None, :]
        z[m] = base + shift + rng.normal(size=(n, k)) @ W.T + rng.normal(0, 0.7, (n, d))
    z = np.clip(z, 0, 14)
    x = np.expm1(z)
    for j, nm in enumerate(names):
        if "Flag Count" in nm or "Flags" in nm:
            x[:, j] = (rng.random(n_rows) < 1 / (1 + np.exp(-(z[:, j] - base[j])))).astype(float)
        elif nm.startswith("Init_Win_bytes"):
            x[:, j] = np.where(rng.random(n_rows) < 0.3, -1.0, np.round(x[:, j]))
        elif nm == "Destination Port":
            x[:, j] = np.round(np.clip(x[:, j], 0, 65535))
    df = pd.DataFrame(x, columns=names)
    # CIC quirks: inf / NaN in rate features, exact-duplicate rows, duplicated column
    for nm, p_inf, p_nan in [("Flow Bytes/s", 0.002, 0.001), ("Flow Packets/s", 0.002, 0.0)]:
        r = rng.random(n_rows)
        df.loc[r < p_inf, nm] = np.inf
        df.loc[(r >= p_inf) & (r < p_inf + p_nan), nm] = np.nan
    m = max(1, n_rows // 100)
    df.iloc[-m:] = df.iloc[:m].to_numpy()
    labels = labels.copy()
    labels[-m:] = labels[:m]
    df.insert(CIC_FEATURES.index("Fwd Header Length.1"), "Fwd Header Length.1",
              df["Fwd Header Length"])
    df = df[CIC_FEATURES]
    df[CIC_LABEL] = labels
    return df


def load_synthetic_cic(cfg: Dict[str, Any]) -> RawDataset:
    sc = cfg["data"].get("synthetic", {})
    raw = generate_synthetic_cic(sc.get("n_rows", 60000), sc.get("data_seed", 0))
    X, cat, info = standardise_cic_frame(raw)
    prov = {"source": "synthetic-cic-schema", "role": "offline-dev-only", "is_synthetic": True,
            "paper_eligible": False, "n_rows_raw": len(X), "generator": sc, **info,
            "notes": "SYNTHETIC. Pipeline development/testing only. Do NOT report as results."}
    return RawDataset(X, cat, prov)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
_LOADERS = {"cic-ids2017": load_cic_ids2017, "kdd99": load_kdd99,
            "synthetic-cic-schema": load_synthetic_cic}


def _banner(prov: Dict[str, Any]) -> None:
    if prov["source"] == "cic-ids2017":
        log.info("Dataset: CIC-IDS2017 (primary).")
    else:
        bar = "=" * 72
        log.warning("%s\n USING FALLBACK DATASET: %s  (paper_eligible=%s)\n %s\n%s",
                    bar, prov["source"], prov["paper_eligible"], prov["notes"], bar)


def load_dataset(cfg: Dict[str, Any]) -> RawDataset:
    src = cfg["data"].get("source", "auto")
    order = cfg["data"].get("fallback_order", list(_LOADERS)) if src == "auto" else [src]
    errors = {}
    for s in order:
        try:
            raw = _LOADERS[s](cfg)
            _banner(raw.provenance)
            raw.provenance["requested_source"] = src
            raw.provenance["load_errors_before_success"] = errors
            return raw
        except Exception as e:
            errors[s] = str(e)
            if src != "auto":
                raise
            log.warning("Source '%s' unavailable (%s); trying next fallback.", s, str(e)[:120])
    raise RuntimeError(f"No dataset source could be loaded: {errors}")
