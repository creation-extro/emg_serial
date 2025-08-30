from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from motion_ai.preprocess.filters import clean_emg
from motion_ai.features.extractors import extract_features_from_window


LABEL_MAP = {
    "1-CLOSE": "fist",
    "0-OPEN": "open",
    "RELAX": "rest",
    "3-POINT": "step",
    "4-FOUR": "step",
    "5-FIVE": "step",
    "6-PEACE": "step",
    -1: "rest",
    0: "open",
    1: "fist",
    2: "step",
    3: "step",
    4: "step",
    5: "step",
    6: "step",
}


@dataclass
class TrainConfig:
    fs: float = 1000.0
    window_ms: int = 200
    hop_ms: int = 100
    mains_freq: float = 50.0
    notch_q: float = 30.0
    band_low: float = 20.0
    band_high: float = 450.0


def window_indices(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    i = 0
    out = []
    while i + win <= n:
        out.append((i, i + win))
        i += hop
    return out


def load_dataset(csv_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No CSVs could be loaded")
    return pd.concat(frames, ignore_index=True)


def infer_emg_matrix(df: pd.DataFrame) -> np.ndarray:
    for cols in (['emg1_clean', 'emg2_clean', 'emg3_clean'], ['ch1', 'ch2', 'ch3'], ['raw_emg1', 'raw_emg2']):
        if all(c in df.columns for c in cols):
            X = df[cols].to_numpy(dtype=float)
            return X.T
    num_df = df.select_dtypes(include=[np.number])
    if 'label' in num_df.columns:
        num_df = num_df.drop(columns=['label'])
    return num_df.to_numpy(dtype=float).T


def infer_label_series(df: pd.DataFrame) -> pd.Series:
    if 'gesture' in df.columns:
        raw = df['gesture']
    elif 'label' in df.columns:
        raw = df['label']
    else:
        if 'gesture' in df.columns:
            raw = df['gesture']
        else:
            raise ValueError('No label/gesture column found')

    def norm_one(x):
        return LABEL_MAP.get(x, str(x).lower())

    return raw.apply(norm_one)


def build_feature_table(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.Series]:
    emg = infer_emg_matrix(df)
    labels = infer_label_series(df)

    win = int(cfg.fs * cfg.window_ms / 1000.0)
    hop = int(cfg.fs * cfg.hop_ms / 1000.0)

    emg_clean = clean_emg(emg, cfg.fs, mains_freq=cfg.mains_freq, band_low=cfg.band_low, band_high=cfg.band_high, notch_q=cfg.notch_q)

    n_samples = emg_clean.shape[1]
    idxs = window_indices(n_samples, win, hop)

    feat_rows: List[Dict[str, float]] = []
    y_rows: List[str] = []

    for a, b in idxs:
        df_window = pd.DataFrame({f"ch{i+1}": emg_clean[i, a:b] for i in range(emg_clean.shape[0])})
        feats = extract_features_from_window(df_window, fs=cfg.fs)
        feat_rows.append(feats)
        if len(labels) == n_samples:
            lab = labels.iloc[a:b].mode()
            y_rows.append(str(lab.iloc[0]) if len(lab) > 0 else 'rest')
        else:
            y_rows.append(str(labels.mode().iloc[0]))

    X = pd.DataFrame(feat_rows).fillna(0.0)
    y = pd.Series(y_rows)
    return X, y


def train_mlp(csv_paths: List[str], cfg: TrainConfig, model_out: str) -> Dict[str, float]:
    df = load_dataset(csv_paths)
    X, y = build_feature_table(df, cfg)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # 1-hidden-layer MLP
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=300, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y.values, test_size=0.2, random_state=42, stratify=y.values)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    acc = float(accuracy_score(yte, ypred))
    report = classification_report(yte, ypred, zero_division=0)

    joblib.dump({
        'scaler': scaler,
        'clf': clf,
        'features': list(X.columns),
        'classes_': list(sorted(set(y.values))),
    }, model_out)

    return {'accuracy': acc, 'n_train': int(len(ytr)), 'n_test': int(len(yte)), 'report': report}


def predict_window_with_confidence(model_path: str, df_window: pd.DataFrame, fs: float) -> Tuple[str, float]:
    bundle = joblib.load(model_path)
    scaler: StandardScaler = bundle['scaler']
    clf: MLPClassifier = bundle['clf']
    feat_names: List[str] = bundle['features']

    feats = extract_features_from_window(df_window, fs)
    X = np.array([[feats.get(f, 0.0) for f in feat_names]], dtype=float)
    Xs = scaler.transform(X)
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(Xs)[0]
        idx = int(np.argmax(proba))
        label = str(clf.classes_[idx])
        conf = float(np.max(proba))
    else:
        ypred = clf.predict(Xs)[0]
        label = str(ypred)
        conf = 1.0
    return label, conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train light MLP on EMG CSVs')
    parser.add_argument('--csv', nargs='+', required=True, help='CSV file paths')
    parser.add_argument('--out', required=True, help='Output model path (joblib)')
    parser.add_argument('--fs', type=float, default=1000.0)
    parser.add_argument('--window_ms', type=int, default=200)
    parser.add_argument('--hop_ms', type=int, default=100)
    parser.add_argument('--mains_freq', type=float, default=50.0)
    args = parser.parse_args()

    cfg = TrainConfig(fs=args.fs, window_ms=args.window_ms, hop_ms=args.hop_ms, mains_freq=args.mains_freq)
    metrics = train_mlp(args.csv, cfg, args.out)
    print({k: (v if k != 'report' else '\n' + v) for k, v in metrics.items()})
