# -*- coding: utf-8 -*-
"""C-MAPSS FD001 資料處理共用模組。

處理流程：原始感測器資料清理 -> 特徵工程 -> 隨機森林特徵重要性篩選 -> RF / LSTM 建模。

本模組只放兩個模型共用的部分，確保 RF 與 LSTM 走一致的資料處理流程，
兩者的效能差異才歸因於模型本身。

本檔案不使用 torch，可在 base 或 Colab2025 任一環境執行。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 相對路徑一律以專案根目錄（本檔的上一層）為基準，
# 這樣不論在 RUL/ 還是 RUL/src/ 下執行都指向同一個位置。
ROOT = Path(__file__).resolve().parent.parent


def path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def out_path(p) -> Path:
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# C-MAPSS 原始檔為空白分隔、無表頭，共 26 欄
INDEX_COLS = ["unit", "cycle"]
OPSET_COLS = [f"opset_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OPSET_COLS + SENSOR_COLS

# 分段線性 RUL 上限。引擎在劣化開始前的訊號幾乎沒有差別，
# 不截斷等於逼模型去配適一段本來就無從預測的區間。
RUL_CAP = 125


def load_fd001(data_dir: str = "data"):
    """讀入 FD001 的三個檔案，回傳 (train, test, rul_true)。"""
    d = path(data_dir)

    def _read(name):
        df = pd.read_csv(d / name, sep=r"\s+", header=None, engine="python")
        df = df.dropna(axis=1, how="all")
        df.columns = ALL_COLS[:df.shape[1]]
        return df

    train = _read("train_FD001.txt")
    test = _read("test_FD001.txt")
    rul_true = pd.read_csv(d / "RUL_FD001.txt", sep=r"\s+", header=None,
                           engine="python").iloc[:, 0].to_numpy()
    return train, test, rul_true


def add_train_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """訓練集每台引擎跑到故障，故 RUL = 該台最大 cycle - 當前 cycle，再截斷於 cap。"""
    df = train.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (last - df["cycle"]).clip(upper=cap)
    return df


def select_sensors(train: pd.DataFrame, tol: float = 1e-6) -> list[str]:
    """清理步驟一：剔除全程為常數（變異數近乎 0）的感測器。

    FD001 會被剔除的是 s1, s5, s10, s16, s18, s19，留下 15 個欄位。
    這一步是資料本身決定的，不含任何調校空間。
    """
    std = train[SENSOR_COLS].std(numeric_only=True)
    return [c for c in SENSOR_COLS if std[c] > tol]


def fit_minmax(train: pd.DataFrame, cols: list[str]):
    """只用訓練集配適最小最大正規化參數，避免測試集資訊外洩。"""
    lo = train[cols].min()
    hi = train[cols].max()
    rng = (hi - lo).replace(0, 1.0)
    return lo, rng


def apply_minmax(df: pd.DataFrame, cols: list[str], lo, rng) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (out[cols] - lo) / rng
    return out


def add_rolling_features(df: pd.DataFrame, cols: list[str], window: int = 5):
    """特徵工程：對每個感測器加上滾動平均、滾動標準差與線性斜率。

    這三種摘要統計量都只看過去的觀測值，不會用到未來資訊。
    """
    out = df.copy()
    new_cols = []
    g = out.groupby("unit")
    for c in cols:
        out[f"{c}_ma"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=1).mean())
        out[f"{c}_sd"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0.0))
        out[f"{c}_sl"] = g[c].transform(
            lambda s: s.diff().rolling(window, min_periods=1).mean().fillna(0.0))
        new_cols += [f"{c}_ma", f"{c}_sd", f"{c}_sl"]
    return out, cols + new_cols


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """取每台引擎的最後一筆紀錄——測試集的評估點。"""
    idx = df.groupby("unit")["cycle"].transform("max") == df["cycle"]
    return df[idx].sort_values("unit").reset_index(drop=True)


def make_windows(df: pd.DataFrame, cols: list[str], window: int,
                 label: str | None = "RUL"):
    """把逐筆時序資料切成 LSTM 用的滑動視窗。

    回傳 X 形狀 (樣本數, window, 特徵數)。長度不足 window 的引擎以首筆資料
    向前補齊（FD001 測試集最短 31 個 cycle，window <= 30 時實際不會觸發）。
    """
    xs, ys = [], []
    for _, g in df.groupby("unit", sort=True):
        arr = g[cols].to_numpy(dtype=np.float32)
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.vstack([pad, arr])
            if label is not None:
                lab = np.concatenate([np.repeat(g[label].to_numpy()[:1],
                                                window - len(g)),
                                      g[label].to_numpy()])
            else:
                lab = None
        else:
            lab = g[label].to_numpy() if label is not None else None
        for i in range(window, len(arr) + 1):
            xs.append(arr[i - window:i])
            if lab is not None:
                ys.append(lab[i - 1])
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32) if label is not None else None
    return X, y


def make_test_windows(test: pd.DataFrame, cols: list[str], window: int):
    """測試集每台引擎只取最後一個視窗，對應 RUL_FD001.txt 的一個真值。"""
    xs = []
    for _, g in test.groupby("unit", sort=True):
        arr = g[cols].to_numpy(dtype=np.float32)
        if len(arr) < window:
            arr = np.vstack([np.repeat(arr[:1], window - len(arr), axis=0), arr])
        xs.append(arr[-window:])
    return np.asarray(xs, dtype=np.float32)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
