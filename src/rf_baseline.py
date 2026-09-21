# -*- coding: utf-8 -*-
r"""隨機森林模型 + 特徵重要性篩選。

先以特徵工程產生的完整特徵集訓練一棵中等規模的森林，取其特徵重要性
篩出最關鍵的參數；再掃過決策樹數量、最大樹深、子集特徵數目三個關鍵
參數的多組設定，每組記錄一次測試集 RMSE / MAE，構成箱型圖裡 RF 的分佈。

執行環境：base（需要 scikit-learn；本檔不 import torch，
沒有 CLAUDE.md 記載的 numpy(MKL) 與 torch OpenMP 衝突問題）。

    C:\Users\test\anaconda3\python.exe src\rf_baseline.py

裝置：CPU。sklearn 的隨機森林沒有 GPU 後端，且在這個資料量
（約 2 萬列、45 個特徵）下，搬進搬出 GPU 的成本高於計算本身。
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import cmapss as C


def build_tabular(data_dir: str, roll_window: int = 5, top_k: int = 20,
                  seed: int = 42):
    """回傳 RF 用的訓練矩陣、測試矩陣、真值，以及被選中的特徵名稱。"""
    train, test, rul_true = C.load_fd001(data_dir)
    train = C.add_train_rul(train)

    sensors = C.select_sensors(train)
    lo, rng = C.fit_minmax(train, sensors)
    train = C.apply_minmax(train, sensors, lo, rng)
    test = C.apply_minmax(test, sensors, lo, rng)

    train, feat_all = C.add_rolling_features(train, sensors, roll_window)
    test, _ = C.add_rolling_features(test, sensors, roll_window)

    X_tr = train[feat_all].to_numpy(np.float32)
    y_tr = train["RUL"].to_numpy(np.float32)

    # ---- 特徵重要性篩選：先用一棵中等大小的森林評估重要性，取前 top_k ----
    probe = RandomForestRegressor(n_estimators=300, max_depth=None,
                                  n_jobs=-1, random_state=seed)
    probe.fit(X_tr, y_tr)
    order = np.argsort(probe.feature_importances_)[::-1]
    keep = [feat_all[i] for i in order[:top_k]]
    imp = pd.DataFrame({"feature": [feat_all[i] for i in order],
                        "importance": probe.feature_importances_[order]})

    test_last = C.last_cycle_rows(test)
    return (train[keep].to_numpy(np.float32), y_tr,
            test_last[keep].to_numpy(np.float32), rul_true, keep, imp, sensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--roll-window", type=int, default=5)
    ap.add_argument("--out", default="results/rf_runs.csv")
    args = ap.parse_args()

    t0 = time.time()
    X_tr, y_tr, X_te, y_te, keep, imp, sensors = build_tabular(
        args.data_dir, args.roll_window, args.top_k)
    print(f"保留感測器 {len(sensors)} 個：{sensors}")
    print(f"特徵重要性前 {args.top_k} 名：{keep}")
    imp.to_csv(C.out_path("results/rf_feature_importance.csv"), index=False,
               encoding="utf-8-sig")

    # 三個關鍵參數的網格；每個組合算一次，共 12 組
    grid = list(itertools.product(
        [200, 500, 1000],          # 決策樹數量
        [None, 12, 20],            # 最大樹深
        ["sqrt", 0.5],             # 子集特徵數目
    ))[:12]

    rows = []
    for n_est, depth, mf in grid:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                   max_features=mf, n_jobs=-1,
                                   random_state=42)
        rf.fit(X_tr, y_tr)
        pred = np.clip(rf.predict(X_te), 0, None)
        r, m = C.rmse(y_te, pred), C.mae(y_te, pred)
        rows.append({"model": "RF", "n_estimators": n_est,
                     "max_depth": -1 if depth is None else depth,
                     "max_features": str(mf), "rmse": r, "mae": m})
        print(f"  n={n_est:<5} depth={str(depth):<5} mf={str(mf):<5} "
              f"RMSE={r:6.3f}  MAE={m:6.3f}")

    df = pd.DataFrame(rows).sort_values("rmse")
    df.to_csv(C.out_path(args.out), index=False, encoding="utf-8-sig")
    best = df.iloc[0]
    print(json.dumps({"best_rmse": best.rmse, "best_mae": best.mae,
                      "median_rmse": float(df.rmse.median()),
                      "median_mae": float(df.mae.median()),
                      "device": "CPU", "seconds": round(time.time() - t0, 1)},
                     indent=2))


if __name__ == "__main__":
    main()
