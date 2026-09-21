# -*- coding: utf-8 -*-
"""繪製測試集 MAE 與 RMSE 的模型比較箱型圖。

兩張圖都是左 LSTM、右 RF 的箱型圖，座標軸與刻度標籤用英文。

箱子裡的樣本：
  LSTM —— 依驗證集 RMSE 排序後的前 N 組超參數設定（預設 N=8）。
  RF   —— 參數網格的全部 12 組設定。
取「調校後的前 N 名」而非整個搜尋過程，是為了讓兩邊比較的都是經過
調校的模型；N 由 --top-n 控制。

執行環境：base 或 Colab2025 皆可（只用 pandas + matplotlib）。

    C:\\Users\\test\\anaconda3\\python.exe src\\make_figures.py
"""
from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

import cmapss as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#4C72B0"  # 論文圖中的藍，即 seaborn 預設色盤第一色


def boxplot(metric: str, lstm_vals, rf_vals, out_path: str):
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=200)
    bp = ax.boxplot([lstm_vals, rf_vals], patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor="0.35",
                                    markeredgecolor="0.35"))
    for box in bp["boxes"]:
        box.set(facecolor=BLUE, edgecolor="black", linewidth=0.9)
    ax.set_xticklabels([f"LSTM {metric}", f"RF {metric}"])
    ax.set_ylabel(metric)
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.out_path(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"已輸出 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", default="results/lstm_trials.csv")
    ap.add_argument("--rf", default="results/rf_runs.csv")
    ap.add_argument("--top-n", type=int, default=8,
                    help="LSTM 取驗證集 RMSE 最好的前幾個 trial")
    args = ap.parse_args()

    lstm = pd.read_csv(C.path(args.lstm)).sort_values("val_rmse").head(args.top_n)
    rf = pd.read_csv(C.path(args.rf))

    boxplot("MAE", lstm["mae"], rf["mae"], "figures/fig1_mae_boxplot.png")
    boxplot("RMSE", lstm["rmse"], rf["rmse"], "figures/fig2_rmse_boxplot.png")

    summary = pd.DataFrame({
        "模型": ["LSTM", "RF"],
        "RMSE 最小": [lstm.rmse.min(), rf.rmse.min()],
        "RMSE 中位數": [lstm.rmse.median(), rf.rmse.median()],
        "RMSE 最大": [lstm.rmse.max(), rf.rmse.max()],
        "MAE 最小": [lstm.mae.min(), rf.mae.min()],
        "MAE 中位數": [lstm.mae.median(), rf.mae.median()],
        "MAE 最大": [lstm.mae.max(), rf.mae.max()],
    }).round(2)
    summary.to_csv(C.out_path("results/summary.csv"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
