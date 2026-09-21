# -*- coding: utf-8 -*-
"""LSTM + Optuna 超參數搜尋。

以 Optuna 調整隱藏層神經元數量、滑動視窗大小、學習率與 Dropout 比例
（另含層數與批次大小）。每個 trial 是一組超參數，訓練完成後在 FD001
測試集上算 RMSE / MAE，所有 trial 的結果構成箱型圖裡 LSTM 的分佈。

執行環境：Colab2025（numpy + torch 併用，依 CLAUDE.md 規則不用 base）。

    C:\\Users\\test\\anaconda3\\envs\\Colab2025\\python.exe src\\lstm_optuna.py --trials 30

裝置：GPU（GTX 1060 3GB）。這個模型很小（約 0.1M 參數、峰值 VRAM 不到 300 MiB），
用不到 AMP 省記憶體那條路，故維持 fp32，也避免 RUL 迴歸的數值誤差。
若 Colab2025 沒裝 optuna，本檔會自動退回內建的亂數搜尋（--sampler random），
搜尋空間完全相同，只是少了 TPE 的引導。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import cmapss as C

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        raise RuntimeError(
            "預期要用 GPU 卻取到 CPU — 先檢查 torch 是不是 +cpu 建置："
            f"目前 torch={torch.__version__}")
    return dev


class LSTMRegressor(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


class Data:
    """把資料前處理做一次就好；只有視窗長度會隨 trial 改變。"""

    def __init__(self, data_dir: str, val_frac: float = 0.2, seed: int = 42):
        train, test, rul_true = C.load_fd001(data_dir)
        train = C.add_train_rul(train)
        self.sensors = C.select_sensors(train)
        lo, rng = C.fit_minmax(train, self.sensors)
        self.train = C.apply_minmax(train, self.sensors, lo, rng)
        self.test = C.apply_minmax(test, self.sensors, lo, rng)
        self.rul_true = rul_true

        # 依引擎切分驗證集，不打散時序、也不讓同一台引擎橫跨兩邊
        units = np.array(sorted(self.train.unit.unique()))
        rs = np.random.RandomState(seed)
        rs.shuffle(units)
        n_val = int(len(units) * val_frac)
        self.val_units = set(units[:n_val])
        self.tr_units = set(units[n_val:])

    def windows(self, window: int):
        tr = self.train[self.train.unit.isin(self.tr_units)]
        va = self.train[self.train.unit.isin(self.val_units)]
        Xtr, ytr = C.make_windows(tr, self.sensors, window)
        Xva, yva = C.make_windows(va, self.sensors, window)
        Xte = C.make_test_windows(self.test, self.sensors, window)
        return Xtr, ytr, Xva, yva, Xte


def train_one(data: Data, params: dict, device, max_epochs: int = 60,
              patience: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    w = params["window"]
    Xtr, ytr, Xva, yva, Xte = data.windows(w)

    to = lambda a: torch.as_tensor(a, device=device)
    Xtr_t, ytr_t = to(Xtr), to(ytr / C.RUL_CAP)
    Xva_t, yva_t = to(Xva), to(yva / C.RUL_CAP)
    Xte_t = to(Xte)

    model = LSTMRegressor(Xtr.shape[2], params["hidden"], params["layers"],
                          params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    lossf = nn.MSELoss()
    bs = params["batch"]
    n = len(Xtr_t)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.inference_mode():
            vp = torch.cat([model(Xva_t[i:i + 1024])
                            for i in range(0, len(Xva_t), 1024)])
            val = float(torch.sqrt(lossf(vp, yva_t))) * C.RUL_CAP
        if val < best_val - 1e-4:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred = model(Xte_t).cpu().numpy() * C.RUL_CAP
    pred = np.clip(pred, 0, None)
    return C.rmse(data.rul_true, pred), C.mae(data.rul_true, pred), best_val


# 搜尋空間
SPACE = {
    "hidden":  [32, 64, 96, 128],
    "layers":  [1, 2],
    "window":  [20, 25, 30, 35, 40],
    "dropout": (0.1, 0.5),
    "lr":      (1e-4, 1e-2),
    "batch":   [128, 256, 512],
}


def sample_random(rs: np.random.RandomState) -> dict:
    return {
        "hidden": int(rs.choice(SPACE["hidden"])),
        "layers": int(rs.choice(SPACE["layers"])),
        "window": int(rs.choice(SPACE["window"])),
        "dropout": float(rs.uniform(*SPACE["dropout"])),
        "lr": float(10 ** rs.uniform(np.log10(SPACE["lr"][0]),
                                     np.log10(SPACE["lr"][1]))),
        "batch": int(rs.choice(SPACE["batch"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sampler", choices=["optuna", "random"], default="optuna")
    ap.add_argument("--out", default="results/lstm_trials.csv")
    args = ap.parse_args()

    device = get_device()
    torch.cuda.reset_peak_memory_stats()
    print(f"裝置：{torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    data = Data(args.data_dir)
    rows = []
    t0 = time.time()

    def evaluate(params, seed):
        r, m, v = train_one(data, params, device, max_epochs=args.epochs,
                            seed=seed)
        rows.append({"model": "LSTM", **params, "val_rmse": v,
                     "rmse": r, "mae": m})
        print("  trial {:>2}  h={:<4} L={} w={:<3} do={:.2f} lr={:.2e}"
              "  -> RMSE={:6.3f} MAE={:6.3f}".format(
                  len(rows), params["hidden"], params["layers"],
                  params["window"], params["dropout"], params["lr"], r, m))
        return v  # 以驗證集 RMSE 當搜尋目標，測試集不參與搜尋

    use_optuna = args.sampler == "optuna" and HAS_OPTUNA
    if args.sampler == "optuna" and not HAS_OPTUNA:
        print("[注意] 本環境未安裝 optuna，改用內建亂數搜尋（搜尋空間相同）")

    if use_optuna:
        def objective(trial):
            params = {
                "hidden": trial.suggest_categorical("hidden", SPACE["hidden"]),
                "layers": trial.suggest_categorical("layers", SPACE["layers"]),
                "window": trial.suggest_categorical("window", SPACE["window"]),
                "dropout": trial.suggest_float("dropout", *SPACE["dropout"]),
                "lr": trial.suggest_float("lr", *SPACE["lr"], log=True),
                "batch": trial.suggest_categorical("batch", SPACE["batch"]),
            }
            return evaluate(params, seed=trial.number)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.trials)
    else:
        rs = np.random.RandomState(42)
        for i in range(args.trials):
            evaluate(sample_random(rs), seed=i)

    df = pd.DataFrame(rows).sort_values("val_rmse")
    df.to_csv(C.out_path(args.out), index=False, encoding="utf-8-sig")
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    best = df.iloc[0]
    print(json.dumps({"best_by_val_rmse": {"rmse": best.rmse, "mae": best.mae},
                      "min_rmse": float(df.rmse.min()),
                      "min_mae": float(df.mae.min()),
                      "device": torch.cuda.get_device_name(0),
                      "peak_vram_MiB": round(peak, 1),
                      "seconds": round(time.time() - t0, 1)}, indent=2))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
