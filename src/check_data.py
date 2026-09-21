# -*- coding: utf-8 -*-
"""資料落地後的第一道檢查：規模、欄位、常數感測器、RUL 分佈。

    C:\\Users\\test\\anaconda3\\python.exe src\\check_data.py
"""
from __future__ import annotations

import sys

import cmapss as C

EXPECT = {"train_rows": 20631, "test_rows": 13096, "units": 100}


def main(data_dir: str = "data") -> int:
    try:
        train, test, rul = C.load_fd001(data_dir)
    except FileNotFoundError as e:
        print(f"找不到資料檔：{e}\n請先依 data/README_data.md 放入 FD001 三個檔案。")
        return 1

    ok = True
    checks = [
        ("訓練集列數", len(train), EXPECT["train_rows"]),
        ("測試集列數", len(test), EXPECT["test_rows"]),
        ("訓練集引擎數", train.unit.nunique(), EXPECT["units"]),
        ("測試集引擎數", test.unit.nunique(), EXPECT["units"]),
        ("RUL 真值筆數", len(rul), EXPECT["units"]),
        ("欄位數", train.shape[1], 26),
    ]
    for name, got, want in checks:
        flag = "OK " if got == want else "不符"
        ok &= got == want
        print(f"  [{flag}] {name}: {got}（預期 {want}）")

    kept = C.select_sensors(train)
    dropped = [c for c in C.SENSOR_COLS if c not in kept]
    print(f"\n  常數感測器（剔除）：{dropped}")
    print(f"  保留感測器 {len(kept)} 個：{kept}")

    t = C.add_train_rul(train)
    print(f"\n  訓練集每台引擎壽命：最短 {train.groupby('unit').cycle.max().min()}、"
          f"最長 {train.groupby('unit').cycle.max().max()} 個 cycle")
    print(f"  測試集每台引擎長度：最短 {test.groupby('unit').cycle.max().min()}、"
          f"最長 {test.groupby('unit').cycle.max().max()} 個 cycle"
          "（決定 LSTM 視窗長度的上限）")
    print(f"  截斷後 RUL 分佈：{t.RUL.min():.0f} ~ {t.RUL.max():.0f}，"
          f"落在上限 {C.RUL_CAP} 的比例 {(t.RUL == C.RUL_CAP).mean():.1%}")
    print(f"  測試集真值 RUL：{rul.min()} ~ {rul.max()}，平均 {rul.mean():.1f}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data"))
