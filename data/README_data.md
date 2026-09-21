# 資料：NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set

本研究使用 **FD001** 子集，模擬飛機渦扇引擎在單一運行條件下的劣化過程。

| 檔案 | 內容 | 規模 |
|---|---|---|
| `train_FD001.txt` | 100 台引擎，每台由正常運行至故障 | 20,631 列 × 26 欄 |
| `test_FD001.txt` | 100 台引擎，時序在故障前被截斷 | 13,096 列 × 26 欄 |
| `RUL_FD001.txt` | 測試集 100 台引擎的真實剩餘壽命 | 100 列 × 1 欄 |
| `NASA_readme.txt` | NASA 隨資料集釋出的原始說明 | — |

執行 `src/check_data.py` 可核對以上規模；數字對不上代表拿錯子集或檔案不完整。

## 欄位格式

空白分隔、無表頭，欄序固定：

```
unit  cycle  opset_1  opset_2  opset_3  s1  s2  ...  s21
```

- `unit`：引擎編號（FD001 為 1–100）
- `cycle`：該台引擎的第幾個運行週期
- `opset_1..3`：運行條件。FD001 只有單一運行條件，這三欄變異極小
- `s1..s21`：21 個感測器讀值

FD001 中 `s1, s5, s10, s16, s18, s19` 全程為常數，模型用不上，
`src/cmapss.py::select_sensors` 會依變異數自動剔除，留下 15 個感測器欄位。

訓練集每台引擎壽命介於 128 至 362 個 cycle；測試集每台引擎長度介於 31 至 303 個
cycle，最短的 31 決定了 LSTM 滑動視窗長度的實務上限。

## 標籤

訓練集沒有 RUL 欄位——每台引擎都跑到故障為止，因此

```
RUL(t) = 該台引擎的最大 cycle − t
```

再套用分段線性截斷 `RUL = min(RUL, 125)`。引擎在劣化開始前的感測器訊號幾乎看不出
差別，不截斷等於逼模型去配適一段本來就無從預測的區間。截斷後約 39.4% 的訓練樣本
落在上限。

測試集的真值 RUL 介於 7 至 145，平均 75.5，直接由 `RUL_FD001.txt` 提供，
對應每台引擎時序被截斷的那個時間點。

## 來源

NASA Prognostics Center of Excellence 釋出，目前掛在 NASA 開放資料入口：

- 資料集頁面：https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- 壓縮檔：https://data.nasa.gov/docs/legacy/CMAPSSData.zip

壓縮檔內含 FD001–FD004 四個子集與資料集說明文件
（`Damage Propagation Modeling.pdf`，已另存於 `references/`）。
本資料夾只保留 FD001 的三個檔案。

資料為 NASA 公開釋出，可自由用於研究；引用時註明 Saxena, Goebel, Simon & Eklund
(2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*,
PHM 2008。
