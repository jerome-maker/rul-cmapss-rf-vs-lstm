# 以機器學習法預測航空引擎剩餘壽命—比較隨機森林與長短期記憶網路

TAAI 2025 論文的資料處理、模型訓練與圖表產生程式。

---

## 一、論文與研討會資料

| 項目 | 內容 |
|---|---|
| 論文題目 | 以機器學習法預測航空引擎剩餘壽命—比較隨機森林與長短期記憶網路 |
| 作者 | 周理陽 |
| 單位 | 國立中央大學機械工程學系 |
| E-mail | 112303573@cc.ncu.edu.tw |
| 稿號 | DD-0574 |
| 投稿類別 | 國內議程（Domestic Track），延伸摘要投稿，以海報形式發表 |
| 關鍵字 | 剩餘壽命預測、隨機森林、LSTM、C-MAPSS、預測性維護 |

### 研討會

| 項目 | 內容 |
|---|---|
| 全名 | 2025 年人工智慧與應用研討會（第 30 屆）<br>The 30th International Conference on Technologies and Applications of Artificial Intelligence |
| 簡稱 | TAAI 2025 |
| 主辦 | 台灣人工智慧學會（TAAI） |
| 合辦 | 中央研究院、國立臺灣師範大學 |
| 日期 | 2025 年 12 月 13 日至 14 日 |
| 地點 | 國立臺灣師範大學 和平校區二（臺北市） |
| 主題 | AI 驅動的跨域科學創新 |
| 議程結構 | 國際議程、國內議程、業界議程，另含專題研討會與競賽 |
| 官方網站 | https://taai2025.org |
| 審查 | 每篇投稿至少三名審查員 |
| 重要日期 | 投稿截止 2025-09-27（兩次延長後）／接受通知 2025-10-13／定稿 2025-11-08 |
| 出席證明 | `(taai2025)DD-0574.pdf`（Certification of Attendance） |

徵稿領域涵蓋機器學習、自然語言處理、電腦視覺、資料探勘、演化計算、
多代理人系統、機器人學、精準醫療、智慧製造等。本研究屬機器學習與智慧製造領域。

### 論文摘要

隨著工業 4.0 推動，智慧設備管理成為提升生產效率與保障操作安全的關鍵技術。在航空
領域，飛機引擎的剩餘壽命（Remaining Useful Life, RUL）預測能評估零件在故障前的安全
運行時間，為預測性維護決策提供依據，藉以合理安排維修時機、降低非計劃性停機。

早期 RUL 預測多依賴傳統機器學習方法，如隨機森林（Random Forest, RF）、支持向量機
（SVM）與多層感知器（MLP）；隨機森林在樣本數較少時展現穩定優勢，能有效篩選關鍵
特徵，但隨感測器資料維度提升，傳統方法在捕捉非線性關係與長期依賴性上存在限制。
近年長短期記憶網路（Long Short-Term Memory, LSTM）因其時間序列建模能力而成為主流。
現有文獻多集中於各自模型的最佳化，缺乏系統性比較。

本研究以 NASA C-MAPSS 資料集的 FD001 子集為實驗平台，透過一致的資料處理流程與超
參數最佳化，系統性分析隨機森林與 LSTM 在 RUL 預測任務中的適用性與效能差異。資料
前處理包含原始感測器資料清理與特徵工程，並以隨機森林進行特徵重要性評估，篩選對引擎
壽命影響最大的關鍵參數；LSTM 則以 Optuna 套件調整隱藏層神經元數量、滑動視窗大小、
學習率與 Dropout 比例。兩模型均以均方根誤差（RMSE）與平均絕對誤差（MAE）評估。

結果顯示 LSTM 在兩項指標上均優於隨機森林，說明深度學習方法在處理 C-MAPSS 資料所
呈現的非線性與長期依賴特徵時具備更強的特徵抽取與學習能力；但 LSTM 的誤差四分位距
較寬，對超參數設定較為敏感，隨機森林則誤差分佈集中、效能較穩定。若應用情境以高精度
為首要考量，建議採用 LSTM；若以穩定性或可解釋性為優先，隨機森林具備更一致的表現
與較低的計算成本，適合資源受限環境。

> 原文為延伸摘要格式（2 頁 A4），未設獨立摘要欄位，上文依論文各節內容整理。

---

## 二、資料

NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set 的 **FD001** 子集，
模擬飛機渦扇引擎在多種運行環境與負載條件下的劣化過程。

| 檔案 | 內容 | 規模 |
|---|---|---|
| `data/train_FD001.txt` | 100 台引擎，每台由正常運行至故障 | 20,631 列 × 26 欄 |
| `data/test_FD001.txt` | 100 台引擎，時序在故障前被截斷 | 13,096 列 × 26 欄 |
| `data/RUL_FD001.txt` | 測試集 100 台引擎的真實剩餘壽命 | 100 列 |

來源與欄位說明見 [`data/README_data.md`](data/README_data.md)。

---

## 三、方法

### 資料處理（`src/cmapss.py`，兩個模型共用）

1. **感測器清理**：依變異數剔除全程為常數的感測器。FD001 剔除 `s1, s5, s10, s16,
   s18, s19`，保留 15 個。
2. **RUL 標註**：訓練集每台引擎跑到故障，故 `RUL(t) = 最大 cycle − t`，再套用分段
   線性截斷 `RUL = min(RUL, 125)`。截斷後約 39.4% 的樣本落在上限。
3. **正規化**：最小最大正規化，僅以訓練集配適，避免測試集資訊外洩。
4. **特徵工程**：每個感測器加上滾動平均、滾動標準差與滾動斜率（視窗 5），
   三者皆只使用過去的觀測值。

兩個模型共用同一套前處理，效能差異才能歸因於模型本身。

### 隨機森林（`src/rf_baseline.py`）

先以完整特徵集訓練一棵 300 棵樹的森林取得特徵重要性，篩出前 20 名特徵；
再掃過決策樹數量（200/500/1000）、最大樹深（不限/12/20）、子集特徵數目
（`sqrt`/0.5）的網格，共 12 組設定。

### LSTM（`src/lstm_optuna.py`）

以滑動視窗切出序列，取最後一個時步接全連接層輸出 RUL。訓練集依引擎切出 20% 作為
驗證集，用於早停與 Optuna 的搜尋目標；測試集全程不參與搜尋。

Optuna（TPE sampler）搜尋 30 組設定，搜尋空間：

| 超參數 | 範圍 |
|---|---|
| 隱藏層神經元數量 | 32 / 64 / 96 / 128 |
| 層數 | 1 / 2 |
| 滑動視窗大小 | 20 / 25 / 30 / 35 / 40 |
| Dropout 比例 | 0.1 – 0.5 |
| 學習率 | 1e-4 – 1e-2（對數尺度） |
| 批次大小 | 128 / 256 / 512 |

### 評估

測試集每台引擎取最後一個時間點預測，對上 `RUL_FD001.txt` 的 100 個真值，
計算 RMSE 與 MAE。

---

## 四、執行結果

測試集表現（LSTM 取驗證集 RMSE 最佳的前 8 組設定，RF 為網格全部 12 組）：

| 模型 | RMSE 最小 | RMSE 中位數 | RMSE 最大 | MAE 最小 | MAE 中位數 | MAE 最大 |
|---|---|---|---|---|---|---|
| LSTM | 14.18 | 14.39 | 14.70 | 10.11 | 10.46 | 10.89 |
| RF | 18.53 | 18.92 | 19.28 | 13.34 | 13.71 | 14.04 |

**最佳設定**

- LSTM：隱藏層 32、2 層、視窗 40、Dropout 0.24、學習率 1.51e-3、批次 256
  → RMSE 14.50、MAE 10.40（30 組搜尋中測試集最低為 RMSE 13.99、MAE 10.11）
- RF：500 棵樹、最大樹深 12、`max_features="sqrt"` → RMSE 18.53、MAE 13.34

圖表：[`figures/fig1_mae_boxplot.png`](figures/fig1_mae_boxplot.png)（測試集 MAE 比較）、
[`figures/fig2_rmse_boxplot.png`](figures/fig2_rmse_boxplot.png)（測試集 RMSE 比較）。
逐組明細見 `results/rf_runs.csv`、`results/lstm_trials.csv`，特徵重要性排序見
`results/rf_feature_importance.csv`。

**執行環境與耗時**

| 階段 | 裝置 | 環境 | 耗時 | 峰值 VRAM |
|---|---|---|---|---|
| 隨機森林（12 組） | CPU | `base` | 520.6 秒 | — |
| LSTM（30 組 Optuna trial） | GPU（GTX 1060 3GB） | `Colab2025` | 518.7 秒 | 534.2 MiB |

隨機森林用 CPU：scikit-learn 的隨機森林沒有 GPU 後端，且在約 2 萬列 × 45 個特徵
的規模下，搬進搬出 GPU 的成本高於計算本身。

---

## 五、檔案結構

```
RUL/
├─ data/                      FD001 原始資料與資料說明
├─ src/
│  ├─ cmapss.py               資料載入、RUL 標註、感測器篩選、特徵工程、視窗化
│  ├─ check_data.py           資料規模與欄位檢查
│  ├─ rf_baseline.py          隨機森林：特徵重要性篩選 + 參數網格
│  ├─ lstm_optuna.py          LSTM + Optuna 超參數搜尋
│  └─ make_figures.py         箱型圖與彙總表
├─ results/                   逐組 RMSE / MAE 明細、特徵重要性、執行紀錄
├─ figures/                   圖 1、圖 2
└─ references/                參考文獻全文與書目（見 references/README.md）
```

---

## 六、如何執行

### 0. 檢查資料

```bash
C:\Users\test\anaconda3\python.exe src\check_data.py
```

### 1. 隨機森林（CPU）

```bash
C:\Users\test\anaconda3\python.exe src\rf_baseline.py
```

用 `base` 環境：需要 scikit-learn，且本檔不 import torch，不會踩到 `base` 的
numpy（MKL 建置）與 torch 自帶 OpenMP 的衝突。

### 2. LSTM + Optuna（GPU）

```bash
C:\Users\test\anaconda3\envs\Colab2025\python.exe src\lstm_optuna.py --trials 30
```

用 `Colab2025`：numpy 與 torch 併用時的指定環境。該環境已安裝 optuna 5.0.0。
若換到沒有 optuna 的環境，程式會自動退回內建亂數搜尋，搜尋空間相同。

模型參數量小（峰值 VRAM 約 534 MiB），未使用 AMP：這張卡沒有 Tensor Core，
AMP 的主要價值是省記憶體，而此處記憶體並非瓶頸，fp32 也避免 RUL 迴歸的數值誤差。

### 3. 產生圖表

```bash
C:\Users\test\anaconda3\python.exe src\make_figures.py
```

---

## 七、參考文獻

正文引用四篇，另附 C-MAPSS 資料集的原始說明文件；可公開取得的全文已下載於
[`references/`](references/README.md)。

1. Wang, H. et al. (2023). *Remaining Useful Life Prediction of Aircraft Turbofan Engine
   Based on Random Forest Feature Selection and Multi-Layer Perceptron.*
   Applied Sciences, 13(12), 7186.
2. Ensarioğlu, K.; İnkaya, T.; Emel, E. (2023). *Remaining Useful Life Estimation of
   Turbofan Engines with Deep Learning Using Change-Point Detection Based Labeling and
   Feature Engineering.* Applied Sciences, 13(21), 11893.
3. Kundu, R. K.; Hoque, K. A. (2023). *Explainable Predictive Maintenance is Not Enough:
   Quantifying Trust in Remaining Useful Life Estimation.*
   Annual Conference of the PHM Society, 15(1).
4. Mothilall, D.; van Zyl, T. L. (2024). *An evaluation of the Long Short-Term Memory
   model for predictive maintenance applications in the aircraft industry.* ACDSA 2024.
5. Saxena, A.; Goebel, K.; Simon, D.; Eklund, N. (2008). *Damage Propagation Modeling for
   Aircraft Engine Run-to-Failure Simulation.* PHM 2008.

---

## 八、授權

本專案自行撰寫的程式碼（`src/`）與文件以 MIT License 釋出，詳見
[`LICENSE`](LICENSE)。歡迎重製、修改與再散布，惟須保留著作權聲明。

以下內容不在本授權範圍內，各自沿用原本的條款：

- **C-MAPSS 資料集**（`data/`）：由 NASA Prognostics Center of Excellence 發布，
  屬美國政府作品，可自由使用；引用時請標註上列第 5 篇原始說明文件。
- **參考文獻**（`references/`）：著作權歸各論文作者與出版方所有，
  重製與再散布須依各篇授權條款（多數為 CC BY 4.0，詳見
  [`references/README.md`](references/README.md)）。
