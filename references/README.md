# 參考文獻

論文正文引用的四篇文獻，加上 C-MAPSS 資料集本身的原始說明文件。
可自由取得的全文已下載於本資料夾。

| 編號 | 檔案 | 取得狀態 | 授權 |
|---|---|---|---|
| [1] | `ref1_Wang2023_RF-MLP.pdf` | 已下載（10 頁） | CC BY 4.0 |
| [2] | `ref2_Ensarioglu2023_CPD-LSTM.pdf` | 已下載（10 頁） | CC BY 4.0 |
| [3] | `ref3_Kundu2023_TrustRUL.pdf` | 已下載 | PHM Society 開放取用 |
| [4] | — | **IEEE 需訂閱，無開放取用版本** | © IEEE |
| [5] | `ref5_Saxena2008_DamagePropagationModeling.pdf` | 已下載（隨資料集釋出） | NASA 公開釋出 |

---

## [1] Wang et al. (2023) — 隨機森林特徵篩選 + 多層感知器

Wang, H.; Li, D.; Li, D.; Liu, C.; Yang, X.; Zhu, G.
"Remaining Useful Life Prediction of Aircraft Turbofan Engine Based on Random Forest
Feature Selection and Multi-Layer Perceptron."
*Applied Sciences*, 2023, 13(12), 7186.
DOI: [10.3390/app13127186](https://doi.org/10.3390/app13127186)

在本研究中的作用：隨機森林作為特徵重要性評估工具的依據，以及隨機森林在高維度特徵、
樣本數有限情境下抗噪聲能力的論據來源。

## [2] Ensarioğlu et al. (2023) — 變點偵測標註與特徵工程

Ensarioğlu, K.; İnkaya, T.; Emel, E.
"Remaining Useful Life Estimation of Turbofan Engines with Deep Learning Using
Change-Point Detection Based Labeling and Feature Engineering."
*Applied Sciences*, 2023, 13(21), 11893.
DOI: [10.3390/app132111893](https://doi.org/10.3390/app132111893)

在本研究中的作用：C-MAPSS 作為 RUL 預測領域公開基準資料的地位，以及 LSTM 逐漸
成為主流方法的依據。該文同樣以 FD001 為實驗子集，其 1D-CNN-LSTM 取得 RMSE 16.1。
分段線性（PwL）標註的作法亦出自此一脈絡。

## [3] Kundu & Hoque (2023) — RUL 估計的可信度量化

Kundu, R. K.; Hoque, K. A.
"Explainable Predictive Maintenance is Not Enough: Quantifying Trust in
Remaining Useful Life Estimation."
*Annual Conference of the PHM Society*, 2023, Vol. 15, No. 1.
DOI: [10.36001/phmconf.2023.v15i1.3472](https://doi.org/10.36001/phmconf.2023.v15i1.3472)

在本研究中的作用：RMSE 與 MAE 兩項指標互補、共同評估 RUL 模型效能的依據。

## [4] Mothilall & van Zyl (2024) — LSTM 在航空預測性維護的評估

Mothilall, D.; van Zyl, T. L.
"An evaluation of the Long Short-Term Memory model for predictive maintenance
applications in the aircraft industry."
*2024 International Conference on Artificial Intelligence, Computer, Data Sciences
and Applications (ACDSA)*, IEEE, 2024.
DOI: [10.1109/ACDSA59508.2024.10467634](https://doi.org/10.1109/ACDSA59508.2024.10467634)

在本研究中的作用：LSTM 表現受隱藏層神經元數量、時間視窗大小、學習率等超參數影響
的依據。該文同樣在 C-MAPSS FD001 上做超參數（視窗大小、單元數、Dropout 比例）的
比較分析，與本研究的 Optuna 搜尋空間設定直接對應。

**全文未下載**：IEEE Xplore 需訂閱，查核 Semantic Scholar 的開放取用索引確認
無合法的公開全文。若需全文，可透過學校圖書館的 IEEE Xplore 授權取得。

## [5] Saxena et al. (2008) — C-MAPSS 資料集原始說明

Saxena, A.; Goebel, K.; Simon, D.; Eklund, N.
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation."
*International Conference on Prognostics and Health Management (PHM 2008)*, IEEE, 2008.

在本研究中的作用：FD001 資料的生成方式、感測器意義與失效模式定義的原始出處。
此檔隨 NASA 的 `CMAPSSData.zip` 一同釋出。

---

## BibTeX

```bibtex
@article{wang2023rulrfmlp,
  author  = {Wang, Hairui and Li, Dongwen and Li, Dongjun and Liu, Cuiqin
             and Yang, Xiuqi and Zhu, Guifu},
  title   = {Remaining Useful Life Prediction of Aircraft Turbofan Engine Based on
             Random Forest Feature Selection and Multi-Layer Perceptron},
  journal = {Applied Sciences},
  volume  = {13},
  number  = {12},
  pages   = {7186},
  year    = {2023},
  doi     = {10.3390/app13127186}
}

@article{ensarioglu2023rulcpd,
  author  = {Ensario{\u{g}}lu, K{\i}ymet and {\.{I}}nkaya, T{\"u}lin and Emel, Erdal},
  title   = {Remaining Useful Life Estimation of Turbofan Engines with Deep Learning
             Using Change-Point Detection Based Labeling and Feature Engineering},
  journal = {Applied Sciences},
  volume  = {13},
  number  = {21},
  pages   = {11893},
  year    = {2023},
  doi     = {10.3390/app132111893}
}

@inproceedings{kundu2023trustrul,
  author    = {Kundu, Ripan Kumar and Hoque, Khaza Anuarul},
  title     = {Explainable Predictive Maintenance is Not Enough: Quantifying Trust
               in Remaining Useful Life Estimation},
  booktitle = {Annual Conference of the PHM Society},
  volume    = {15},
  number    = {1},
  year      = {2023},
  doi       = {10.36001/phmconf.2023.v15i1.3472}
}

@inproceedings{mothilall2024lstmpdm,
  author    = {Mothilall, Devesh and van Zyl, Terence L.},
  title     = {An evaluation of the Long Short-Term Memory model for predictive
               maintenance applications in the aircraft industry},
  booktitle = {2024 International Conference on Artificial Intelligence, Computer,
               Data Sciences and Applications (ACDSA)},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/ACDSA59508.2024.10467634}
}

@inproceedings{saxena2008cmapss,
  author    = {Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  title     = {Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation},
  booktitle = {International Conference on Prognostics and Health Management (PHM 2008)},
  year      = {2008},
  publisher = {IEEE}
}
```
