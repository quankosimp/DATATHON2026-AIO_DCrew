# VinDatathon 2026 — Task 3: Time-Series Forecasting

**Nhóm:** AIO_DCrew

| Thành viên |
|---|
| Nguyễn Anh Quân |
| Nguyễn Minh Tiến |
| Trần Công Cường |
| Võ Nhật Minh |

---

## Mô tả bài toán

Dự báo **doanh thu** (`Revenue`) và **giá vốn hàng bán** (`COGS`) theo ngày cho giai đoạn **2023-01-01 → 2024-07-01** (548 ngày), dựa trên dữ liệu lịch sử **2012-07-04 → 2022-12-31** (3.833 ngày). Dữ liệu đầu vào chỉ gồm 3 cột: `Date`, `Revenue`, `COGS`.

---

## Kết quả

| Metric | Giá trị |
|---|---|
| MAE (public LB) | 683,504.68 |
| Dự báo Revenue (mean) | ~4,038,171 |
| Dự báo COGS (mean) | ~3,707,846 |

---

## Pipeline tổng quan

```
Raw sales.csv (Date, Revenue, COGS)
        │
        ▼
Feature Engineering — build_features(dates)  [calendar-only, no leakage]
        │  • Calendar: year, month, day, dow, doy, quarter, weekend, EOM/SOM edges
        │  • Trend + regime dummies (pre-2019 / 2019 / post-2019)
        │  • Fourier: yearly (k=1..5), weekly (k=1,2), monthly (k=1,2)  →  24 harmonics
        │  • 6 promotion windows (spring_sale, mid_year, fall_launch,
        │                          year_end, urban_blowout, rural_special)
        │  • Lag-365 từ lịch sử (trung bình lag-364/365/366 để bù năm nhuận)
        │                                               Total: 67 features
        ▼
 Sample weights: w=1.0 (2014–2018, era rõ seasonality), w=0.01 (còn lại)
        │
        ├─── M1: Ridge Regression (log-scale, α=3.0)
        │         Ridge Revenue MSE ≈ 3,022,412
        │
        ├─── M2: LightGBM base (two-stage: early-stopping → retrain)
        │         best_iter Revenue = 251
        │
        ├─── M3: Prophet (log-scale, có promo regressors)
        │         Prophet Revenue MSE ≈ 3,869,609
        │
        └─── M4: Q-Specialists — 4 LightGBM × 2 targets = 8 models
                  Mỗi model boost w×2 cho quý trọng điểm (Q1/Q2/Q3/Q4)
                  Khi predict: ghép kết quả theo quý của test date

                        │
                        ▼
        ┌─────────────────────────────────────┐
        │  Layer 1 — LGB blend                │
        │  lgb_blend = 60% Q-Spec + 40% Base  │
        ├─────────────────────────────────────┤
        │  Layer 2 — 3-way blend              │
        │  raw = 10% Ridge + 10% Prophet      │
        │        + 80% lgb_blend              │
        ├─────────────────────────────────────┤
        │  Layer 3 — Global calibration       │
        │  final_Rev  = 1.26 × raw_Rev        │
        │  final_COGS = 1.32 × raw_COGS       │
        └─────────────────────────────────────┘
                        │
                        ▼
        (Optional) Mean-preserving margin fix
          Kéo Q3 COGS/Revenue margin về historical mean,
          giữ nguyên COGS mean toàn cục  →  cải thiện ~1K LB
                        │
                        ▼
             Submissions/submission.csv
```

---

## Feature Engineering chi tiết

### Calendar (10 features)
`year`, `month`, `day`, `dow`, `doy`, `quarter`, `is_weekend`, `days_to_eom`, `days_from_som`, `dim`

### Edge-of-month (12 features)
`is_last{1,2,3}`, `is_first{1,2,3}` (6 cặp)

### Trend + regime (5 features)
`t_days`, `t_years` (tính từ 2020-01-01), `regime_pre2019`, `regime_2019`, `regime_post2019`

### Fourier harmonics (24 features)
| Chu kỳ | Harmonics | Features |
|---|---|---|
| Yearly (365.25d) | k=1..5 | sin_y1..5, cos_y1..5 |
| Weekly (7d) | k=1,2 | sin_w1..2, cos_w1..2 |
| Monthly (dim d) | k=1,2 | sin_m1..2, cos_m1..2 |

### Promotion windows (6 × 4 = 24 features)
Mỗi promo sinh: `in_{name}`, `days_since_{name}`, `days_until_{name}`, `discount_{name}`

| Promo | Start | Duration | Discount | Recurring |
|---|---|---|---|---|
| spring_sale | Mar-18 | 30d | 12% | Hàng năm |
| mid_year | Jun-23 | 29d | 18% | Hàng năm |
| fall_launch | Aug-30 | 32d | 10% | Hàng năm |
| year_end | Nov-18 | 45d | 20% | Hàng năm |
| urban_blowout | Jul-30 | 33d | — | Năm lẻ |
| rural_special | Jan-30 | 30d | 15% | Năm lẻ |

### Lag features (3 features)
`rev_lag365`, `cogs_lag365`, `margin_lag365` — trung bình của lag-364/365/366 để bù năm nhuận

---

## Leakage control

- `build_features()` **chỉ nhận `dates`**, không nhận Revenue/COGS trực tiếp.
- Lag-365 dùng dữ liệu lịch sử (train set), test set không lấy từ tương lai.
- Walk-forward validation với cutoff **2022-07-04**.

---

## Cấu trúc repo

```
├── P3_Pipeline_Datathon.ipynb   # Notebook pipeline chính
├── Data/
│   ├── sales.csv                # Dữ liệu gốc (Date, Revenue, COGS + extra cols)
│   └── ...                      # Các bảng phụ (customers, orders, products, ...)
├── Submissions/
│   └── submission.csv           # File nộp cuối cùng (548 rows)
└── README.md
```

---

## Cài đặt & chạy lại

```bash
pip install lightgbm prophet shap pandas numpy scikit-learn matplotlib

jupyter notebook "P3_Pipeline_Datathon.ipynb"
```

**Lưu ý về reproducibility:**
- Seed cố định: `seed=42` (LightGBM), `random_state=42` (Ridge).
- Prophet có thể sai khác nhẹ do MCMC sampling.
- Lag-365 NaN (năm đầu train) được điền bằng median cho Ridge; LightGBM tự xử lý NaN natively.
