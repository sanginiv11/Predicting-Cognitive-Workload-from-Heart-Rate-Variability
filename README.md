# Predicting Cognitive Workload from Heart Rate Variability: A Multiple Linear Regression Approach

An individual ML-regression mini-project. It asks whether **Multiple Linear Regression (MLR)** can predict a person's self-reported **NASA-TLX workload score** from **heart rate variability (HRV)** features, and how well it does on **participants it has never seen**.

The emphasis is on honest evaluation: a participant-wise split, a baseline to beat, cross-validation, and a clear account of why the first model fails.

## Dataset

[Heart Rate Variability and Cognitive (Mental) Load](https://www.kaggle.com/datasets/audityasutarto/heart-rate-variability-and-cognitive-mental-load) (Kaggle, file `hrv_mwl_30.csv`).

- 30 participants x 2 trials = 60 rows (semicolon-separated file)
- Features: `hr`, `sdnn`, `rmssd`, `lf`, `hf`, `lfhf`
- Target: `nasa`, the NASA-TLX score (0 to 100, self-reported)

## Method

1. **Data checks:** missing values, duplicates, target verification, plausibility check. One row with an impossible `hf` value (332.91, but `hf` is a percentage) was **excluded rather than corrected**, leaving 59 rows.
2. **EDA:** target distribution, correlation matrix, each feature against `nasa`.
3. **Participant-wise split** with `GroupShuffleSplit` (24 participants train, 6 test). Each person has two rows, so a random row split would leak the same person into both sets.
4. **Feature selection with VIF:** `lf` and `hf` are complementary percentages (VIF above 800), so `hf` and `lfhf` were dropped. Final features: `hr, sdnn, rmssd, lf`.
5. **Model:** ordinary least squares MLR, compared with a "predict the training mean" baseline.
6. **Evaluation:** MAE, RMSE and R² on the held-out test set, plus **leave-one-participant-out cross-validation**.
7. **Follow-ups on the negative R²:** Ridge regression (alpha tuned by grouped CV) and within-participant centering.

## Results

| Model / evaluation | n | MAE | RMSE | R² |
|---|---|---|---|---|
| MLR, train | 47 | 12.41 | 14.80 | 0.068 |
| MLR, held-out test | 12 | 18.63 | 21.44 | -0.384 |
| Baseline (train mean), test | 12 | 18.32 | 21.70 | -0.417 |
| MLR, leave-one-participant-out CV | 59 | 15.06 | 17.47 | -0.100 |
| Baseline, leave-one-participant-out CV | 59 | 14.30 | 17.09 | -0.054 |

### Main finding

**Linear regression on absolute HRV values does not beat simply predicting the average workload for new participants.** R² is negative on both the test set and cross-validation, and the baseline has similar or lower error.

Why it fails:
- **Tiny sample:** 47 training rows, and only 6 test participants, so metrics swing with the split.
- **Collinear features:** `sdnn` and `rmssd` still overlap (VIF about 6 to 9), giving huge opposite-signed standardised coefficients (+8.98 and -9.71) that don't generalise.
- **People differ at baseline:** resting HRV and heart rate vary a lot between individuals, and NASA-TLX is a subjective score with personal bias.

### Attempted fixes

| Approach | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Original MLR (absolute values) | 18.625 | 21.442 | -0.384 |
| Ridge regression (absolute values) | 18.315 | 21.706 | -0.418 |
| Within-participant centered MLR | 5.999 | 7.426 | 0.609 |


- **Ridge** did not help. The tuned penalty (alpha = 10,000) shrinks the coefficients almost to zero, so the model effectively reverts to predicting the mean.
- **Within-participant centering** subtracts each person's own mean from their features and their NASA score, so the model predicts *change relative to that person's baseline*.

**How to read the centered result.** The R² of 0.609 answers a *different and easier question* than the first model, and it is not directly comparable:
- The target is now the deviation from a person's own mean score, and computing that mean needs the person's actual NASA scores. So this is not a way to predict an unseen person's workload from HRV alone.
- With only two trials per person, centering reduces each participant to a pair of mirrored points, and the test set contains only 6 participants. The result is suggestive but statistically fragile.
- The takeaway is better stated as: **within a person, trial-to-trial changes in HRV track trial-to-trial changes in reported workload; across people, absolute HRV does not predict workload with this data.**

## Limitations

- Very small sample (30 participants, 12 test rows) and a single train/test split.
- NASA-TLX is subjective; HRV is also affected by age, fitness, breathing, caffeine and so on.
- One row was excluded for an impossible value.
- Linear models only; results show association, not causation, and may not generalise beyond this experiment.

## Future work

More participants, a proper per-person calibration baseline (for example a rest-period recording rather than the person's own test scores), extra signals (EDA, respiration), and non-linear or mixed-effects models.

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn kagglehub
jupyter notebook MLR_Cognitive_Workload__posR2.ipynb
```
The notebook looks for `hrv_mwl_30.csv` next to it, then tries a Kaggle download, then asks for an upload (on Colab).

## Repository contents

```
.
├── MLR_Cognitive_Workload__posR2.ipynb
├── images/                  # figures from the notebook
├── results_metrics.csv      # generated when you run the notebook
├── results_coefficients.csv # generated when you run the notebook
└── README.md
```

## References

- Izzah, Sutarto and Hariyadi (2022). *Jurnal Teknik Industri*, 24(2), 83-94.
- Izzah et al. (2023). *Jurnal Optimasi Sistem Industri*, 22(2), 81-98.

## Author

Sangini Verma, B.Tech Electronics & Telecommunication Engineering, Symbiosis Institute of Technology, Pune
