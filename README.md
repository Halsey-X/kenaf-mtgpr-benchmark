# Reproduction package

**Small-Sample Multi-Task Prediction of Pretreated Kenaf Core Composition: Multi-Task Gaussian Process Regression Benchmarked against Field-Standard, Modern, and Trivial Baselines**

This repository contains the dataset and all code needed to reproduce every
table and figure in the manuscript.

## Repository layout

```
data/kenaf_pretreatment_dataset.xlsx   # 31-sample experimental matrix (time, temperature, pH; cellulose, hemicellulose, lignin; glucose yield)
code/                                  # all analysis scripts (no absolute paths; run from anywhere)
process_data/                          # pre-computed intermediate results (regenerated when scripts re-run)
```

## Environment

Python >= 3.10, then:

```
pip install -r requirements.txt
```

## Script-to-output map

| Script | Produces |
|---|---|
| `code/mtgpr.py` | Multi-task Gaussian process regression (ICM kernel) implementation |
| `code/test_mtgpr.py` | Unit sanity checks for the MTGPR implementation |
| `code/analysis.py` | Main benchmark (13 models, 20 repeated support-query splits, support-size ablation 4/6/8/10/12, LOOCV, Friedman/Nemenyi, RF descriptor importance) -> Tables 2, 4, 5; Fig. 1-7 |
| `code/figures.py` | Manuscript figures from `process_data/results.json` |
| `code/revise_additions.py` | RSM (ridge quadratic) baseline, downstream composition->yield chain -> Table 2 (RSM row), Fig. 12 |
| `code/round2_fix.py` | Mass-conservation-constrained C-MTGPR (ILR coordinates) negative-result experiment -> Section on constrained predictions |
| `code/round2_loocv.py` | Pooled LOOCV RMSE for all models -> Table 5 |
| `code/rebuild_table3.py` | Paired t-tests with Holm correction (13 comparisons) -> Table 3 |
| `code/q1_upgrade.py` | PLS / XGBoost / route-mean lookup baselines, leave-one-route-out, cross-conformal intervals -> Table 6, Fig. 8, 9 |
| `code/q1_stats_fig.py` | Paired bootstrap CIs and Figs. 8-9 |
| `code/loro_paired.py` | Reproducible (5-seed) leave-one-route-out with paired tests -> Table 6 |
| `code/round2_ablation.py` | Support-size ablation and LOOCV tables |
| `code/round2_tables_fig.py` / `code/round2_stats.py` / `code/make_fig12.py` | In-sample vs cross-validated R^2 comparison (Fig. 13) and supplementary statistics |

Notes
- `analysis.py` is imported as a module by several scripts; it loads
  `data/kenaf_pretreatment_dataset.xlsx` relative to the repository root.
- Stochastic models (MTGPR, ST-PN) are seeded; leave-one-route-out results are
  averaged over 5 seeds as stated in the manuscript.

## License

MIT (see `LICENSE`).

## Citation

If you use this code or dataset, please cite the associated manuscript
(link to be added upon publication).
