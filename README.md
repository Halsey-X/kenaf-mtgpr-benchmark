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
| `code/analysis.py` | Main benchmark: 13 candidates, 20 repeated support-query splits, support-size ablation (4/6/8/10/12), LOOCV, MTGPR hyperparameter sensitivity, RF descriptor importance -> Tables 2, 4, 5; Fig. 1-6, 9 |
| `code/figures.py` | Manuscript figures rendered from `process_data/results.json` (Fig. 1-6, 9) |
| `code/revise_additions.py` | RSM (ridge quadratic) baseline, MTGPR LOOCV parity, downstream composition -> glucose yield chain -> Table 2 (RSM row); Fig. 10, 11, 12 |
| `code/round2_fix.py` | Mass-conservation-constrained C-MTGPR (ILR coordinates) negative-result experiment -> Section 3.8 |
| `code/round2_loocv.py` | Pooled LOOCV RMSE for all models -> Table 5 |
| `code/round2_ablation.py` | Support-size ablation and LOOCV tables -> Tables 4, 5 |
| `code/rebuild_table3.py` | Paired t-tests with Holm correction (13 comparisons) -> Table 3 |
| `code/q1_upgrade.py` | PLS / XGBoost / route-mean lookup baselines, leave-one-route-out, cross-conformal intervals -> Tables 2, 6; Fig. 7, 8 |
| `code/q1_stats_fig.py` | Paired bootstrap CIs; renders Fig. 7 (LORO) and Fig. 8 (conformal) |
| `code/loro_paired.py` | Reproducible (5-seed) leave-one-route-out with paired tests -> Table 6; Fig. 7 |
| `code/round2_tables_fig.py` / `code/round2_stats.py` / `code/make_fig12.py` | In-sample (full-fit) vs leave-one-out R^2 for glucose-yield prediction -> Fig. 13 |

Notes
- `analysis.py` is imported as a module by several scripts; it loads
  `data/kenaf_pretreatment_dataset.xlsx` relative to the repository root.
- Stochastic models (MTGPR, ST-PN) are seeded; leave-one-route-out results are
  averaged over 5 seeds as stated in the manuscript.
- Figure numbers above follow the manuscript (numbered by order of first
  citation). Two scripts still write their PNGs under the legacy filenames
  assigned before that renumbering: `fig12_loro.png` -> manuscript Fig. 7,
  `fig13_conformal.png` -> Fig. 8, `fig11_downstream_r2.png` -> Fig. 13.

## License

MIT (see `LICENSE`).

## Citation

If you use this code or dataset, please cite the associated manuscript
(link to be added upon publication).
