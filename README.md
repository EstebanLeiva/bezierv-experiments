# bezierv-experiments — Reproducibility scripts for the paper

Command-line scripts that regenerate every table and figure from the paper's
numerical experiments. Each artefact is produced by exactly one script.

| Paper artefact                                  | Script                  | Output                                     |
|-------------------------------------------------|-------------------------|--------------------------------------------|
| Figure 4 — multimodal Bézier fit                | `fig_multimodal.py`     | `figures/multimodal.pgf`                   |
| Table 1 — MLE comparison, synthetic data        | `tab_synthetic.py`      | `tables/syntheticmle.tex` (+ raw CSV)      |
| Table 2 — MSE comparison, synthetic data        | `tab_synthetic.py`      | `tables/syntheticmse.tex`                  |
| Table 3 — Chicago descriptive statistics        | `tab_real_stats.py`     | `tables/realdist.tex`                      |
| Table 4 — MLE comparison, transportation data   | `tab_real_mle.py`       | `tables/realmle.tex` (+ raw CSV)           |
| Table 5 — MSE comparison, transportation data   | `tab_real_mse.py`       | `tables/MDEtab.tex` (+ raw CSV)            |
| Figure 5 — restriction quality vs degree        | `fig_restriction.py`    | `figures/restriction.pgf` (+ raw CSV)      |
| Fig. 5 companion — empirical $\mathcal{O}(1/N)$ rate (supplementary) | `fig_restriction_rate.py` | `figures/restriction_rate.pgf` (+ raw & checkpoint CSV) |
| §8 — S-αRP application (Figures 6a & 6b)        | `exp_sarp.py`           | `figures/sarp_path_grid.pgf`, `figures/sarp_path_dist.pgf` + console |

The toy examples (Figures 1, 2, 3a, 3b) are intentionally not reproduced — they
are TikZ illustrations or GUI screenshots that do not depend on experimental
data.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The Chicago dataset (`data/Chicago_main.json`, ≈280 MB) is the timestamped raw
data used by `tab_real_*` and `exp_sarp.py`. It is not bundled with this
repository; download it from Zenodo (CC BY 4.0) and place it under `data/`:

> Goerigk, M., Hartisch, M., Merten, S., & Miehlich, A. (2025). *Chicago Graph
> and Travel Time Data* (v1) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.15267992

## Smoke mode vs. paper-faithful runs

Three scripts default to a small subset so iteration is fast; pass the
documented flag to regenerate the paper-faithful numbers:

```bash
# Tables 1, 2 (synthetic):   3 instances/family  ->  200 instances/family
python tab_synthetic.py --n-per-family 200

# Tables 4, 5 (Chicago):     20 arcs/regime   ->  all arcs (~1086/regime)
python tab_real_mle.py --max-arcs 0
python tab_real_mse.py --max-arcs 0
```

The remaining scripts already default to paper-faithful settings; for these a
flag instead gives a *quicker* run:

```bash
# Figure 4 (multimodal), Table 3 (stats): always paper-faithful, and fast
python fig_multimodal.py
python tab_real_stats.py

# Figure 5 (restriction): default sweeps the paper range n=3..20
python fig_restriction.py                 # paper sweep
python fig_restriction.py --n-max 6       # quick look

# Fig. 5 companion (O(1/N) rate): default reproduces it (slow; resumable)
python fig_restriction_rate.py                       # paper-faithful
python fig_restriction_rate.py --n-reps 3 --n-max 30 # quick look

# §8 (S-αRP): default uses the full Chicago network
python exp_sarp.py                        # paper-faithful
python exp_sarp.py --max-arcs 200         # smoke run on a subset
```

## Wiring outputs back into the paper

Tables emit a complete `\begin{table}...\end{table}` block

Figures emit a `.pgf` file (for `\input{}` into the paper) and a sibling `.png`
of the same name for quick visual inspection without compiling LaTeX.

## Layout

```
bezierv-experiments/
├── README.md, requirements.txt
├── data/                       # Chicago_main.json (raw, 280 MB)
├── figures/                    # generated .pgf
├── tables/                     # generated .tex + raw .csv
├── _plot_style.py              # matplotlib rcParams (publication-quality)
├── _latex_tables.py            # booktabs/multirow formatters
├── _fit_benchmark.py           # Bezier/Beta/Johnson MLE + Bezier MSE fits
├── _chicago_graph.py           # Chicago_main.json -> per-regime arc dicts
├── _sarp.py                    # pulse callbacks + Bezier-weighted graph builder
├── fig_multimodal.py
├── fig_restriction.py
├── fig_restriction_rate.py
├── tab_synthetic.py
├── tab_real_stats.py
├── tab_real_mle.py
├── tab_real_mse.py
└── exp_sarp.py
```

## Reproducibility notes

- All scripts accept `--seed` (default `42`). The synthetic and SARP runs are
  deterministic given the seed. The Chicago lognormal synthesis (used only for
  arcs with a single unique observation) is also seeded.
