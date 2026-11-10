# Session 2026-11-10: GPU machine and first push of the repo

## Decisions
- The GPU machine for the imagery step is `fj-thinkpad` (Tailscale SSH; Ubuntu 24.04; NVIDIA RTX 2000 Ada 8 GB;
  32 cores; 125 GB RAM; ~820 GB free; miniforge with envs base/lbn-chile/mapuche-allotments/mapuche-mayors, no torch yet;
  Earth Engine credentials present; **no Dropbox**). `fj-office` refuses key authentication and is dropped.
- Felipe takes the imagery work up from that machine himself (clone the repo there; rsync inputs listed in README).
- The repository is pushed to `github.com/SALBUTAMOL12/Ayudantia-de-investigacion` (the RA's repo, public), including
  `quality_reports/` (plans, session logs, INE request receipts). Raw and intermediate data stay out of git
  (`data/A_raw`, `data/B_inter`, `overleaf/` ignored); `data/C_final` (16 MB) and `results/` are committed.
- `CLAUDE.md` and `README.md` rewritten to the current design: censuses 1992/2017/2024, 2002 parked, Part 2 as a
  descriptive contiguous-pair comparison (Mapuche vs non-Mapuche), self-identification descriptive, imagery 2017/2024.

## Suggested next steps on fj-thinkpad
1. `mamba create -n eii-gpu python=3.12 pytorch torchvision pytorch-cuda rasterio geopandas earthengine-api scikit-learn pandas pyarrow`.
2. Sentinel-2 summer mosaics per zone for 2017 and 2024 (`scripts/eo_ml/gee/download_batch_s2.py`, write to local disk).
3. Fine-tune BigEarthNet ResNet-50 on 2017 zone means (plan `2026-11-07_eo_ml_multi_year.md`, step 3) and export
   zone features for both years; 2017 assignment with the RA's per-comuna random forest, 2024 with naive Bayes /
   matching on the published aggregates until INE answers request AH007T0014633.

## Still open
- INE requests AH007T0014632 (2002 microdata with portafolio table) and AH007T0014633 (2024 crosswalk), due 2026-12-07.
- 1982 census as a pre-period for the pair comparison (needs microdata: INE or IPUMS; outcomes would be observed
  deprivation or CASEN 1985/1987 imputations). Not decided.
