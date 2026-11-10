# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

`eii` = *ethnic income inequality*. Research project (Felipe Jordán, PUC Chile; RA Felipe Guerrero). Working
paper "Tracking Ethnic Inequality in Chile: 1992–2024" (Overleaf). It estimates poverty by indigenous people in
Chile by combining household surveys (CASEN), census microdata and satellite imagery, and tracks the Mapuche
poverty gap 1992–2024 inside closed birth-cohort × sex × mother's-comuna cells.

Read `quality_reports/plans/2026-11-06_working-paper-setup.md` first (data availability, assumptions, progress
log) and `quality_reports/session_logs/` for the decisions taken in each session. Decisions recorded there
override older wording in the plan's header.

## Paper design (current)

1. **Part 1, 2024 cross-section.** Poverty by ethnicity with three first stages: (a) ELL replication of
   Agostini–Brown–Román (2010) with CASEN 2024 + Census 2024, normal errors; (b) LASSO on every variable the
   census and the survey share, empirical residual draws, comuna-cluster bootstrap; (c) (b) + Sentinel-2
   image features via the household→zone probabilistic expansion. Column (c) is not computable for 2024 with
   the public microdata (comuna-level geography only; ids are shuffled). The paper reports the RA's 2017
   pilot and the roadmap. Never present pilot numbers as 2024 results. Tables are also produced for 1992 and 2017.
2. **Part 2, history.** Closed cells = (5-year birth cohort, sex, mother's comuna at birth, 1992 comuna
   vintage), cohorts 1914–1978 (asked about ethnicity in 1992). Self-identification change is **descriptive
   only** (share in t vs t−1 plots, region table, gradients); no regression claims to explain it. The gap is
   measured with a **contiguous-pair comparison** of adjacent comunas of origin that differ by ≥5 pp in their
   1992 Mapuche share, with pair × year fixed effects, in 1992/2017/2024. Contrast is **Mapuche vs
   non-Mapuche**. It is presented as a descriptive statistic of the gap for a stable population that gets
   around the self-identification problem, **not** as an evaluation of the 1993 Indigenous Law. Robustness:
   residence-comuna × year FE, 1992 comuna controls × year (urban, secondary schooling, deprived housing), no
   large cities, young cohorts, schooling placebo, observed deprivation outcomes, CASEN by residence.
   The earlier fixed-effects ecological panel and the proxy-logit idea were dropped (`scripts/analysis/exploratory/`).
3. **Census years.** 1992, 2017, 2024 are in the pipeline. 2002 is parked: the public microdata carry only a
   sequential `portafolio` id and the local reconstruction of comuna blocks failed validation; do not retry,
   wait for INE request AH007T0014632. 1982 as a pre-period is an open idea. Imagery only for 2017 and 2024.
4. **Funding.** The project was NOT funded by the Cátedra Larraín-Bascuñán. Never acknowledge that funding.

## Environment

```bash
source ~/.bash_profile && mamba activate mapuche-allotments   # python 3.14, pandas 2.3, sklearn 1.8, statsmodels
python configure_data.py --mode dev        # symlinks data/A_raw/raw/* -> ~/Dropbox/_ddbb/raw/*, data/B_inter -> ~/Dropbox/eii/B_inter
python -m scripts.data_processing.01_casen_harmonize --year 1992 2017 2024   # run from project root as modules
```

`data/A_raw` and `data/B_inter` are git-ignored symlinks; `data/C_final` holds small analysis-ready files
(groups panel, comuna pairs, crosswalks). `overleaf/` is a symlink to the Overleaf git clone at
`/Users/fj/overleaf-git/eii`: **always `git pull` there before editing** (Felipe edits live), then
`git add/commit/push` inside that clone to sync. `scripts/tools/sync_overleaf.py` copies tables, figures and
`paper_numbers.tex` into it and compiles.

GPU machine for the imagery step: `ssh fj-thinkpad` (Tailscale SSH, RTX 2000 Ada 8 GB, miniforge installed, no
Dropbox). `fj-office` is not reachable.

## Data (all under ~/Dropbox/_ddbb/raw, append-only)

| Source | Path | Key variables |
|---|---|---|
| Census 1992 | `INE/census_1992/microdata/{PERSONA,HOGAR,VIVIENDA}.csv` + `PORTAFOLIOS.csv` | `Cultura` (age≥14: 1 Mapuche, 2 Aymara, 3 Rapanui, 4 none, 0 not asked), `Comuna_Madre_Origen3`, assets, dwelling |
| Census 2002 | `INE/census_2002/microdata/*.dta` | parked: `portafolio` has no geography; codes in `quality_reports/session_logs/2026-11-07_census2002_codes.md` |
| Census 2017 | `INE/census_2017/Microdato_Censo2017-{Personas,Hogares,Viviendas}.csv` + manzana/entidad shapefiles | no assets, no tenure (renter status imputed); keeps `ID_ZONA_LOC` for the zone model |
| Census 2024 | `INE/census_2024/viv_hog_per_censo2024/*.parquet` | `p28_autoid_pueblo/p28_pueblo_pert`, `p25_lug_nacimiento_esp` (−66 anonymized in `comuna_bajo_umbral==1`), comuna × area only |
| Cartography 2024 | `INE/Cartografia_censo2024_Pais/*.parquet` | zone/manzana/entidad polygons with population and dwelling counts |
| CASEN | `MDS/casen/casen_{1992,2003,2017,2024}.dta` | 2024: `r3` ethnicity, `r1b` mother's comuna, `pobreza`, `ytotcorh`; 2003: `r25`, `corte`, `ytothaj` |
| Comunas | `BCN/comunas` | current polygons, used for contiguity (`scripts/tools/comuna_pairs.py`) |

Official poverty rules are reproduced exactly per wave (2024/2017: income < lp(renter) · n^0.7; 1992/2003:
per-capita lines by urban/rural) and checked against each CASEN's official status variable.

## Pipeline

`scripts/data_processing/` 01 casen harmonize → 02 census harmonize → 03 first stage (specs A ELL, B LASSO,
K core, G) → 04 simulate poverty (200 draws; cells by ethnicity × urban/cohort/region, Mapuche cells, origin ×
residence) → 05 build groups (`C_final/groups_panel.parquet`). `scripts/analysis/` has one script per exhibit;
each writes to `results/tables/`, `results/figures/` and `results/paper_numbers.tex` via
`scripts/tools/paper_results.py`. `scripts/run_pipeline.sh` (1992/2024 core) and `scripts/run_final.sh` (2017,
Mapuche cells, panels, tables, Overleaf sync) are resumable: `bash scripts/run_final.sh 3` starts at step 3.
`scripts/run_2002.sh` is parked.

`scripts/eo_ml/` keeps the RA's Earth-observation pipeline: `gee/` (Google Earth Engine downloads of
Landsat/Sentinel-2 mosaics per census zone) and `pilot_2017_notebooks/` (Colab notebooks: CNN fine-tuning of
BigEarthNet ResNet-50 on zone means, per-comuna random forests for P(zone | X), LASSO on the expanded CASEN
2017). Their data live on the RA's Google Drive; when recovered, drop them in `~/Dropbox/eii/received/`.
The multi-year plan is `quality_reports/plans/2026-11-07_eo_ml_multi_year.md`.

## Conventions

- Plans in `quality_reports/plans/`, session decisions in `quality_reports/session_logs/`, information
  requests (Ley 20.285 receipts) in `quality_reports/requests/`. These folders are committed.
- Never put raw data in git. Keep `C_final` files < 100 MB.
- Spanish variable names from the sources are kept in the harmonization scripts; harmonized names are English snake_case.
- Official INE/MDS requests are written as one short plain paragraph (the portal breaks lists).
