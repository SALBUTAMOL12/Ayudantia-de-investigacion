# eii — Ethnic income inequality in Chile

Working paper: *Tracking Ethnic Inequality in Chile: 1992–2024* (Felipe Guerrero and Felipe Jordán, PUC Chile).

The paper (i) replicates the ELL poverty-mapping estimates of Agostini, Brown and Román (2010) with the 2024 census and CASEN 2024, (ii) replaces the first stage by a LASSO on every variable the census and the survey share and by an architecture that adds Sentinel-2 image features through a household-to-zone expansion (2017 pilot), and (iii) follows the Mapuche poverty gap 1992–2017–2024 inside closed cells (birth cohort × sex × mother's comuna at birth), describing self-identification change within cells and measuring the gap for a stable population with a contiguous-pair comparison of neighboring comunas of origin (pair × year fixed effects). The pair comparison is a descriptive statistic, not an evaluation of the 1993 Indigenous Law.

## Layout

```
configure_data.py        wires data/A_raw -> ~/Dropbox/_ddbb and data/B_inter -> ~/Dropbox/eii/B_inter
scripts/directories.py   paths; import first in every script
scripts/data_processing/ 01 casen harmonize | 02 census harmonize | 03 first stage | 04 simulate | 05 groups
scripts/analysis/        one script per exhibit -> results/tables, results/figures, results/paper_numbers.tex
scripts/tools/           hh_features, features (design matrices), comuna_xwalk, compare_means, sync_overleaf
scripts/eo_ml/           GEE download scripts and the RA's 2017 Colab notebooks (CNN, RF, expanded LASSO)
scripts/run_pipeline.sh  core 1992/2024 chain; scripts/run_final.sh adds 2017, Mapuche cells, panels, tables, Overleaf sync (`bash scripts/run_final.sh 3` resumes from step 3); scripts/run_2002.sh is parked
overleaf/ -> ~/overleaf-git/eii   manuscript (main.tex); `python -m scripts.tools.sync_overleaf` copies results and compiles
quality_reports/          plans/ (design + progress log), session_logs/ (decisions), requests/ (INE Ley 20.285 receipts)
```

## Running

```bash
mamba activate mapuche-allotments        # or: pip install -r requirements.txt
python configure_data.py --mode dev
bash scripts/run_pipeline.sh && bash scripts/run_final.sh   # several hours; logs in data/B_inter/logs/
python -m scripts.tools.sync_overleaf    # tables/figures/macros into overleaf/ and pdflatex
```

## Data

All raw inputs are public and live in `~/Dropbox/_ddbb/raw` (INE censuses 1992, 2017 and 2024 with the 2024 cartography; CASEN 1992, 2003, 2017, 2024; BCN comuna polygons). The 2002 census microdata are on disk but parked: the public file has no geography beyond a sequential enumeration-area id, and an INE transparency request for the lookup table is pending (`quality_reports/requests/`). The 2017 census has no assets and no tenure (renter status is imputed), so its first stage is weaker.

Poverty rules verified against the official status variable of each CASEN: 2024, monetary household income below the line (237,094 pesos per equivalent adult for non-renters, 352,359 for renters) times household size^0.7; 1992, per-capita total income below 25,750 (urban) / 17,362 (rural).

## Running on the GPU laptop (fj-thinkpad)

The imagery step (Sentinel-2 features for 2017 and 2024, `quality_reports/plans/2026-11-07_eo_ml_multi_year.md`) runs on `fj-thinkpad` (Ubuntu 24.04, RTX 2000 Ada 8 GB, miniforge). The laptop has no Dropbox, so after `git clone` run `python configure_data.py --mode dev --data-root <local copy> --inter-root <local copy>` and rsync from the Mac the inputs the step needs: `_ddbb/raw/INE/Cartografia_censo2024_Pais/*.parquet`, `_ddbb/raw/INE/census_2017/Microdatos_{Manzana,Entidad}.*`, and `~/Dropbox/eii/B_inter/harmonized/census_{2017,2024}_hh.parquet`. Earth Engine credentials are already in `~/.config/earthengine` there.
