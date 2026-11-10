# Plan: first working paper draft (eii)

Status: APPROVED-BY-DEFAULT (autonomous session; user asked for a shareable WP soon). 2026-11-06.

## Goal
A shareable working paper, "Ethnic income inequality in Chile: poverty mapping with
machine learning and satellite imagery, and the history of the indigenous poverty gap",
built from a reproducible pipeline laid out like `mapuche-allotments`.

## Paper structure (user's design)
1. **2024 cross-section.** Table with three columns of ethnic poverty rates:
   (a) replication of Agostini, Brown and Román (2010) with CASEN 2024 + Census 2024
   (ELL two-stage, OLS, parametric error simulation);
   (b) same census-available information, LASSO first stage with empirical-residual
   simulation (the model that the imagery column extends);
   (c) (b) + features extracted from Sentinel-2 imagery through the household-to-zone
   probabilistic expansion (proposal architecture).
2. **History of the gap, 1992–2024.** Closed groups g = (5-year birth cohort, sex,
   comuna where the mother lived at birth; 1992 comuna vintage) built from each census.
   (i) document how self-identification changes within groups; (ii) impute poverty
   in each census with the closest CASEN using the column-(b) methodology on a
   harmonized core variable set; (iii) ecological inference of the indigenous/
   non-indigenous poverty gap over time with explicit identification assumptions.

## What the data allow (verified 2026-11-06)
- Census 2024 public microdata: comuna × urban/rural is the finest geography (no zone,
  manzana or distrito). Ethnicity p28; mother's comuna at birth p25 (anonymized, -66, in
  the 24 comunas with < 6,000 people; age also anonymized there). Dwelling, services,
  phone/computer/tablet/internet, schooling, labour status, CISE, CIUO, CAENES.
  => columns (a) and (b) are fully feasible; column (c) is NOT computable for 2024 with
  the public microdata (the household-to-zone model needs zone ids). Report the 2017
  pilot (RA notebooks: test R² 0.432 OLS / 0.432 LASSO / 0.467 LASSO+embeddings;
  proposal: Mapuche poverty 13.7 -> 16.7, rural 24 -> 32, urban 9.5 -> 11) and the roadmap.
- Census 1992: ethnicity (Cultura, age>=14, Mapuche/Aymara/Rapanui), mother's comuna
  (Comuna_Madre_Origen3), assets, dwelling, Puntaje. CASEN 1992 downloaded (no ethnicity
  question, not needed).
- Census 2002: only copy is a cloud-only OneDrive placeholder that does not hydrate.
  BLOCKED until the user makes it available locally (or INE REDATAM export). Pipeline
  leaves a slot (`raw/INE/census_2002`). CASEN 2003 is on disk.
- Census 2017: EXCLUDED by design (incomplete census, no asset questions). Files stay in the data lake unused.
- CASEN 2024 on disk (r3 ethnicity, r1b mother's comuna, pobreza, ytotcorh, v*, r17*).

## Pipeline (scripts/data_processing)
01_casen_harmonize.py      -> B_inter/harmonized/casen_{1992,2003,2024}_hh.parquet
02_census_harmonize.py     -> B_inter/harmonized/census_{1992,2002,2024}_hh.parquet (+ person ethnicity/group ids)
03_first_stage_models.py   -> B_inter/models/ (OLS-ELL and LASSO per wave; CV metrics)
04_simulate_poverty.py     -> B_inter/simulations/ (R=200 draws; ethnicity × area aggregates; group aggregates)
05_build_groups.py         -> C_final/groups_panel.parquet (group × census: n, indigenous share, poverty, covariates)
analysis/table1_poverty_2024.py, analysis/table2_selfid.py, analysis/table3_gap_history.py, figures

## Harmonized variable sets
- `H2024` (Part 1): everything common to Census 2024 and CASEN 2024.
- `K` (Part 2 core, common to 1992/2002/2024 census & CASEN): hh size (+sq), n/frac children
  0-14, n/frac 65+, mean age, female head, head age, head schooling, mean adult schooling,
  head employed, share employed, dwelling type, walls (3 cat), roof (3), floor (3), water
  source (2), sanitation (2), electricity, rooms/bedrooms, urban, region.

## Identification assumptions for Part 2 (write explicitly in the paper)
A1 Closed groups: birth year, sex and mother's comuna are immutable; membership changes
   only through mortality and international migration, assumed non-differential by
   ethnicity within group (or bounded).
A2 Ecological (no aggregation bias): conditional on group and period effects, the
   poverty of non-indigenous members of group g does not vary with the group's
   indigenous share π_g (Goodman regression with group FE: P_gt = α_g + λ_t + Σ_t β_t π_g·1[t] + e_gt,
   E[e_gt | π_g, g, t] = 0 in first differences).
A3 Measurement: π_g taken from 1992 self-identification; classical error attenuates β_t by
   a constant reliability ratio, so the *shape* of β_t over time is identified.
A4 Transferability of the first-stage model from CASEN to census within each period.

## Verification
- First-stage out-of-sample R² reported for (a) and (b); 2024 national poverty from
  the simulation must match CASEN 2024 official 22.3%/… within simulation error.
- Group panel: sum of group sizes vs census totals; self-ID shares vs INE published.

## Deliverables this session
Repo scaffold; scripts 01–05; analysis tables for Part 1 (a),(b) and Part 2 (1992/2024 now; 2002 when the file is recovered);
Overleaf draft `main.tex` with all sections, real numbers from `results/paper_numbers.tex`.

## Progress log
- 2026-11-07 00:00 — Scripts 01–05 and analysis scripts written; 1992 and 2024 harmonized; first stages fit
  (held-out R²: 2024 A 0.40 / B 0.57 / G 0.60; 1992 A 0.51 / B 0.64). In-sample validation: ELL normal errors
  overstate 2024 poverty (20.6 vs 17.0 official); LASSO with empirical residuals reproduces it (17.0).
  Census-vs-survey mean diagnostics led to: 1992 rooms excluding bathrooms, CASEN 1992 wall recode, literacy
  restricted to 15+, 2024 tablet and overcrowding dropped (non-comparable definitions). Full 200-draw run launched.
- Part 2 finding (closed cohorts 1914–1978): aggregate indigenous share 10.6 -> 10.4; Mapuche 9.9 -> 8.5; but
  Mapuche self-ID halves in central Chile (RM 10.2 -> 5.9) and doubles in the south (Los Lagos 11 -> 22, Aysén 6 -> 24);
  northern groups gain through peoples not listed in 1992 (Diaguita, Chango, Atacameño...).
- Open: 2002 census microdata; EO-ML column for 2024 (needs zone-coded microdata); RA's Drive artifacts.

## Scope change 2026-11-07 (Felipe)
- Censuses: 1992, 2002, 2017, 2024 (2017 re-included; weaker first stage, no assets). CASEN 1992, 2003, 2017, 2024.
- EO-ML in every year: Landsat 5 (1992), Landsat 7 (2002; the natural Agostini replication + imagery experiment),
  Landsat 8 / Sentinel-2 (2017), Sentinel-2 (2024). Backbones: SSL4EO-L (torchgeo) for Landsat, BigEarthNet for S2.
- Household-to-zone assignment: random forest on zone-coded microdata for 1992/2002/2017; naive-Bayes and a learned
  matching model on published zone aggregates for 2024, both validated on 2017 held-out comunas (plan in session log).
- 2002 census microdata filed in `_ddbb/raw/INE/census_2002` (persona/hogares/vivienda/uso_pieza .dta + C2002.pdf).
- GPU machine: `ssh fj-office` (key-based login still to be enabled).
- 2026-11-07: 2002 microdata have only a sequential `portafolio` id (4–801,127) and no comuna/zone lookup table; the
  1992 delivery had PORTAFOLIOS.csv (geographically ordered). BLOCKED until a 2002 portafolio table is found (asked Felipe).
  Fallback if none exists: reconstruct comuna blocks from the geographic ordering + published 2002 counts by comuna.
- 2026-11-07: CASEN 2017 harmonized (poverty rule reproduced: monetary income < lp(renter)·n^0.7, lp = 157,832 / 240,632);
  census 2017 harmonization running (keeps zone ids `zone_id` = ID_ZONA_LOC for the household-to-zone model).
- 2026-11-07 (later): Felipe narrows imagery to 2017 and 2024 (Sentinel-2); 2002 parked (no portafolio geography;
  ids are region-ordered, reconstruction possible later). Part 2 panel: 1992/2017/2024, two specs (broad FE and
  saturated m×s×t + c×s×t FE) plus the self-ID persistence regression; see memory note eii-census-years.
- 2026-11-07 02:30 — All simulations done (1992/2017/2024, A and B, 200 draws, Mapuche cells). Part 1 table filled.
  Part 2 redesigned as contiguous-pair DiD around the 1993 Indigenous Law (scripts/tools/comuna_pairs.py,
  scripts/analysis/part2_pair_did.py; 117 pairs at ≥5pp). Result: within-pair poverty gradient on 1992 Mapuche
  share 13.8 (1992) -> 25.6 (2017) -> 28.2 (2024); DiD +11.8 (2.6), +14.4 (3.0); robust to 10pp, young cohorts,
  direct classifier (+15/+17). Checks: schooling placebo drifts -0.48 yrs/unit share; observed housing deprivation
  CONVERGED (dirt floor 15.1 -> 0.4; water -84 -> -63); CASEN-by-residence same sign, imprecise (+29.8, se 25.8).
  Caveat written in the paper: imputed-outcome DiD can reflect changing informativeness of census variables.
- Open: 2002 (parked), imagery 2017/2024 (GPU access), commit of the repo (not requested yet).
- 2026-11-07: two INE transparency requests filed (receipts in quality_reports/requests/): AH007T0014632 (2002 microdata
  with portafolio table) and AH007T0014633 (2024 id_vivienda -> census geography crosswalk). Due 2026-12-07.
- 2026-11-07 08:40 — Residence×year FE version of the pair DiD: gradient 7.3 (1992) -> 11.0 (2017) -> 8.3 (2024);
  DiD +3.7 (1.6), +1.0 (1.9). The origin-only divergence (+14.4) is between places of residence. Written into 5.4.
- 2026-11-07: proxy-logit (1992-model Mapuche probability) idea examined and dropped: equivalent to cell shares under the same
  ecological assumption; scripts moved to scripts/analysis/exploratory/. Large-city exclusion (>100k in 1992): DiD +13.7 (3.1), 105 pairs.
- 2026-11-07 09:10 — 2002 geography reconstruction from enumeration-area order FAILED validation (comuna means vs CASEN 2003
  correlate ~0.1–0.4 under every ordering tried). Decision (Felipe): stop; wait for INE's answer to request AH007T0014632
  (2002 microdata with portafolio table). 2002 files parked in B_inter/harmonized/parked_2002/. CASEN 2003 harmonization is valid.
- 2026-11-10 — GPU machine is `fj-thinkpad` (Tailscale SSH works; RTX 2000 Ada 8 GB, miniforge, Earth Engine credentials,
  no Dropbox). Felipe will pick up the imagery step from that machine. Repo committed and pushed for the first time,
  including quality_reports/ (plans, session logs, INE receipts). CLAUDE.md and README rewritten to the current design.
