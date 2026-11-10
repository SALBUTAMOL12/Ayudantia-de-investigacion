# Plan: EO-ML first stage in every census year (1992, 2002, 2017, 2024)

Status: DRAFT (2026-11-07; narrowed 2026-11-07 later to Sentinel-2 in 2017 and 2024 only, Landsat rows kept for the record). GPU: `fj-thinkpad` (reachable 2026-11-10; `fj-office` is not).

## Inputs per year

| Year | Census zones | Zone id in microdata | Imagery | Backbone |
|---|---|---|---|---|
| 1992 | zona/manzana (urban), localidad/entidad (rural) via PORTAFOLIOS.csv | yes (Portafolio -> Comuna/Distrito/Zona/Manzana) | Landsat 5 TM, 30 m, 1991–1993 composite | SSL4EO-L (torchgeo) ResNet-50, TM weights |
| 2002 | same scheme | **missing** (sequential `portafolio`, no lookup) | Landsat 7 ETM+, 30 m (+15 m pan), 2001–2002 (pre-SLC-off) | SSL4EO-L, ETM+ weights |
| 2017 | ZC_LOC / ID_ZONA_LOC | yes | Landsat 8 OLI 2016–2018 and Sentinel-2 2017 | SSL4EO-L OLI / BigEarthNet S2 |
| 2024 | 4,911 zones + 216k manzanas (INE cartography, with aggregates) | **no** (comuna only) | Sentinel-2 2023–2024 | BigEarthNet ResNet-50 (RA's pilot) |

Zone polygons: 2024 cartography (parquet) is in `_ddbb`; 2017 polygons in `census_2017/Microdatos_{Manzana,Entidad}.shp`;
1992 sector polygons exist only for the Araucanía study region (`sectores_92_updated.shp`) -> need INE 1992/2002
zone cartography (REDATAM maps) or approximate 1992/2002 zones by their 2017 counterparts where codes persist.

## Steps
1. **Zone centroids/stacks** (`scripts/eo_ml/gee/`): for each zone, building centroids (Google Open Buildings for 2017/2024;
   for 1992/2002 use the zone polygon grid, since Open Buildings reflects the present) -> boxes of 256 px.
2. **Download** (`download_batch_multi.py`, sensors L5/L7/L8/S2, quality mosaic by NDVI, cloud masks): one Drive folder
   per year; then `generate_images.py` cuts per-zone tifs.
3. **Fine-tune** per year (notebook `modelo_1_CNN.ipynb` generalized to a script `scripts/eo_ml/train_zone_cnn.py`):
   targets = zone means of the harmonized household variables (from `census_{year}_hh.parquet` + zone id; for 2024 the
   published aggregates). Output: 512-d penultimate features F(z) -> `B_inter/eo/{year}_zone_features.parquet`.
4. **Household-to-zone probabilities** (`scripts/eo_ml/zone_assignment.py`):
   - 1992/2002/2017: per-comuna random forest on census microdata (RA's `modelo_2_probabilidades.ipynb`).
   - 2024: naive Bayes on published zone aggregates with temperature τ, and a learned matching model
     (household x, zone profile a_z) -> P(z|x) trained on 2017; both validated on 2017 held-out comunas against the RF
     (log-likelihood, calibration, downstream R²).
5. **Expanded first stage** (`scripts/eo_ml/expanded_lasso.py`): CASEN household x zones of its comuna with weights
   p_z(x)·expr; LASSO on [x, F(z)]; prediction in census with F(z) (zone known) or E[F|x,c] (2024).
6. Plug into `04_simulate_poverty.py` as spec `C`.

## Open items
- 2002 portafolio -> geography lookup (blocks 2002 entirely, including Part 2).
- 1992/2002 zone polygons nationwide.
- GPU access solved (`fj-thinkpad`, 2026-11-10); the laptop has no Dropbox, inputs must be rsynced (README).
