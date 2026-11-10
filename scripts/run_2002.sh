#!/bin/bash
# 2002 chain: geography reconstruction -> census harmonization -> first stage -> validation -> simulation -> groups -> tables
set -e
FROM=${1:-1}
step() { [ "$1" -ge "$FROM" ]; }
cd "$(dirname "$0")/.."
L=data/B_inter/logs; mkdir -p $L
PY="/opt/homebrew/Caskroom/miniforge/base/envs/mapuche-allotments/bin/python -u"
step 1 && echo "[$(date)] reconstruct geography" && $PY -m scripts.tools.reconstruct_2002_geography > $L/2002_geo.log 2>&1
step 2 && echo "[$(date)] 02 census 2002" && $PY -m scripts.data_processing.02_census_harmonize --year 2002 > $L/02_census_2002.log 2>&1
step 2 && { $PY -m scripts.tools.compare_means 2002 > $L/2002_compare.log 2>&1 || true; }
step 2 && { $PY -m scripts.tools.validate_2002_geography > $L/2002_geo_validate.log 2>&1 || true; }
step 3 && echo "[$(date)] 03 models 2002" && $PY -m scripts.data_processing.03_first_stage --year 2002 --spec A B K G > $L/03_models_2002.log 2>&1
step 4 && echo "[$(date)] 04 validate 2002" && $PY -m scripts.data_processing.04_simulate_poverty --validate --year 2002 --spec A B > $L/04_validate_2002.log 2>&1
step 4 && echo "[$(date)] 04 simulate 2002" && $PY -m scripts.data_processing.04_simulate_poverty --year 2002 --spec A B --R 200 > $L/04_simulate_2002.log 2>&1
step 5 && echo "[$(date)] 05 groups" && $PY -m scripts.data_processing.05_build_groups > $L/05_groups.log 2>&1
echo "[$(date)] analysis"
$PY -m scripts.analysis.table_first_stage > $L/a_firststage.log 2>&1; $PY -m scripts.analysis.table_validation > $L/a_validation.log 2>&1
$PY -m scripts.analysis.table1_poverty_2024 --year 2002 > $L/a_table1_2002.log 2>&1
$PY -m scripts.analysis.part2_selfid_and_gap > $L/a_part2_selfid.log 2>&1
for t in 5pp 10pp 01sd; do $PY -m scripts.analysis.part2_pair_did --threshold $t > $L/a_pairdid_$t.log 2>&1; done
$PY -m scripts.analysis.part2_pair_did --threshold 5pp --young > $L/a_pairdid_young.log 2>&1
$PY -m scripts.analysis.part2_pair_did_casen > $L/a_pairdid_casen.log 2>&1 || true
echo "[$(date)] direct classifier 2002"; $PY -m scripts.analysis.direct_poverty_classifier --year 2002 > $L/a_direct_2002.log 2>&1 || true
true
echo "[$(date)] done"
