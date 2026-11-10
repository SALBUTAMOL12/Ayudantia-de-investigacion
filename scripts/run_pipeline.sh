#!/bin/bash
# Full pipeline for the eii working paper (1992 and 2024). Logs in data/B_inter/logs/.
set -e
FROM=${1:-1}   # first step to run (1..6)
step() { [ "$1" -ge "$FROM" ]; }
cd "$(dirname "$0")/.."
L=data/B_inter/logs; mkdir -p $L
PY="/opt/homebrew/Caskroom/miniforge/base/envs/mapuche-allotments/bin/python -u"
step 1 && echo "[$(date)] 01 casen" && $PY -m scripts.data_processing.01_casen_harmonize --year 1992 2024 > $L/01_casen.log 2>&1
step 2 && echo "[$(date)] 02 census" && $PY -m scripts.data_processing.02_census_harmonize --year 1992 2024 > $L/02_census.log 2>&1
step 3 && echo "[$(date)] 03 models" && $PY -m scripts.data_processing.03_first_stage --year 2024 1992 --spec A B K G > $L/03_models.log 2>&1
step 4 && rm -f data/B_inter/models/*_boot.npz && echo "[$(date)] 04 validate" && $PY -m scripts.data_processing.04_simulate_poverty --validate --year 1992 2024 --spec A B > $L/04_validate.log 2>&1
step 4 && echo "[$(date)] 04 simulate" && $PY -m scripts.data_processing.04_simulate_poverty --year 1992 2024 --spec A B --R 200 > $L/04_simulate.log 2>&1
step 5 && echo "[$(date)] 05 groups" && $PY -m scripts.data_processing.05_build_groups > $L/05_groups.log 2>&1
step 6 && echo "[$(date)] analysis" && $PY -m scripts.analysis.table_first_stage > $L/a_firststage.log 2>&1
$PY -m scripts.analysis.table1_poverty_2024 > $L/a_table1.log 2>&1
$PY -m scripts.analysis.part2_selfid_and_gap --spec B > $L/a_part2.log 2>&1
true
echo "[$(date)] done"
