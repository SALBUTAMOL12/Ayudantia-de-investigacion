#!/bin/bash
# Post-pipeline chain: 2017 simulations, Mapuche cells for all years, panels, tables, Overleaf sync.
set -e
FROM=${1:-1}
step() { [ "$1" -ge "$FROM" ]; }
cd "$(dirname "$0")/.."
L=data/B_inter/logs; mkdir -p $L
PY="/opt/homebrew/Caskroom/miniforge/base/envs/mapuche-allotments/bin/python -u"
step 1 && echo "[$(date)] 04 validate 2017" && $PY -m scripts.data_processing.04_simulate_poverty --validate --year 2017 --spec A B > $L/04_validate_2017.log 2>&1
step 2 && echo "[$(date)] 04 simulate 2017" && $PY -m scripts.data_processing.04_simulate_poverty --year 2017 --spec A B --R 200 > $L/04_simulate_2017.log 2>&1
step 3 && echo "[$(date)] 04 rerun 1992/2024 (Mapuche cells; bootstraps cached)" && $PY -m scripts.data_processing.04_simulate_poverty --year 1992 2024 --spec A B --R 200 > $L/04_simulate_rerun.log 2>&1
step 4 && echo "[$(date)] 05 groups" && $PY -m scripts.data_processing.05_build_groups > $L/05_groups.log 2>&1
echo "[$(date)] analysis"
$PY -m scripts.analysis.table_first_stage > $L/a_firststage.log 2>&1
$PY -m scripts.analysis.table_validation > $L/a_validation.log 2>&1
for y in 2024 1992 2017; do $PY -m scripts.analysis.table1_poverty_2024 --year $y > $L/a_table1_$y.log 2>&1; done
$PY -m scripts.analysis.part2_selfid_and_gap > $L/a_part2_selfid.log 2>&1
for smp in adults young younghh; do $PY -m scripts.analysis.part2_gap_panel --spec B --sample $smp > $L/a_part2_panel_$smp.log 2>&1; done
$PY -m scripts.analysis.part2_gap_panel --spec B --sample adults --measure indigenous > $L/a_part2_panel_ind.log 2>&1
echo "[$(date)] direct classifier";  $PY -m scripts.analysis.direct_poverty_classifier --year 1992 2017 2024 > $L/a_direct.log 2>&1
echo "[$(date)] sync overleaf";      $PY -m scripts.tools.sync_overleaf > $L/a_sync.log 2>&1
true
echo "[$(date)] done"
