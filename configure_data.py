#!/usr/bin/env python3
"""
Configure data sources for the eii project.

    python configure_data.py --mode dev   [--data-root ~/Dropbox/_ddbb] [--inter-root ~/Dropbox/eii]
    python configure_data.py --mode public

dev:    create symlinks data/A_raw/<key> -> <data-root>/<key> and
        data/B_inter -> <inter-root>/B_inter
public: create empty data/A_raw/<key> folders and print download instructions.
"""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_SOURCES = {
    "raw/INE/census_1992": dict(access="public",
        description="1992 Census microdata (PERSONA/HOGAR/VIVIENDA + lookups) and sector maps",
        url="https://www.ine.gob.cl/", instructions="INE transparency request or IPUMS; see _ddbb README"),
    "raw/INE/census_2002": dict(access="public",
        description="2002 Census microdata (REDATAM export)",
        url="https://www.ine.gob.cl/", instructions="INE REDATAM export; NOT YET IN THE DATA LAKE (see quality_reports/plans)"),
    "raw/INE/census_2017": dict(access="public",
        description="2017 Census microdata: Personas, Hogares, Viviendas CSV + manzana/entidad polygons",
        url="https://www.ine.gob.cl/estadisticas-por-tema/demografia-y-poblacion/censo-de-poblacion-y-vivienda",
        instructions="Download csv-personas/hogares/viviendas-censo-2017.zip from INE"),
    "raw/INE/census_2024": dict(access="public",
        description="2024 Census microdata (parquet: viviendas, hogares, personas) + codebook + manual",
        url="https://www.ine.gob.cl/", instructions="INE microdata portal"),
    "raw/INE/Cartografia_censo2024_Pais": dict(access="public",
        description="2024 Census cartography with zone/manzana/entidad aggregates",
        url="https://www.ine.gob.cl/", instructions="INE geoportal"),
    "raw/INE/dpa_historica": dict(access="public",
        description="Historical comuna boundaries by census year (1940-2002)",
        url="https://www.ine.gob.cl/", instructions="INE transparency request"),
    "raw/MDS/casen": dict(access="public",
        description="CASEN microdata: 1992, 2003, 2017, 2024 (.dta) + codebooks",
        url="https://observatorio.ministeriodesarrollosocial.gob.cl/encuesta-casen",
        instructions="Download each wave's Stata database from the Observatorio Social"),
    "raw/BCN/comunas": dict(access="public",
        description="Current comuna boundaries (BCN)",
        url="https://www.bcn.cl/siit/mapas_vectoriales/", instructions="BCN SIIT"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "public"], default="dev")
    ap.add_argument("--data-root", default=str(Path.home() / "Dropbox" / "_ddbb"))
    ap.add_argument("--inter-root", default=str(Path.home() / "Dropbox" / "eii"))
    a = ap.parse_args()

    a_raw = DATA_DIR / "A_raw"
    a_raw.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "C_final").mkdir(exist_ok=True)

    for key, meta in DATA_SOURCES.items():
        dst = a_raw / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if a.mode == "dev":
            src = Path(a.data_root) / key
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink():
                    dst.unlink()
                else:
                    print(f"  [skip] {dst} exists and is not a symlink")
                    continue
            if not src.exists():
                print(f"  [missing] {src}  ({meta['description']})")
                continue
            os.symlink(src, dst)
            print(f"  [link] {dst} -> {src}")
        else:
            dst.mkdir(parents=True, exist_ok=True)
            print(f"{key}: {meta['description']}\n    {meta['url']}\n    {meta['instructions']}")

    b_inter = DATA_DIR / "B_inter"
    if a.mode == "dev":
        src = Path(a.inter_root) / "B_inter"
        src.mkdir(parents=True, exist_ok=True)
        if b_inter.is_symlink():
            b_inter.unlink()
        if not b_inter.exists():
            os.symlink(src, b_inter)
        print(f"  [link] {b_inter} -> {src}")
    else:
        b_inter.mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
