#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 02: harmonize census microdata (1992, 2002, 2024) into
  B_inter/harmonized/census_{year}_hh.parquet   one row per private household
  B_inter/harmonized/census_{year}_per.parquet  one row per person (ethnicity, sex,
                                                age, mother's comuna, cohort, hh_id)

Usage: python -m scripts.data_processing.02_census_harmonize [--year 1992|2002|2024]

Harmonized dwelling block (same names in 01_casen_harmonize.py):
  dw_house dw_apt dw_precarious | wall_solid wall_wood wall_precarious |
  roof_good roof_zinc roof_precarious | floor_good floor_cement floor_dirt |
  water_public water_inside | sanit_sewer sanit_none | elec_public elec_none |
  n_bedrooms | tenure_owned tenure_rented | fuel_wood
Period-specific assets are kept with an `a_` prefix.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

from scripts.directories import census_1992, census_2002, census_2017, census_2024, data_inter, data_final
from scripts.tools.hh_features import build_hh_features
from scripts.tools.comuna_xwalk import to_parent1992, to_origen3

OUT = os.path.join(data_inter, 'harmonized')


def yesno(s, yes=1):
    """1 if s==yes, 0 if other valid code, NaN if missing (<0 or NaN)."""
    s = pd.to_numeric(s, errors='coerce')
    out = (s == yes).astype(float)
    out[s.isna() | (s < 0)] = np.nan
    return out


def isin(s, codes):
    s = pd.to_numeric(s, errors='coerce')
    out = s.isin(codes).astype(float)
    out[s.isna() | (s < 0)] = np.nan
    return out


# =============================================================================
# 2024
# =============================================================================
def harmonize_2024():
    P = os.path.join(census_2024, 'viv_hog_per_censo2024')
    print('[2024] loading persons ...')
    per = pd.read_parquet(os.path.join(P, 'personas_censo2024.parquet'), columns=[
        'id_vivienda', 'id_hogar', 'id_persona', 'region', 'comuna', 'comuna_bajo_umbral', 'area',
        'tipo_operativo', 'sexo', 'edad', 'edad_quinquenal', 'parentesco', 'p25_lug_nacimiento',
        'p25_lug_nacimiento_esp', 'p28_autoid_pueblo', 'p28_pueblo_pert', 'p37_alfabet',
        'escolaridad', 'sit_fuerza_trabajo', 'p40_cise_rec', 'p27_nacionalidad_rec'])
    per = per[per.tipo_operativo == 2].copy()
    per['hh_id'] = per.id_vivienda.astype('int64') * 100 + per.id_hogar.astype('int64')
    age = per.edad.astype(float)
    age[per.edad == -66] = per.edad_quinquenal[per.edad == -66].astype(float) + 2
    per['age'] = age
    per['female'] = (per.sexo == 2).astype(float)
    per['head'] = (per.parentesco == 1).astype(int)
    sch = pd.to_numeric(per.escolaridad, errors='coerce').astype(float)
    sch[(sch < 0) | (per.age < 15)] = np.nan
    per['schooling'] = sch
    emp = yesno(per.sit_fuerza_trabajo, 1)
    emp[per.age < 15] = np.nan
    per['employed'] = emp
    lit = yesno(per.p37_alfabet, 1); lit[per.age < 15] = np.nan
    per['literate'] = lit
    ind = yesno(per.p40_cise_rec, 1)
    ind[emp != 1] = np.nan
    per['indep'] = ind
    per['indigenous'] = yesno(per.p28_autoid_pueblo, 1)
    pert = pd.to_numeric(per.p28_pueblo_pert, errors='coerce')
    per['mapuche'] = ((per.indigenous == 1) & (pert == 1)).astype(float)
    per.loc[per.indigenous.isna(), 'mapuche'] = np.nan
    per['pueblo'] = np.where(per.indigenous == 1, pert.fillna(-99), 0)
    per['foreign_born'] = (per.p25_lug_nacimiento == 3).astype(float)
    per.loc[per.p25_lug_nacimiento == -99, 'foreign_born'] = np.nan
    # mother's comuna at birth, in 1992 parent-comuna vintage
    mc = np.where(per.p25_lug_nacimiento == 1, per.comuna,
                  np.where(per.p25_lug_nacimiento == 2, per.p25_lug_nacimiento_esp, np.nan)).astype(float)
    mc[(mc < 1000)] = np.nan            # -66 anonymized, -99, country codes
    per['mother_comuna'] = to_parent1992(mc)
    per['birth_year'] = 2024 - per.age
    per['cohort5'] = (per.birth_year // 5) * 5
    per['comuna92'] = to_parent1992(per.comuna)
    per['urban'] = (per.area == 1).astype(float)
    per['weight'] = 1.0

    print('[2024] household features ...')
    hh = build_hh_features(per[['hh_id', 'head', 'female', 'age', 'schooling', 'employed',
                                'literate', 'indep', 'indigenous']])
    geo = per.drop_duplicates('hh_id')[['hh_id', 'id_vivienda', 'id_hogar', 'region', 'comuna', 'comuna92', 'area', 'urban', 'comuna_bajo_umbral']]
    hh = hh.merge(geo, on='hh_id', how='left')

    print('[2024] dwellings and households ...')
    viv = pd.read_parquet(os.path.join(P, 'viviendas_censo2024.parquet'), columns=[
        'id_vivienda', 'p2_tipo_vivienda', 'p4a_mat_paredes', 'p4b_mat_techo', 'p4c_mat_piso',
        'p5_num_dormitorios', 'p6_fuente_agua', 'p7_distrib_agua', 'p8_serv_hig', 'p9_fuente_elect',
        'p11a_num_personas', 'indice_hacinamiento'])
    hog = pd.read_parquet(os.path.join(P, 'hogares_censo2024.parquet'), columns=[
        'id_vivienda', 'id_hogar', 'p12_tenencia_viv', 'p13_comb_cocina', 'p14_comb_calefaccion',
        'p15a_serv_tel_movil', 'p15b_serv_compu', 'p15c_serv_tablet', 'p15d_serv_internet_fija',
        'p15e_serv_internet_movil', 'p15f_serv_internet_satelital'])
    d = pd.DataFrame({'id_vivienda': viv.id_vivienda})
    t = pd.to_numeric(viv.p2_tipo_vivienda, errors='coerce')
    d['dw_house'] = isin(t, [1]); d['dw_apt'] = isin(t, [3]); d['dw_precarious'] = isin(t, [5, 6, 7, 8, 9])
    w = viv.p4a_mat_paredes
    d['wall_solid'] = isin(w, [1, 2]); d['wall_wood'] = isin(w, [3]); d['wall_precarious'] = isin(w, [4, 5, 6])
    r = viv.p4b_mat_techo
    d['roof_good'] = isin(r, [1, 2]); d['roof_zinc'] = isin(r, [3, 4]); d['roof_precarious'] = isin(r, [5, 6, 7, 8])
    f = viv.p4c_mat_piso
    d['floor_good'] = isin(f, [1]); d['floor_cement'] = isin(f, [2, 3, 4]); d['floor_dirt'] = isin(f, [5])
    d['water_public'] = yesno(viv.p6_fuente_agua, 1)
    d['water_inside'] = yesno(viv.p7_distrib_agua, 1)
    d['sanit_sewer'] = isin(viv.p8_serv_hig, [1, 2]); d['sanit_none'] = isin(viv.p8_serv_hig, [9])
    d['elec_public'] = yesno(viv.p9_fuente_elect, 1); d['elec_none'] = yesno(viv.p9_fuente_elect, 6)
    nb = pd.to_numeric(viv.p5_num_dormitorios, errors='coerce').astype(float)
    nb[nb < 0] = np.nan
    d['n_bedrooms'] = nb.clip(upper=6)
    d['hacinamiento'] = isin(viv.indice_hacinamiento, [2, 3])
    h = pd.DataFrame({'id_vivienda': hog.id_vivienda, 'id_hogar': hog.id_hogar})
    h['tenure_owned'] = isin(hog.p12_tenencia_viv, [1, 2, 9]); h['tenure_rented'] = isin(hog.p12_tenencia_viv, [3, 4])
    h['fuel_wood'] = isin(hog.p13_comb_cocina, [3, 4, 5])
    h['heat_wood'] = isin(hog.p14_comb_calefaccion, [3, 4, 5])
    h['a_cellphone'] = yesno(hog.p15a_serv_tel_movil, 1)
    h['a_computer'] = yesno(hog.p15b_serv_compu, 1)
    h['a_tablet'] = yesno(hog.p15c_serv_tablet, 1)
    h['a_internet_fixed'] = yesno(hog.p15d_serv_internet_fija, 1)
    h['a_internet_mobile'] = yesno(hog.p15e_serv_internet_movil, 1)
    h['a_internet_sat'] = yesno(hog.p15f_serv_internet_satelital, 1)
    hh = hh.merge(d, on='id_vivienda', how='left').merge(h, on=['id_vivienda', 'id_hogar'], how='left')
    hh['year'] = 2024
    hh.to_parquet(os.path.join(OUT, 'census_2024_hh.parquet'), index=False)
    per[['hh_id', 'id_persona', 'region', 'comuna', 'comuna92', 'urban', 'comuna_bajo_umbral', 'female', 'age',
         'birth_year', 'cohort5', 'head', 'schooling', 'employed', 'indigenous', 'mapuche', 'pueblo',
         'foreign_born', 'mother_comuna', 'weight']].to_parquet(os.path.join(OUT, 'census_2024_per.parquet'), index=False)
    print('[2024] done: hh', len(hh), 'persons', len(per))
    print(hh.describe().T.to_string())


# =============================================================================
# 1992
# =============================================================================
def schooling_1992(tipo, curso):
    """Years of schooling from 1992 census 'Tipo_educacion' x 'Curso'."""
    tipo = tipo.astype(float); curso = curso.astype(float)
    y = pd.Series(np.nan, index=tipo.index)
    y[tipo == 0] = 0                                    # never attended
    y[tipo == 1] = 0                                    # kinder
    y[tipo == 2] = curso[tipo == 2]                      # básica/primaria (1-8)
    old_hum = tipo == 4                                  # humanidades (old system: 6 primaria + 6)
    y[old_hum] = 6 + curso[old_hum]
    sec = tipo.isin([3, 5, 6, 7, 8, 9, 10, 11])           # media and old technical/normal
    y[sec] = 8 + curso[sec]
    ter = tipo.isin([12, 13, 14])
    y[ter] = 12 + curso[ter]
    return y.clip(upper=21)


def harmonize_1992():
    M = os.path.join(census_1992, 'microdata')
    print('[1992] loading ...')
    per = pd.read_csv(os.path.join(M, 'PERSONA.csv'), usecols=[
        'Portafolio', 'Vivienda', 'Hogar', 'Persona', 'Parentesco', 'Sexo', 'Edad',
        'Comuna_Madre_Origen3', 'Tipo_educacion', 'Curso', 'Leer', 'Situacion_empleo',
        'Categoria_ocupacional', 'Cultura'])
    hog = pd.read_csv(os.path.join(M, 'HOGAR.csv'), index_col=0)
    viv = pd.read_csv(os.path.join(M, 'VIVIENDA.csv'), index_col=0)
    pf = pd.read_csv(os.path.join(M, 'PORTAFOLIOS.csv'), index_col=0)[['Portafolios', 'Comuna', 'Area']]
    print('[1992] max Vivienda', per.Vivienda.max(), 'max Hogar', per.Hogar.max())
    assert per.Vivienda.max() < 1_000_000 and per.Hogar.max() < 100
    key = lambda df: (df.Portafolio.astype('int64') * 1_000_000 + df.Vivienda.astype('int64')) * 100 + df.Hogar.astype('int64')
    per['hh_id'] = key(per); hog['hh_id'] = key(hog)
    viv['viv_id'] = viv.Portafolio.astype('int64') * 1_000_000 + viv.Vivienda.astype('int64')
    per['viv_id'] = per.Portafolio.astype('int64') * 1_000_000 + per.Vivienda.astype('int64')
    # private, occupied dwellings only (collective: Tipo_vivienda 8-11; parentesco 12 = member of collective hh)
    viv = viv[(viv.Tipo_vivienda <= 7) & (viv.Cond_ocupacion == 1)]
    per = per[per.viv_id.isin(viv.viv_id) & (per.Parentesco != 12)].copy()

    per['age'] = per.Edad.astype(float)
    per['female'] = (per.Sexo == 2).astype(float)
    per['head'] = (per.Parentesco == 1).astype(int)
    sch = schooling_1992(per.Tipo_educacion, per.Curso)
    sch[per.age < 15] = np.nan
    per['schooling'] = sch
    emp = per.Situacion_empleo.isin([1, 2, 3]).astype(float)
    emp[per.age < 14] = np.nan                      # asked to 14+; keep 14 as adult-ish? use 15+ for comparability
    emp[per.age < 15] = np.nan
    per['employed'] = emp
    lit = yesno(per.Leer, 1); lit[(per.Leer == 0) | (per.age < 15)] = np.nan
    per['literate'] = lit
    ind = per.Categoria_ocupacional.isin([1, 2]).astype(float); ind[emp != 1] = np.nan
    per['indep'] = ind
    # ethnicity: asked to 14+ ; 1 Mapuche 2 Aymara 3 Rapanui 4 none ; 0 not asked
    cu = per.Cultura
    per['indigenous'] = np.where(cu.isin([1, 2, 3]), 1.0, np.where(cu == 4, 0.0, np.nan))
    per['mapuche'] = np.where(cu == 1, 1.0, np.where(cu.isin([2, 3, 4]), 0.0, np.nan))
    per['pueblo'] = np.where(cu.isin([1, 2, 3]), cu, np.where(cu == 4, 0, -99))
    mc = per.Comuna_Madre_Origen3.astype(float)
    per['foreign_born'] = ((mc < 1000) & (mc > 0)).astype(float)
    mc[(mc < 1000) | (mc >= 99999)] = np.nan
    per['mother_comuna'] = to_parent1992(mc)
    per['birth_year'] = 1992 - per.age
    per['cohort5'] = (per.birth_year // 5) * 5
    per = per.merge(pf.rename(columns={'Portafolios': 'Portafolio'}), on='Portafolio', how='left')
    per['comuna'] = per.Comuna.astype(float)
    per['comuna92'] = to_parent1992(per.comuna)
    per['region'] = (per.comuna // 1000).astype(float)
    per['urban'] = (per.Area == 1).astype(float)
    per['weight'] = 1.0
    per['id_persona'] = per.Persona

    print('[1992] household features ...')
    hh = build_hh_features(per[['hh_id', 'head', 'female', 'age', 'schooling', 'employed',
                                'literate', 'indep', 'indigenous']])
    geo = per.drop_duplicates('hh_id')[['hh_id', 'viv_id', 'region', 'comuna', 'comuna92', 'urban']]
    hh = hh.merge(geo, on='hh_id', how='left')
    d = pd.DataFrame({'viv_id': viv.viv_id})
    t = viv.Tipo_vivienda
    d['dw_house'] = isin(t, [1]); d['dw_apt'] = isin(t, [2]); d['dw_precarious'] = isin(t, [3, 4, 5, 6, 7])
    d['wall_solid'] = isin(viv.Paredes, [1]); d['wall_wood'] = isin(viv.Paredes, [2]); d['wall_precarious'] = isin(viv.Paredes, [3, 4, 5, 6])
    d['roof_good'] = isin(viv.Techo, [2, 4, 5]); d['roof_zinc'] = isin(viv.Techo, [1, 3]); d['roof_precarious'] = isin(viv.Techo, [6, 7, 8])
    d['floor_good'] = isin(viv.Piso, [1, 2, 3, 4]); d['floor_cement'] = isin(viv.Piso, [5, 6, 8]); d['floor_dirt'] = isin(viv.Piso, [7])
    d['water_public'] = yesno(viv.Origen_agua, 1); d['water_inside'] = yesno(viv.Agua_cañeria, 1)
    d['sanit_sewer'] = yesno(viv.Servicio_Higienico, 1); d['sanit_none'] = yesno(viv.Servicio_Higienico, 5)
    d['elec_public'] = yesno(viv.Alumbrado_electrico, 1); d['elec_none'] = yesno(viv.Alumbrado_electrico, 4)
    d['tenure_owned'] = isin(viv.Cond_tenencia, [1, 2]); d['tenure_rented'] = isin(viv.Cond_tenencia, [3])
    d['n_rooms'] = viv.Total_piezas.astype(float).clip(upper=10)
    d['a_shower'] = yesno(viv.Ducha, 1)
    for c in d.columns[1:]:
        d.loc[t == 0, c] = np.nan
    h = pd.DataFrame({'hh_id': hog.hh_id})
    nb = hog.Pieza_dormir.astype(float); nb[nb == 99] = np.nan
    h['n_bedrooms'] = nb.clip(upper=6)
    h['fuel_wood'] = yesno(hog.Combustible, 3)
    for a, c in [('a_radio', 'Radio'), ('a_tv_color', 'TV_Color'), ('a_video', 'Videograbador'), ('a_hifi', 'Minicomponente'),
                 ('a_washer', 'Lavadora'), ('a_washer_auto', 'Lav_prog'), ('a_fridge', 'Refrigerador'), ('a_microwave', 'Microondas'),
                 ('a_cellphone', 'Celular'), ('a_phone', 'Telefono'), ('a_car', 'Auto_Part'), ('a_pickup', 'Camioneta_Part')]:
        h[a] = yesno(hog[c], 1)
    h['puntaje_cas'] = hog.Puntaje.astype(float)
    hh = hh.merge(d, on='viv_id', how='left').merge(h, on='hh_id', how='left')
    hh['year'] = 1992
    hh.to_parquet(os.path.join(OUT, 'census_1992_hh.parquet'), index=False)
    per[['hh_id', 'id_persona', 'region', 'comuna', 'comuna92', 'urban', 'female', 'age', 'birth_year', 'cohort5',
         'head', 'schooling', 'employed', 'indigenous', 'mapuche', 'pueblo', 'foreign_born', 'mother_comuna',
         'weight']].to_parquet(os.path.join(OUT, 'census_1992_per.parquet'), index=False)
    print('[1992] done: hh', len(hh), 'persons', len(per))
    print(hh.describe().T.to_string())


def harmonize_2017():
    print('[2017] loading persons (2.6 GB csv) ...')
    per = pd.read_csv(os.path.join(census_2017, 'Microdato_Censo2017-Personas.csv'), sep=';', usecols=[
        'REGION', 'COMUNA', 'AREA', 'DC', 'ZC_LOC', 'ID_ZONA_LOC', 'NVIV', 'NHOGAR', 'PERSONAN', 'P07', 'P08', 'P09',
        'P12', 'P12COMUNA', 'P15', 'P16', 'P16A', 'P17', 'ESCOLARIDAD'], dtype='int32')
    per = per[~per.P07.isin([17, 18, 19])].copy()
    per['viv_id'] = per.ID_ZONA_LOC.astype('int64') * 1_000_000 + per.NVIV.astype('int64')
    per['hh_id'] = per.viv_id * 100 + per.NHOGAR.astype('int64')
    per['age'] = per.P09.astype(float); per.loc[per.age > 120, 'age'] = np.nan
    per['female'] = (per.P08 == 2).astype(float)
    per['head'] = (per.P07 == 1).astype(int)
    sch = per.ESCOLARIDAD.astype(float); sch[(sch >= 90) | (per.age < 15)] = np.nan
    per['schooling'] = sch
    emp = isin(per.P17, [1, 2, 3]); emp[(per.P17 >= 98) | (per.age < 15)] = np.nan
    per['employed'] = emp
    per['literate'] = np.nan; per['indep'] = np.nan
    per['indigenous'] = np.where(per.P16 == 1, 1.0, np.where(per.P16 == 2, 0.0, np.nan))
    per['mapuche'] = np.where((per.P16 == 1) & (per.P16A == 1), 1.0, np.where(per.P16.isin([1, 2]), 0.0, np.nan))
    pue = per.P16A.astype(float).replace({10: 12}); pue[pue >= 98] = -99
    per['pueblo'] = np.where(per.P16 == 1, pue, np.where(per.P16 == 2, 0, -99))
    per['foreign_born'] = np.where(per.P12.between(3, 8), 1.0, np.where(per.P12.isin([1, 2]), 0.0, np.nan))
    mc = np.where(per.P12 == 1, per.COMUNA, np.where(per.P12 == 2, per.P12COMUNA, np.nan)).astype(float)
    mc[(mc < 1000) | (mc >= 90000)] = np.nan
    per['mother_comuna'] = to_parent1992(mc)
    per['birth_year'] = 2017 - per.age
    per['cohort5'] = (per.birth_year // 5) * 5
    per['comuna'] = per.COMUNA.astype(float); per['comuna92'] = to_parent1992(per.comuna)
    per['region'] = per.REGION.astype(float); per['urban'] = (per.AREA == 1).astype(float); per['weight'] = 1.0
    per['id_persona'] = per.PERSONAN; per['zone_id'] = per.ID_ZONA_LOC; per['distrito'] = per.DC
    print('[2017] household features ...')
    hh = build_hh_features(per[['hh_id', 'head', 'female', 'age', 'schooling', 'employed', 'literate', 'indep', 'indigenous']])
    geo = per.drop_duplicates('hh_id')[['hh_id', 'viv_id', 'region', 'comuna', 'comuna92', 'urban', 'zone_id', 'distrito']]
    hh = hh.merge(geo, on='hh_id', how='left')
    viv = pd.read_csv(os.path.join(census_2017, 'Microdato_Censo2017-Viviendas.csv'), sep=';', usecols=[
        'ID_ZONA_LOC', 'NVIV', 'P01', 'P02', 'P03A', 'P03B', 'P03C', 'P04', 'P05'], dtype='int32')
    viv = viv[(viv.P02 == 1) & (viv.P01 <= 7)].copy()
    viv['viv_id'] = viv.ID_ZONA_LOC.astype('int64') * 1_000_000 + viv.NVIV.astype('int64')
    d = pd.DataFrame({'viv_id': viv.viv_id})
    def cc(s):   # 98/99 -> NaN
        s = s.astype(float); s[s >= 98] = np.nan; return s
    t = cc(viv.P01); d['dw_house'] = isin(t, [1]); d['dw_apt'] = isin(t, [2]); d['dw_precarious'] = isin(t, [3, 4, 5, 6, 7])
    w = cc(viv.P03A); d['wall_solid'] = isin(w, [1, 2]); d['wall_wood'] = isin(w, [3]); d['wall_precarious'] = isin(w, [4, 5, 6])
    r = cc(viv.P03B); d['roof_good'] = isin(r, [1, 2]); d['roof_zinc'] = isin(r, [3]); d['roof_precarious'] = isin(r, [4, 5, 6, 7])
    f = cc(viv.P03C); d['floor_good'] = isin(f, [1]); d['floor_cement'] = isin(f, [2, 3, 4]); d['floor_dirt'] = isin(f, [5])
    d['n_bedrooms'] = cc(viv.P04).clip(upper=6)
    d['water_public'] = yesno(cc(viv.P05), 1)
    hh = hh.merge(d, on='viv_id', how='left')
    hh['year'] = 2017
    hh.to_parquet(os.path.join(OUT, 'census_2017_hh.parquet'), index=False)
    per[['hh_id', 'id_persona', 'region', 'comuna', 'comuna92', 'urban', 'zone_id', 'distrito', 'female', 'age', 'birth_year',
         'cohort5', 'head', 'schooling', 'employed', 'indigenous', 'mapuche', 'pueblo', 'foreign_born', 'mother_comuna',
         'weight']].to_parquet(os.path.join(OUT, 'census_2017_per.parquet'), index=False)
    print('[2017] done: hh', len(hh), 'persons', len(per)); print(hh.describe().T.to_string())


def harmonize_2002():
    """2002 census: persons + households + dwellings. Comuna of residence comes from the reconstruction of the
    enumeration-area numbering (scripts/tools/reconstruct_2002_geography.py); the file has no urban/rural code,
    so the urban status of each household is imputed from a logit fitted on CASEN 2003 (dwelling, services,
    assets, region) and stored as a probability in `urban` (drawn per replication in script 04)."""
    M = os.path.join(census_2002, 'microdata')
    geo = pd.read_parquet(os.path.join(data_final, 'census2002_portafolio_comuna.parquet'))
    print('[2002] loading persons ...')
    per = pd.read_stata(os.path.join(M, 'persona.dta'), convert_categoricals=False,
                        columns=['portafolio', 'vn', 'hn', 'pn', 'p17', 'p18', 'p19', 'p21', 'p22a', 'p22b', 'p25', 'p26a', 'p26b', 'p29', 'p30'])
    per = per[per.p17 != 15].copy()
    per['viv_id'] = per.portafolio.astype('int64') * 10000 + per.vn.astype('int64')
    per['hh_id'] = per.viv_id * 100 + per.hn.astype('int64')
    per = per.merge(geo, on='portafolio', how='left')
    per['age'] = per.p19.astype(float); per.loc[per.age > 120, 'age'] = np.nan
    per['female'] = (per.p18 == 2).astype(float)
    per['head'] = (per.p17 == 1).astype(int)
    sch = schooling_2002(per.p26a, per.p26b); sch[per.age < 15] = np.nan
    per['schooling'] = sch
    emp = per.p29.isin([1, 2, 4]).astype(float); emp[(per.age < 15) | (per.p29 == 0)] = np.nan
    per['employed'] = emp
    lit = yesno(per.p25, 1); lit[(per.age < 15) | (per.p25 == 0)] = np.nan
    per['literate'] = lit
    ind = per.p30.isin([3, 4]).astype(float); ind[emp != 1] = np.nan
    per['indep'] = ind
    p21 = per.p21.astype(float)
    per['indigenous'] = np.where(p21.between(1, 8), 1.0, np.where(p21 == 9, 0.0, np.nan))
    per['mapuche'] = np.where(p21 == 5, 1.0, np.where(p21.between(1, 9), 0.0, np.nan))
    P21 = {1: 8, 2: 4, 3: 2, 4: 6, 5: 1, 6: 5, 7: 3, 8: 9}
    per['pueblo'] = np.where(p21 == 9, 0, p21.map(P21).fillna(-99))
    per['foreign_born'] = np.where(per.p22a == 3, 1.0, np.where(per.p22a.isin([1, 2]), 0.0, np.nan))
    mc = np.where(per.p22a == 1, per.comuna, np.where(per.p22a == 2, per.p22b, np.nan)).astype(float)
    mc[(mc < 1000) | (mc >= 90000)] = np.nan
    per['mother_comuna'] = to_parent1992(mc)
    per['birth_year'] = 2002 - per.age
    per['cohort5'] = (per.birth_year // 5) * 5
    per['region'] = (per.comuna_old // 1000).astype(float)
    per['weight'] = 1.0; per['id_persona'] = per.pn

    print('[2002] dwellings and households ...')
    viv = pd.read_stata(os.path.join(M, 'vivienda.dta'), convert_categoricals=False)
    viv = viv[(viv.v2 == 1) & (viv.v1 <= 8)].copy()
    viv['viv_id'] = viv.portafolio.astype('int64') * 10000 + viv.vn_renumerado.astype('int64')   # persons/households carry the renumbered dwelling id
    hog = pd.read_stata(os.path.join(M, 'hogares.dta'), convert_categoricals=False)
    hog['hh_id'] = (hog.portafolio.astype('int64') * 10000 + hog.vn.astype('int64')) * 100 + hog.hn.astype('int64')
    per = per[per.viv_id.isin(viv.viv_id)].copy()
    d = pd.DataFrame({'viv_id': viv.viv_id})
    t = viv.v1; d['dw_house'] = yesno(t, 1); d['dw_apt'] = yesno(t, 2); d['dw_precarious'] = isin(t, [3, 4, 5, 6, 7, 8])
    d['wall_solid'] = isin(viv.v4a, [1, 2, 3]); d['wall_wood'] = yesno(viv.v4a, 4); d['wall_precarious'] = isin(viv.v4a, [5, 6, 7])
    d['roof_good'] = isin(viv.v4b, [1, 2, 3]); d['roof_zinc'] = isin(viv.v4b, [4, 5, 6]); d['roof_precarious'] = isin(viv.v4b, [7, 8, 9])
    d['floor_good'] = isin(viv.v4c, [1, 2, 3, 4, 6]); d['floor_cement'] = isin(viv.v4c, [5, 7, 8]); d['floor_dirt'] = yesno(viv.v4c, 9)
    d['water_public'] = yesno(viv.v6, 1); d['water_inside'] = yesno(viv.v7, 1)
    d['sanit_sewer'] = isin(viv.v8, [1, 2]); d['sanit_none'] = yesno(viv.v8, 6)
    d['elec_public'] = yesno(viv.v5, 1); d['elec_none'] = yesno(viv.v5, 4)
    d['tenure_owned'] = isin(viv.v3, [1, 2]); d['tenure_rented'] = yesno(viv.v3, 3)
    nr = viv.v10a.astype(float); nr[nr > 20] = np.nan; d['n_rooms'] = nr.clip(upper=10)
    d['a_shower'] = (viv.v9.astype(float) >= 1).astype(float)
    for c in d.columns[1:]:
        d.loc[viv[['v4a', 'v4b', 'v4c']].eq(0).all(axis=1), c] = np.nan
    h = pd.DataFrame({'hh_id': hog.hh_id})
    nb = hog.h13.astype(float); nb[nb > 20] = np.nan; h['n_bedrooms'] = nb.clip(upper=6)
    h['fuel_wood'] = isin(hog.h12, [4, 5])
    for a, c in [('a_tv_color', 'h15_2'), ('a_video', 'h15_3'), ('a_cable_tv', 'h15_4'), ('a_hifi', 'h15_5'), ('a_washer', 'h15_6'), ('a_fridge', 'h15_8'),
                 ('a_microwave', 'h15_10'), ('a_calefont', 'h15_12'), ('a_cellphone', 'h15_13'), ('a_phone', 'h15_14'), ('a_computer', 'h15_15'), ('a_internet', 'h15_16')]:
        h[a] = yesno(hog[c], 1)
    h['a_car'] = yesno(hog.h16_4, 1)

    print('[2002] household features ...')
    hh = build_hh_features(per[['hh_id', 'head', 'female', 'age', 'schooling', 'employed', 'literate', 'indep', 'indigenous']])
    geo_hh = per.drop_duplicates('hh_id')[['hh_id', 'viv_id', 'portafolio', 'region', 'comuna', 'comuna92', 'comuna_old']]
    hh = hh.merge(geo_hh, on='hh_id', how='left').merge(d, on='viv_id', how='left').merge(h, on='hh_id', how='left')
    hh['region13'] = hh.region
    # urban status imputed from CASEN 2003
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    c = pd.read_parquet(os.path.join(OUT, 'casen_2003_hh.parquet'))
    uvars = ['dw_apt', 'dw_precarious', 'wall_solid', 'wall_precarious', 'roof_good', 'floor_dirt', 'water_public', 'water_inside', 'sanit_sewer',
             'sanit_none', 'elec_public', 'n_bedrooms', 'tenure_rented', 'a_phone', 'a_cable_tv', 'a_computer', 'a_cellphone', 'head_schooling',
             'mean_schooling_adults', 'n_people', 'frac_employed_adults', 'head_indep']
    def U(df):
        X = df[uvars].astype(float).copy()
        for v in uvars: X[v] = X[v].fillna(c[v].median())
        for r_ in range(2, 14): X[f'reg_{r_}'] = (df.region13 == r_).astype(float)
        return X
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(U(c), c.urban.to_numpy(), logisticregression__sample_weight=c.weight.to_numpy())
    hh['urban'] = lg.predict_proba(U(hh))[:, 1]
    hh['urban_imputed'] = 1.0
    print(f"[2002] urban imputed: mean P(urban) {hh.urban.mean():.3f} (CASEN 2003 weighted share {np.average(c.urban, weights=c.weight):.3f}); in-sample AUC {__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(c.urban, lg.predict_proba(U(c))[:, 1]):.3f}")
    per = per.merge(hh[['hh_id', 'urban']], on='hh_id', how='left')
    hh['year'] = 2002
    hh.to_parquet(os.path.join(OUT, 'census_2002_hh.parquet'), index=False)
    per[['hh_id', 'id_persona', 'region', 'comuna', 'comuna92', 'urban', 'female', 'age', 'birth_year', 'cohort5', 'head', 'schooling', 'employed',
         'indigenous', 'mapuche', 'pueblo', 'foreign_born', 'mother_comuna', 'weight']].to_parquet(os.path.join(OUT, 'census_2002_per.parquet'), index=False)
    print('[2002] done: hh', len(hh), 'persons', len(per), '| mother comuna resolved', per.mother_comuna.notna().mean().round(3))
    print(hh.describe().T.to_string())


def schooling_2002(nivel, curso):
    nivel = nivel.astype(float); curso = curso.astype(float).fillna(0)
    y = pd.Series(np.nan, index=nivel.index)
    y[nivel.isin([1, 2, 3])] = 0
    y[nivel == 4] = curso[nivel == 4]
    y[nivel == 6] = 6 + curso[nivel == 6]                       # humanidades (old system)
    sec = nivel.isin([5, 7, 8, 9, 10, 11, 12]); y[sec] = 8 + curso[sec]
    ter = nivel.isin([13, 14, 15]); y[ter] = 12 + curso[ter]
    return y.clip(upper=21)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, nargs='*', default=[1992, 2002, 2017, 2024])
    a = ap.parse_args()
    for y in a.year:
        {1992: harmonize_1992, 2002: harmonize_2002, 2017: harmonize_2017, 2024: harmonize_2024}[y]()
