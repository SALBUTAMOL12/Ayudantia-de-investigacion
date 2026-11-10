#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct the comuna of residence of every 2002 census enumeration area (`portafolio`).

The 2002 microdata carry only a sequential `portafolio` id. The ids are numbered in geographic order
(as the 1992 PORTAFOLIOS table shows for 1992, and as the mothers' comunas of movers show for 2002):
region, then comuna in the official code order of the 2002 partition. Given the 2002 census population of
every comuna (scraped from BCN's reportes comunales, "Población total 2002", INE), the sorted sequence
of enumeration areas is cut into comuna blocks by cumulative person counts, at enumeration-area
boundaries. Errors are confined to block edges. Validation: comuna-level means of dwelling services in
the reconstructed census vs CASEN 2003 by comuna (both in the 2002-vintage partition).

Inputs : scratchpad/bcn/bcn_pop2002.json (current codes -> 2002 population), C_final/casen2003_comuna_labels.csv
         (old 2002 codes -> names), raw 2002 persona.dta
Output : C_final/census2002_portafolio_comuna.parquet (portafolio, comuna_old, comuna (Origen3 code), comuna92)
"""
import os, re, json, unicodedata
import numpy as np
import pandas as pd
from scripts.directories import census_2002, data_final
from scripts.tools.comuna_xwalk import to_parent1992, to_origen3

BCN_JSON = '/private/tmp/claude-501/-Users-fj-research-eii/cdc12122-0ace-4412-b470-cbdaf7d49466/scratchpad/bcn/bcn_pop2002.json'
POST2002_NAMES = {'ALTO HOSPICIO': 'IQUIQUE', 'HUALPEN': 'TALCAHUANO', 'CHOLCHOL': 'NUEVA IMPERIAL', 'ALTO BIOBIO': 'SANTA BARBARA'}   # created 2004 -> parent
MISSING_POP = {'ANTARTICA': 130}   # 2002 census population not on the BCN page
FIX = {'PAIHUANO': 'PAIGUANO', 'DONIHUE': 'DONIGUE', 'TREHUACO': 'TREGUACO', 'CUNCO': 'CIUNCO', 'TEODORO SCHIMDT': 'TEODORO SCHMIDT',
       'IMPERIAL': 'NUEVA IMPERIAL', 'COYHAIQUE': 'COIHAIQUE', 'PUERTO AISEN': 'AISEN', 'O HIGGINS': 'OHIGGINS', 'CABO DE ORNOS NAVARINO': 'CABO DE HORNO',
       'ALTO BIO BIO': 'ALTO BIOBIO', 'AYSEN': 'AISEN', 'HUALPEN': 'HUALPEN'}


def norm(s):
    return re.sub(r'[^A-Z ]', '', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().upper()).strip()


def main():
    bcn = json.load(open(BCN_JSON))
    bcn = pd.DataFrame([{'code_new': int(k), 'name': norm(v['name']) if v['name'] else None, 'pop2002': v['pop2002']} for k, v in bcn.items()])
    bcn['code_new'] = to_origen3(bcn.code_new).astype(int)
    old = pd.read_csv(os.path.join(data_final, 'casen2003_comuna_labels.csv'))
    old['name_m'] = old.name.map(lambda x: FIX.get(x, x))
    new92 = pd.read_csv(os.path.join(census_2002, '..', 'census_1992', 'microdata', 'COMUNAS.csv'), index_col=0)
    new92['name_m'] = new92.Nombre.map(norm)
    old = old.merge(new92[['Codigo', 'name_m']], on='name_m', how='left').rename(columns={'Codigo': 'code_new'})
    miss = old[old.code_new.isna()]
    assert len(miss) == 0, f'unmatched old comunas: {miss.to_string()}'
    old['code_new'] = old.code_new.astype(int)
    old = old.merge(bcn[['code_new', 'pop2002']], on='code_new', how='left')
    # comunas created after the 2002 census: fold into the parent (the BCN 2002 figure of the parent includes them)
    name2old = dict(zip(old.name_m, old.old))
    parent = {name2old[k]: name2old[v] for k, v in POST2002_NAMES.items()}
    old['old_block'] = old.old.map(lambda c: parent.get(c, c))
    for nm, v in MISSING_POP.items():
        old.loc[(old.name_m == nm) & old.pop2002.isna(), 'pop2002'] = v
    blocks = old[old.old == old.old_block].copy().sort_values('old')
    assert blocks.pop2002.notna().all(), blocks[blocks.pop2002.isna()]
    print(f'{len(blocks)} comuna blocks, total BCN 2002 population {int(blocks.pop2002.sum()):,}')
    # person counts per portafolio, in id order
    per = pd.read_stata(os.path.join(census_2002, 'microdata', 'persona.dta'), convert_categoricals=False, columns=['portafolio'])
    cnt = per.groupby('portafolio').size().sort_index()
    print(f'census persons {int(cnt.sum()):,}; enumeration areas {len(cnt):,}')
    scale = cnt.sum() / blocks.pop2002.sum()
    target = (blocks.pop2002 * scale).cumsum().to_numpy()
    cum = cnt.cumsum().to_numpy()
    # cut: block k gets the enumeration areas whose cumulative count crosses target[k-1]..target[k]
    edges = np.searchsorted(cum, target, side='left')      # index of the first area beyond each block
    edges[-1] = len(cum)
    assign = np.repeat(np.arange(len(blocks)), np.diff(np.concatenate([[0], edges])))
    out = pd.DataFrame({'portafolio': cnt.index.to_numpy(), 'block': assign})
    out = out.merge(blocks.reset_index(drop=True).reset_index().rename(columns={'index': 'block'})[['block', 'old', 'code_new', 'pop2002']], on='block')
    out['comuna_old'] = out.old; out['comuna'] = out.code_new; out['comuna92'] = to_parent1992(out.comuna)
    got = out.merge(cnt.rename('n'), left_on='portafolio', right_index=True).groupby('old').n.sum()
    chk = blocks.set_index('old').pop2002 * scale
    print('relative error of block sizes: mean |err| %.4f, max %.4f' % ((got / chk - 1).abs().mean(), (got / chk - 1).abs().max()))
    out[['portafolio', 'comuna_old', 'comuna', 'comuna92']].to_parquet(os.path.join(data_final, 'census2002_portafolio_comuna.parquet'), index=False)
    return out


if __name__ == '__main__':
    main()
