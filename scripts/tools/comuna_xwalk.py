#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comuna code harmonization to the 1992-census vintage used in the 1992 microdata
("Origen 3" codes: 2007 regional reform codes, Ñuble still coded under 84xx).

Two operations:
  to_origen3(code)  : map 2018+ Ñuble codes (16xxx) back to their 84xx codes.
  to_parent1992(code): additionally merge comunas created after 1992 into the comuna
                       they were carved from, so that "comuna where the mother lived
                       at birth" is a stable partition across 1992, 2002 and 2024.
"""
import numpy as np
import pandas as pd

NUBLE_TO_84 = {
    16101: 8401, 16102: 8402, 16103: 8406, 16104: 8407, 16105: 8410, 16106: 8411,
    16107: 8413, 16108: 8418, 16109: 8421, 16201: 8414, 16202: 8403, 16203: 8404,
    16204: 8408, 16205: 8412, 16206: 8415, 16207: 8420, 16301: 8416, 16302: 8405,
    16303: 8409, 16304: 8417, 16305: 8419,
    # Marga Marga province (2009): 1992-vintage codes
    5801: 5106, 5802: 5505, 5803: 5507, 5804: 5108,
}

# comunas created after the 1992 census -> parent comuna (1992 boundaries)
POST1992_TO_PARENT = {
    1107: 1101,    # Alto Hospicio (2004) <- Iquique
    8112: 8110,    # Hualpén (2004) <- Talcahuano
    9121: 9111,    # Cholchol (2004) <- Nueva Imperial
    8314: 8311,    # Alto Biobío (2004) <- Santa Bárbara
    13604: 13605,  # Padre Hurtado (1994) <- Peñaflor
    8103: 8101,    # Chiguayante (1996) <- Concepción
    8108: 8101,    # San Pedro de la Paz (1995) <- Concepción
    5103: 5109,    # Concón (1995) <- Viña del Mar
    9112: 9101,    # Padre Las Casas (1995) <- Temuco
    7110: 7106,    # San Rafael (1994) <- Pelarco
}


def to_origen3(code):
    s = pd.Series(code).astype('float')
    return s.replace(NUBLE_TO_84).to_numpy()


def to_parent1992(code):
    s = pd.Series(code).astype('float')
    s = s.replace(NUBLE_TO_84)
    s = s.replace(POST1992_TO_PARENT)
    return s.to_numpy()
