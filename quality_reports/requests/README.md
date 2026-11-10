# Information requests (Ley 20.285, Portal de Transparencia)

| Receipt | Filed | Organism | Request | Due |
|---|---|---|---|---|
| `Recibo-SAI-AH007T0014632.pdf` | 2026-11-07 | INE | 2002 census microdata **with the portafolio (enumeration-area) table** that locates each observation geographically | 2026-12-07 (+10 working days possible) |
| `Recibo-SAI-AH007T0014633.pdf` | 2026-11-07 | INE | Crosswalk `id_vivienda` (public 2024 microdata) -> census geography (ID_ZONA / ID_ENTIDAD / ID_LOCALIDAD; if possible MANZENT, ID_DISTRITO; fallback distrito) | 2026-12-07 (+10 working days possible) |

Text of the 2024 request: `2026-11-07_solicitud_INE_geografia_censo2024.md`.

While the answers are pending: the 2002 census stays out of the paper (the reconstruction of comuna blocks from the
enumeration-area ordering, `scripts/tools/reconstruct_2002_geography.py`, failed validation against CASEN 2003 comuna
means, see `scripts/tools/validate_2002_geography.py`; decision 2026-11-07: wait for INE). The 2024 zone assignment
will go through the published zone aggregates (naive Bayes / matching model, validated on 2017) if the crosswalk is denied.

The receipts carry the requester's name and e-mail (no RUT); they are kept here as the record of what was asked and when.
