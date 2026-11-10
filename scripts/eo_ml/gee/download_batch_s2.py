"""
---- DOWNLOAD IMAGERY FOR MULTIPLE BATCHES (SENTINEL-2 + S2CLOUDLESS) ----

"""

import ee
import sys


ee.Initialize(project="ee-felipeguerrero")

# Parámetros
CLOUD_PROB_THRESHOLD = 40
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
NDVI_BANDS = ["B8", "B4"]


def maskCloudsByProbability(sr_image, prob_image):
    cloud_mask = prob_image.lt(CLOUD_PROB_THRESHOLD)
    return sr_image.updateMask(cloud_mask)


def addNDVIProperty(region):
    def _addNDVI(img):
        ndvi = img.normalizedDifference(NDVI_BANDS).rename('ndvi')
        ndvi = ndvi.where(ndvi.lt(0), 0)
        avg_ndvi = ndvi.reduceRegion(ee.Reducer.mean(), region, 10).get('ndvi')
        return img.set({'NDVI': avg_ndvi})
    return _addNDVI


def TerrainCorrection(scale, band_names, smooth=5):

    def TerrainCorrection_child(img):
        deg2rad = 0.0174533
        foot = ee.Geometry(img.get('system:footprint'))
        area_foot = foot.area(1)
        is_empty = area_foot.eq(0)

        def do_correction(i):
            region = ee.Geometry(i.get('system:footprint'))
            sub_img = i.select(band_names)

            terrain = ee.Terrain.products(ee.Image('USGS/SRTMGL1_003')).clip(region)
            slope = terrain.select('slope').multiply(deg2rad)
            aspect = terrain.select('aspect').multiply(deg2rad)
            p = slope.tan().divide(smooth).atan()

            z = ee.Image.constant(ee.Number(i.get('MEAN_SOLAR_ZENITH_ANGLE')).multiply(deg2rad))
            az = ee.Image.constant(ee.Number(i.get('MEAN_SOLAR_AZIMUTH_ANGLE')).multiply(deg2rad))

            cosao = (az.subtract(aspect)).cos()
            cosi = i.expression(
                'cosP*cosZ + sinP*sinZ*cosao',
                {
                    'cosP': p.cos(),
                    'cosZ': z.cos(),
                    'sinP': p.sin(),
                    'sinZ': z.sin(),
                    'cosao': cosao
                }
            )

            reg_img = ee.Image.cat(ee.Image(1).rename('a'), cosi.rename('slope'), sub_img)
            lr = ee.Reducer.linearRegression(numX=2, numY=len(band_names))

            fit = reg_img.reduceRegion(
                reducer=lr,
                geometry=region,
                scale=scale,
                maxPixels=1e13
            )

            coeff_array = ee.Array(fit.get('coefficients'))
            intercept = ee.Array(coeff_array.toList().get(0))
            slope_reg = ee.Array(coeff_array.toList().get(1))
            C = intercept.divide(slope_reg)
            Cimg = ee.Image.constant(C.toList())

            corrected = sub_img.expression(
                '(img*(cosZ + C)) / (slope + C)',
                {
                    'img': sub_img,
                    'cosZ': z.cos(),
                    'slope': cosi,
                    'C': Cimg
                }
            )
            return corrected.copyProperties(i)

        return ee.Image(
            ee.Algorithms.If(
                is_empty,
                img,
                do_correction(img)
            )
        )
    return TerrainCorrection_child


def makeName(s):
    return s.strip().lower().replace(' ', '_')


class downloadImageryS2():
    def __init__(self, folder, year, size, topocorrection=True):
        self.folder = folder
        self.year = year
        self.size = size
        self.topocorrection = topocorrection
        self.scale = 10
        self.d = 111319.49079327357

        self.sr_id = "COPERNICUS/S2_SR"
        self.cloud_id = "COPERNICUS/S2_CLOUD_PROBABILITY"

        self.final_bands = S2_BANDS
        self.ndvi_bands = NDVI_BANDS

        self.batch_spec = None
        # Filtrar tile si se desea
        self.tile = None

    def coords_to_box(self, coords):
        half_side_deg = 0.5*self.size*self.scale/self.d
        ll_x = coords[0]-half_side_deg
        ll_y = coords[1]-half_side_deg
        ur_x = coords[0]+half_side_deg
        ur_y = coords[1]+half_side_deg
        return ee.Geometry.Rectangle([ll_x, ll_y, ur_x, ur_y])

    def prepare_batch(self, coords):

        # Definir region
        boxes = [ee.Feature(self.coords_to_box(c)) for c in coords]
        self.region = ee.FeatureCollection(boxes).union().geometry()

        start_date = f"{self.year-1}-01-01"
        end_date = f"{self.year+1}-12-31"

        sr_coll = (ee.ImageCollection(self.sr_id)
                   .filterDate(start_date, end_date)
                   .filterBounds(self.region))

        # Filtrar las 10 bandas
        band_filters = []
        for b in self.final_bands:
            band_filters.append(ee.Filter.listContains('system:band_names', b))
        combined = band_filters[0]
        for bf in band_filters[1:]:
            combined = ee.Filter.And(combined, bf)
        sr_coll = sr_coll.filter(combined)

        # Filtrar props iluminacion
        sr_coll = sr_coll.filter(
            ee.Filter.notNull(['MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_SOLAR_AZIMUTH_ANGLE'])
        )

        if self.tile:
            sr_coll = sr_coll.filterMetadata('MGRS_TILE', 'equals', self.tile)

        cloud_coll = (ee.ImageCollection(self.cloud_id)
                      .filterDate(start_date, end_date)
                      .filterBounds(self.region))

        if self.tile:
            cloud_coll = cloud_coll.filterMetadata('MGRS_TILE', 'equals', self.tile)

        join_filter = ee.Filter.maxDifference(
            difference=8*60*60*1000,
            leftField='system:time_start',
            rightField='system:time_start'
        )
        inner_join = ee.Join.inner()
        joined = inner_join.apply(sr_coll, cloud_coll, join_filter)

        def mergeBands(f):
            sr_img = ee.Image(f.get('primary'))
            prob_img = ee.Image(f.get('secondary'))
            masked = maskCloudsByProbability(sr_img, prob_img.select('probability'))
            return masked.copyProperties(sr_img, sr_img.propertyNames())

        merged_coll = joined.map(mergeBands)
        masked_ic = ee.ImageCollection(merged_coll.map(lambda x: ee.Image(x)))
        if masked_ic.size().getInfo() == 0:
            self.batch_spec = None
            return

        if self.topocorrection:
            masked_ic = masked_ic.map(TerrainCorrection(self.scale, self.final_bands))

        masked_ic = masked_ic.map(addNDVIProperty(self.region)).sort('NDVI', False)
        if masked_ic.size().getInfo() == 0:
            self.batch_spec = None
            return

        def resampleBilinear(img):
            return img.resample('bilinear')

        mosaic = (masked_ic
                  .select(self.final_bands)
                  .map(resampleBilinear)
                  .reduce(ee.Reducer.median())
                  .multiply(ee.Image.constant(255.0/10000.0))
                  .toUint8()
                  .clip(self.region)
                  .rename(*self.final_bands))

        if mosaic.bandNames().size().getInfo() == 0:
            self.batch_spec = None
            return

        self.batch_spec = mosaic


def main():
    import argparse
    import pandas as pd
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Google Drive folder")
    parser.add_argument("year",   type=int, help="Year para las imagenes")
    parser.add_argument("batches", nargs="+", help="Lista batch IDs (ej. 0 1 2 ...)")
    parser.add_argument("-s", "--size", type=int, default=256,
                        help="Tamaño en pix, must be even. Default=256")

    args = parser.parse_args()

    try:
        villages = pd.read_csv(os.path.join(os.pardir, 'local', 'villages_batch.csv'), dtype=str)
    except FileNotFoundError:
        sys.exit(1)

    if args.size % 2 != 0:
        sys.exit(1)

    all_tasks = []
    start_global = time.time()

    for batch_id in args.batches:
        batch = str(batch_id)
        subset = villages.loc[villages.batch.eq(batch)]
        N = subset.shape[0]

        downloader = downloadImageryS2(args.folder, args.year, args.size)
        coords_list = [[float(r.lon), float(r.lat)] for i, r in subset.iterrows()]
        downloader.prepare_batch(coords_list)

        if downloader.batch_spec is None:
            continue

        desc = f"S2_BMSP_{batch}"
        task = ee.batch.Export.image.toDrive(
            image=downloader.batch_spec,
            folder=downloader.folder,
            description=desc,
            scale=downloader.scale,
            maxPixels=1e13
        )
        task.start()
        all_tasks.append((batch, task))

        loc_path = os.path.join(os.pardir, 'local', 'villages_loc.csv')
        with open(loc_path, 'a') as vf:
            for i, r in subset.iterrows():
                vf.write(f"{makeName(r.village)},{batch}\n")

        log_path = os.path.join(os.pardir, 'local', 'log_parallel.csv')
        with open(log_path, 'a') as lf:
            lf.write(f"{batch},Started,{N},{time.time()-start_global}\n")

    still_active = True
    while still_active:
        still_active = False
        active_batches = []
        for (bid, tsk) in all_tasks:
            if tsk.active():
                active_batches.append(bid)
                still_active = True
        if still_active:
            time.sleep(5)

    print(time.time()-start_global)


if __name__ == "__main__":
    main()
