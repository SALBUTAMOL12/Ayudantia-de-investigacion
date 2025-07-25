"""
---- DOWNLOAD IMAGERY FOR BATCHES OF VILLAGES ----
this script fetches satellite imagery for groups of nearby villages
identified by a batch number. for each batch and year it collects the
available scenes, builds a quality mosaic favouring high mean NDVI and
exports the result to a folder in Google Drive.

Positional arguments:
    folder   - Drive folder where images will be stored
    year     - target year for the mosaic
    sensor   - sensor code (L5, L7, L8 or S2)
    batches  - batch IDs or ranges, e.g. "0 2 5-10"

Optional arguments:
    -s, --size  size of the box around each village in pixels,
                even number (default 256)

Output:
    Drive/{folder}/{sensor}_BMSP_{batch}.tif  multispectral mosaic
    Drive/{folder}/{sensor}_BPAN_{batch}.tif  panchromatic mosaic if available
    local/villages_loc.csv  mapping of villages to batches
    log.csv                 execution times
"""

import ee
import sys

# initialise Earth Engine using default credentials

PROJECT_ID = "test-449001"

ee.Initialize(project=PROJECT_ID)


CLOUD_PROB_THRESHOLD = 25
# full Sentinel-2 band list
S2_BANDS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
]

S2_RGB_BANDS = ["B4", "B3", "B2"]
NDVI_BANDS_S2 = ["B8", "B4"]

TIME_DIFFERENCE = 20 * 60 * 1000  # 20 minutes in milliseconds

# -------------------------- sentinel 2 ---------------------------------


def maskCloudsByProbability(sr_image, prob_image):
    cloud_mask = prob_image.lt(CLOUD_PROB_THRESHOLD)
    return sr_image.updateMask(cloud_mask)


def addNDVIProperty(region, bands):
    def _add(img):
        ndvi = img.normalizedDifference(bands).rename("ndvi")
        ndvi = ndvi.where(ndvi.lt(0), 0)
        avg = ndvi.reduceRegion(ee.Reducer.mean(), region,
                                10, bestEffort=True).get("ndvi")
        return img.set({"NDVI": avg})
    return _add


def TerrainCorrection(scale, band_names, smooth=5):
    def _tc(img):
        deg2rad = 0.0174533
        foot = ee.Geometry(img.get('system:footprint'))
        area_foot = foot.area(1)
        is_empty = area_foot.eq(0)

        def do_correction(i):
            region = ee.Geometry(i.get('system:footprint'))
            sub_img = i.select(band_names)
            terrain = ee.Terrain.products(
                ee.Image('USGS/SRTMGL1_003')).clip(region)
            slope = terrain.select('slope').multiply(deg2rad)
            aspect = terrain.select('aspect').multiply(deg2rad)
            p = slope.tan().divide(smooth).atan()

            z = ee.Image.constant(
                ee.Number(i.get('MEAN_SOLAR_ZENITH_ANGLE')).multiply(deg2rad))
            az = ee.Image.constant(
                ee.Number(i.get('MEAN_SOLAR_AZIMUTH_ANGLE')).multiply(deg2rad))

            cosao = (az.subtract(aspect)).cos()
            cosi = i.expression(
                'cosP*cosZ + sinP*sinZ*cosao',
                {
                    'cosP': p.cos(),
                    'cosZ': z.cos(),
                    'sinP': p.sin(),
                    'sinZ': z.sin(),
                    'cosao': cosao,
                }
            )

            reg_img = ee.Image.cat(ee.Image(1).rename(
                'a'), cosi.rename('slope'), sub_img)
            lr = ee.Reducer.linearRegression(numX=2, numY=len(band_names))

            fit = reg_img.reduceRegion(
                reducer=lr,
                geometry=region,
                scale=scale,
                bestEffort=True,
                maxPixels=1e14,
            )

            coeff_array = ee.Array(fit.get('coefficients'))
            intercept = ee.Array(coeff_array.toList().get(0))
            slope_reg = ee.Array(coeff_array.toList().get(1))
            C = intercept.divide(slope_reg)
            Cimg = ee.Image.constant(C.toList())

            corrected = sub_img.expression(
                '(img*(cosZ + C)) / (slope + C)',
                {'img': sub_img, 'cosZ': z.cos(), 'slope': cosi, 'C': Cimg}
            )
            return corrected.copyProperties(i)

        return ee.Image(ee.Algorithms.If(is_empty, img, do_correction(img)))
    return _tc


class Sentinel2Downloader:
    def __init__(self, folder, year, size, topocorrection=True):
        self.folder = folder
        self.year = year
        self.size = size
        self.topocorrection = topocorrection
        self.scale = 10
        self.d = 111319.49079327357
        # use the harmonized Sentinel-2 surface reflectance collection
        self.sr_id = "COPERNICUS/S2_SR_HARMONIZED"
        self.cloud_id = "COPERNICUS/S2_CLOUD_PROBABILITY"
        # export only the RGB bands
        self.final_bands = S2_BANDS
        self.ndvi_bands = NDVI_BANDS_S2
        self.batch_spec = None
        self.tile = None

    def coords_to_box(self, coords):
        half = 0.5 * self.size * self.scale / self.d
        ll_x = coords[0] - half
        ll_y = coords[1] - half
        ur_x = coords[0] + half
        ur_y = coords[1] + half
        return ee.Geometry.Rectangle([ll_x, ll_y, ur_x, ur_y])

    def prepare_batch(self, coords):
        boxes = [ee.Feature(self.coords_to_box(c)) for c in coords]
        self.region = ee.FeatureCollection(boxes).union().geometry()

        start_date = f"{self.year}-01-01"
        end_date = f"{self.year}-12-31"

        sr_coll = (ee.ImageCollection(self.sr_id)
                   .filterDate(start_date, end_date)
                   .filterBounds(self.region))

        band_filters = []
        for b in self.final_bands:
            band_filters.append(ee.Filter.listContains('system:band_names', b))
        combined = band_filters[0]
        for bf in band_filters[1:]:
            combined = ee.Filter.And(combined, bf)
        sr_coll = sr_coll.filter(combined)

        sr_coll = sr_coll.filter(
            ee.Filter.notNull(['MEAN_SOLAR_ZENITH_ANGLE',
                              'MEAN_SOLAR_AZIMUTH_ANGLE'])
        )

        if self.tile:
            sr_coll = sr_coll.filterMetadata('MGRS_TILE', 'equals', self.tile)

        # quick check to avoid heavy operations when there are no images
        if sr_coll.limit(1).size().getInfo() == 0:
            self.batch_spec = None
            return

        cloud_coll = (ee.ImageCollection(self.cloud_id)
                      .filterDate(start_date, end_date)
                      .filterBounds(self.region))

        if self.tile:
            cloud_coll = cloud_coll.filterMetadata(
                'MGRS_TILE', 'equals', self.tile)

        join_filter = ee.Filter.maxDifference(
            difference=TIME_DIFFERENCE,
            leftField='system:time_start',
            rightField='system:time_start',
        )
        inner_join = ee.Join.inner()
        joined = inner_join.apply(sr_coll, cloud_coll, join_filter)

        def mergeBands(f):
            sr_img = ee.Image(f.get('primary'))
            prob_img = ee.Image(f.get('secondary'))
            masked = maskCloudsByProbability(
                sr_img, prob_img.select('probability'))
            return masked.copyProperties(sr_img, sr_img.propertyNames())

        merged_coll = joined.map(mergeBands)
        masked_ic = ee.ImageCollection(merged_coll.map(lambda x: ee.Image(x)))
        if masked_ic.limit(1).size().getInfo() == 0:
            self.batch_spec = None
            return

        if self.topocorrection:
            masked_ic = masked_ic.map(
                TerrainCorrection(self.scale, self.final_bands))

        masked_ic = masked_ic.map(addNDVIProperty(
            self.region, self.ndvi_bands)).sort('NDVI', False)
        if masked_ic.limit(1).size().getInfo() == 0:
            self.batch_spec = None
            return

        def resampleBilinear(img):
            return img.resample('bilinear')

        mosaic = (masked_ic
                  .select(self.final_bands)
                  .map(resampleBilinear)
                  .reduce(ee.Reducer.median())
                  .multiply(ee.Image.constant(255.0 / 10000.0))
                  .toUint8()
                  .clip(self.region)
                  .rename(*self.final_bands))

        if mosaic.bandNames().size().getInfo() == 0:
            self.batch_spec = None
            return

        self.batch_spec = mosaic
        self.batch_pan = None


# ---------------------------- landsat ----------------------------------

def keepClear(region, sat):
    def _keep(im):
        qa = im.select('QA_PIXEL')
        qa_clouds = extractQABits(qa, 8, 9)
        qa_shadow = extractQABits(qa, 10, 11)
        qa2 = im.select('QA_RADSAT')
        if sat == 'L8':
            qa_sat = extractQABits(qa2, 1, 3)
        else:
            qa_sat = extractQABits(qa2, 0, 2)
        mask = qa_clouds.lte(2).And(qa_shadow.lte(2)).And(qa_sat.eq(0))
        if sat == 'L8':
            qa_cirrus = extractQABits(qa, 14, 15)
            qa_terrain = extractQABits(qa2, 11, 11)
            mask = mask.And(qa_cirrus.lte(1)).And(qa_terrain.eq(0))
        valid = mask.reduceRegion(ee.Reducer.sum(), region).get('QA_PIXEL')
        tot = mask.reduceRegion(ee.Reducer.count(), region).get('QA_PIXEL')
        return im.updateMask(mask).copyProperties(
            im).set({'QI': ee.Number(valid).divide(tot)})
    return _keep


def extractQABits(qaBand, bitStart, bitEnd):
    numBits = bitEnd - bitStart + 1
    return qaBand.rightShift(bitStart).mod(2 ** numBits)


def addNDVI(region, ndvi_bands):
    def _add(image):
        ndvi = ee.Image(image).normalizedDifference(ndvi_bands).rename('ndvi')
        ndvi = ndvi.where(ndvi.lt(0), 0)
        average_ndvi = ndvi.reduceRegion(ee.Reducer.mean(), region).get('ndvi')
        return image.copyProperties(image).set(
            {'NDVI': ee.Number(average_ndvi)})
    return _add


def TerrainCorrectionL8(scale, n_bands, smooth=5):
    def _tc(img):
        degree2radian = 0.0174533
        region = ee.Geometry.Polygon(ee.Geometry(
            img.get('system:footprint')).coordinates())
        terrain = ee.call('Terrain', ee.Image('USGS/SRTMGL1_003')).clip(region)
        p = terrain.select(['slope']).multiply(
            degree2radian).tan().divide(smooth).atan()
        z = ee.Image(ee.Number(img.get('SUN_ELEVATION')
                               ).multiply(degree2radian))
        az = ee.Image(ee.Number(img.get('SUN_AZIMUTH')
                                ).multiply(degree2radian))
        o = terrain.select(['aspect']).multiply(degree2radian)
        cosao = (az.subtract(o)).cos()
        cosi = img.expression('((cosp*cosz) + ((sinp*sinz)*(cosao)))',
                              {'cosp': p.cos(), 'cosz': z.cos(), 'sinp': p.sin(),
                               'sinz': z.sin(), 'az': az, 'o': o, 'cosao': cosao})
        reg_img = ee.Image.cat(ee.Image(1).rename('a'), cosi, img)
        lr_reducer = ee.Reducer.linearRegression(numX=2, numY=n_bands)
        fit = reg_img.reduceRegion(
            reducer=lr_reducer, geometry=region, scale=scale, maxPixels=1e10)
        coeff_array = ee.Array(fit.get('coefficients'))
        intercept = ee.Array(coeff_array.toList().get(0))
        slope_reg = ee.Array(coeff_array.toList().get(1))
        C = intercept.divide(slope_reg)
        Cimg = ee.Image.constant(C.toList())
        newimg = img.expression('((img * ((cosz) + C))/(cosi + C))',
                                {'img': img, 'cosz': z.cos(), 'cosi': cosi, 'C': Cimg})
        return newimg.copyProperties(img)
    return _tc


class LandsatDownloader:
    def __init__(self, folder, year, sensor, size, topocorrection=True):
        collections = {
            'L5': {
                'SR': ['LANDSAT/LT05/C02/T1_SR', 'LANDSAT/LT05/C02/T2_SR'],
                'TOA': ['LANDSAT/LT05/C02/T1_TOA', 'LANDSAT/LT05/C02/T2_TOA'],
            },
            'L7': {
                'SR': ['LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LE07/C02/T2_L2'],
                'TOA': ['LANDSAT/LE07/C02/T1_TOA', 'LANDSAT/LE07/C02/T2_TOA'],
            },
            'L8': {
                'SR': ['LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC08/C02/T2_L2'],
                'TOA': ['LANDSAT/LC08/C02/T1_TOA', 'LANDSAT/LC08/C02/T2_TOA'],
            },
        }
        final_bands = {
            'L5': [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'], [], ['SR_B4', 'SR_B3']],
            'L7': [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'], ['B8'], ['SR_B4', 'SR_B3']],
            'L8': [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], ['B8'], ['SR_B5', 'SR_B4']],
        }

        self.folder = folder
        self.year = year
        self.sensor = sensor
        self.topocorrection = topocorrection
        try:
            self.collection = collections[sensor]
            self.final_spec_bands = final_bands[sensor][0]
            self.final_pan_bands = final_bands[sensor][1]
            self.ndvi_bands = final_bands[sensor][2]
            self.pan = bool(len(self.final_pan_bands))
        except KeyError:
            raise ValueError('Sensor must be L5, L7, or L8')
        self.size = size
        self.size_adj = int(size / (1 + int(self.pan)))
        self.scale = 30
        self.scale_adj = int(self.scale / (1 + int(self.pan)))
        self.d = 111319.49079327357
        self.batch_spec = None
        self.batch_pan = None

    def coords_to_box(self, coords):
        ll_x = coords[0] - (0.5 / (1 + int(self.pan))) * \
            self.size * self.scale / self.d
        ll_y = coords[1] - (0.5 / (1 + int(self.pan))) * \
            self.size * self.scale / self.d
        ur_x = coords[0] + (0.5 / (1 + int(self.pan))) * \
            self.size * self.scale / self.d
        ur_y = coords[1] + (0.5 / (1 + int(self.pan))) * \
            self.size * self.scale / self.d
        return ee.Geometry.Rectangle(ll_x, ll_y, ur_x, ur_y)

    def collect_images(self):
        def toa_collection():
            toa_collection = ee.ImageCollection(self.collection['TOA'][0])
            for c in self.collection['TOA'][1:]:
                toa_collection = toa_collection.merge(ee.ImageCollection(c))
            return toa_collection

        def addTOA(img):
            toa_coll = toa_collection()
            pan = toa_coll.filter(
                ee.Filter.eq('LANDSAT_PRODUCT_ID',
                             img.get('L1_LANDSAT_PRODUCT_ID'))
            ).select(['B8']).first()
            return ee.Algorithms.If(pan, img.addBands(pan), img)

        collection = ee.ImageCollection(self.collection['SR'][0])
        for c in self.collection['SR'][1:]:
            collection = collection.merge(ee.ImageCollection(c))

        collection = (collection
                      .filterDate(f'{self.year - 1}-01-01', f'{self.year + 1}-12-31')
                      .filterBounds(self.region))

        if self.pan:
            collection = collection.map(addTOA, True)

        collection = collection.map(
            keepClear(
                self.region,
                self.sensor)).filter(
            ee.Filter.gte(
                'QI',
                0.95))
        return collection

    def prepare_batch(self, coords):
        self.region = ee.FeatureCollection(
            [ee.Feature(self.coords_to_box(c)) for c in coords]).union().geometry()
        image = self.collect_images().map(
            addNDVI(self.region, self.ndvi_bands)).sort('NDVI', False)
        image_spec = image.select(self.final_spec_bands)

        if self.sensor == 'L8':
            if self.topocorrection:
                image_spec = image_spec.map(TerrainCorrectionL8(
                    self.scale, len(self.final_spec_bands)))
            image_spec = image_spec.reduce(ee.Reducer.median()).multiply(
                ee.Image.constant(255 / 65536)).toUint8()
            self.batch_spec = ee.Image(image_spec).clip(
                self.region).rename(*self.final_spec_bands)
            if self.pan:
                image_pan = image.select(self.final_pan_bands)
                if self.topocorrection:
                    image_pan = image_pan.map(TerrainCorrectionL8(
                        self.scale_adj, len(self.final_pan_bands)))
                image_pan = image_pan.reduce(ee.Reducer.median()).multiply(
                    ee.Image.constant(255 / 1.6)).toUint8()
                self.batch_pan = ee.Image(image_pan).clip(
                    self.region).rename(*self.final_pan_bands)
        else:
            if self.topocorrection:
                image_spec = image_spec.map(TerrainCorrectionL8(
                    self.scale, len(self.final_spec_bands)))
            image_spec = image_spec.reduce(ee.Reducer.firstNonNull()).multiply(
                ee.Image.constant(255 / 10000)).toUint8()
            self.batch_spec = ee.Image(image_spec).clip(
                self.region).rename(*self.final_spec_bands)
            if self.pan:
                image_pan = image.select(self.final_pan_bands)
                if self.topocorrection:
                    image_pan = image_pan.map(TerrainCorrectionL8(
                        self.scale_adj, len(self.final_pan_bands)))
                image_pan = image_pan.reduce(
                    ee.Reducer.firstNonNull()).multiply(
                    ee.Image.constant(255)).toUint8()
                self.batch_pan = ee.Image(image_pan).clip(
                    self.region).rename(*self.final_pan_bands)


# ------------------------------ helpers --------------------------------

def makeName(s):
    return s.strip().lower().replace(' ', '_')


def expand_batches(batch_args):
    batches = set()
    for token in batch_args:
        if '-' in token:
            start, end = token.split('-', 1)
            start, end = int(start), int(end)
            if start > end:
                start, end = end, start
            batches.update(range(start, end + 1))
        else:
            batches.add(int(token))
    return [str(b) for b in sorted(batches)]


# map sensor codes to the class responsible for each platform
SENSOR_MAP = {
    'S2': Sentinel2Downloader,
    'L5': LandsatDownloader,
    'L7': LandsatDownloader,
    'L8': LandsatDownloader,
}


# --------------------------------- main --------------------------------

def main():
    import argparse
    import pandas as pd
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Google Drive folder")
    parser.add_argument("year", type=int, help="Year of imagery")
    parser.add_argument("sensor", help="Sensor code (L5, L7, L8, S2)")
    parser.add_argument(
        "batches",
        nargs="+",
        help="Batch IDs or ranges (e.g. 0 2 5-10)",
    )
    parser.add_argument("-s", "--size", type=int, default=256,
                        help="Image size in pixels, even number")
    args = parser.parse_args()

    # expand batch arguments that may include ranges into an ordered list
    # of ids
    batch_ids = expand_batches(args.batches)

    try:
        villages = pd.read_csv(os.path.join('local', 'villages_batch.csv'), dtype=str)
    except FileNotFoundError:
        print("Error: villages_batch.csv not found in local directory.")
        sys.exit(1)

    if args.size % 2 != 0:
        sys.exit(1)

    if args.sensor not in SENSOR_MAP:
        sys.exit(1)

    # list of export tasks submitted to Google Earth Engine
    all_tasks = []
    start_global = time.time()

    # process each requested batch
    for batch in batch_ids:
        subset = villages.loc[villages.batch.eq(batch)]
        if subset.empty:
            continue
        # list of village coordinates within the batch
        coords_list = [[float(r.lon), float(r.lat)]
                       for _, r in subset.iterrows()]
        if args.sensor == 'S2':
            downloader = Sentinel2Downloader(args.folder, args.year, args.size)
        else:
            downloader = LandsatDownloader(
                args.folder, args.year, args.sensor, args.size)
        downloader.prepare_batch(coords_list)
        if downloader.batch_spec is None:
            continue
        desc = f"{args.sensor}_BMSP_{batch}"
        # export the multispectral image to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=downloader.batch_spec,
            folder=downloader.folder,
            description=desc,
            scale=downloader.scale,
            maxPixels=1e13,
        )
        task.start()
        all_tasks.append((batch, task))

        if getattr(downloader, 'batch_pan', None):
            desc_pan = f"{args.sensor}_BPAN_{batch}"
            task_pan = ee.batch.Export.image.toDrive(
                image=downloader.batch_pan,
                folder=downloader.folder,
                description=desc_pan,
                scale=getattr(downloader, 'scale_adj', downloader.scale),
                maxPixels=1e13,
            )
            task_pan.start()
            all_tasks.append((batch, task_pan))

        loc_path = os.path.join('local', 'villages_loc.csv')
        # append village names and batch number to local csv
        with open(loc_path, 'a') as vf:
            for _, r in subset.iterrows():
                vf.write(f"{makeName(r.village)},{batch}\n")

    # wait for all export tasks to finish
    still_active = True
    while still_active:
        still_active = False
        for _, tsk in all_tasks:
            if tsk.active():
                still_active = True
        if still_active:
            time.sleep(5)

    print(time.time() - start_global)


if __name__ == "__main__":
    main()
