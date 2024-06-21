"""
THIS FILE IS USED TO GENERATE URBAN CENTROID BY REGION
"""
# Importing required libraries
import geopandas as gpd
import os
import pandas as pd
import numpy as np
import argparse
from shapely.geometry import Point

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('-region', '-region', type=int, help='Region number')
# parser.set_defaults(region=13)
args = parser.parse_args()

# Load urban village data

# mapped regions
mapped_regions = {'REGIÓN DE TARAPACÁ': 1,
                  'REGIÓN DE ANTOFAGASTA': 2,
                  'REGIÓN DE ATACAMA': 3,
                  'REGIÓN DE COQUIMBO': 4,
                  'REGIÓN DE VALPARAÍSO': 5,
                  'REGIÓN DEL LIBERTADOR GENERAL BERNARDO O´HIGGINS': 6,
                  'REGIÓN DEL MAULE': 7,
                  'REGIÓN DEL BIOBÍO': 8,
                  'REGIÓN DE LA ARAUCANÍA': 9,
                  'REGIÓN DE LOS LAGOS': 10,
                  'REGIÓN DE AYSÉN DEL GENERAL CARLOS IBÁÑEZ DEL CAMPO': 11,
                  'REGIÓN DE MAGALLANES Y DE LA ANTÁRTICA CHILENA': 12,
                  'REGIÓN METROPOLITANA DE SANTIAGO': 13,
                  'REGIÓN DE LOS RÍOS': 14,
                  'REGIÓN DE ARICA Y PARINACOTA': 15,
                  'REGIÓN DE ÑUBLE': 16}

path_urban_village = os.path.join(os.pardir, "local", "manzanas", "Microdatos_Manzana.shp")
urban_village_data = gpd.read_file(r'C:\Users\Felipe\Escritorio\Ayudantia-de-investigacion\local\manzanas\Microdatos_Manzana.shp')

urban_village_data['REGION'] = urban_village_data['REGION'].replace("REGIÓN DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS", 'REGIÓN DEL LIBERTADOR GENERAL BERNARDO O´HIGGINS')
urban_village_data['COD_REGION'] = urban_village_data['REGION'].map(mapped_regions)

# Filter by region
urban_village_data = urban_village_data[urban_village_data['COD_REGION'] == args.region].copy()

names_comunas = urban_village_data['COMUNA'].unique().tolist()


# loop through each comuna

for comuna in names_comunas:
    # print(f"Processing {comuna}...")
    # Filter by comuna
    urban_village_data_comuna = urban_village_data[urban_village_data['COMUNA'] == comuna].copy()
    # cut number
    cut_number = urban_village_data_comuna['CUT'][urban_village_data_comuna["COMUNA"] == comuna].unique().tolist()[0]

    # Calculate total population in each zone
    urban_village_data_comuna['total_h'] = urban_village_data_comuna.groupby('ZONA_CENSA')['CANTIDAD_H'].transform('sum')

    # Calculate weights, handling divisions by zero
    urban_village_data_comuna['weights'] = urban_village_data_comuna['CANTIDAD_H'] / urban_village_data_comuna['total_h'].replace(0, np.nan)

    # Calculate centroids of the zones
    urban_village_data_comuna['centroid'] = urban_village_data_comuna['geometry'].centroid

    # Multiply the centroid by the weights, handling NaN weights
    urban_village_data_comuna['weighted_centroid'] = urban_village_data_comuna.apply(
        lambda row: Point(row["centroid"].x * row["weights"],
                          row["centroid"].y * row["weights"]) if not pd.isna(row["weights"]) else None, axis=1)

    # Ensure all centroids are valid before proceeding
    urban_village_data_comuna = urban_village_data_comuna.dropna(subset=['weighted_centroid'])

    # Extract x and y values of the weighted centroid
    urban_village_data_comuna['x'] = urban_village_data_comuna['weighted_centroid'].apply(lambda p: p.x if p else None)
    urban_village_data_comuna['y'] = urban_village_data_comuna['weighted_centroid'].apply(lambda p: p.y if p else None)

    # Group by zone and sum the x and y values
    urban_village_data_comuna_zona_censa = urban_village_data_comuna.groupby('ZONA_CENSA').agg({'x': 'sum', 'y': 'sum'}).reset_index()

    # Create points from x and y values
    urban_village_data_comuna_zona_censa['geometry'] = [Point(x, y) for x, y in zip(urban_village_data_comuna_zona_censa['x'], urban_village_data_comuna_zona_censa['y'])]

    # Convert to a GeoDataFrame and set the coordinate system
    urban_village_data_comuna_zona_censa = gpd.GeoDataFrame(urban_village_data_comuna_zona_censa, crs=urban_village_data_comuna.crs).to_crs(epsg=4326)

    # Create row in string format (urban: 1, rural: 0)
    urban_village_data_comuna_zona_censa['row'] = urban_village_data_comuna_zona_censa.apply(
        lambda x: f"{cut_number}1{x['ZONA_CENSA']},{x['geometry'].x},{x['geometry'].y}", axis=1
    )

    # Save the row in a common CSV file
    file_path = os.path.join(os.pardir, "local", "centroid", "villages.csv")
    with open(file_path, 'a') as f:
        for row in urban_village_data_comuna_zona_censa['row']:
            f.write(f"{row}\n")
