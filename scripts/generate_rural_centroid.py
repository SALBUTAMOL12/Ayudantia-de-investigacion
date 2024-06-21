"""
THIS FILE IS USED TO GENERATE THE RURAL CENTROID FOR THE RURAL AREAS IN THE COUNTRY.
THE CENTROID IS CALCULATED USING THE LATITUDE AND LONGITUDE OF THE SPECIFIC ENTIDAD
"""

# Importing required libraries
import geopandas as gpd
import os
import argparse

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('-region', '-region', type=int, help='Region number')
parser.set_defaults(region=1)
args = parser.parse_args()

# Load rural village data

path_rural_village = os.path.join(os.pardir, "local", "entidades", "Microdatos_Entidad.shp")
rural_village_data = gpd.read_file(path_rural_village)
rural_village_data["COD_REGION"] = rural_village_data["COD_REGION"].astype(int)
# Filter by region
rural_village_data = rural_village_data[(rural_village_data['COD_REGION'] == args.region) & (rural_village_data['CANTIDAD_H'] != 0)].copy()

names_comunas = rural_village_data['NOMBRE_COM'].unique().tolist()

# tranform the COD_COM to int
rural_village_data['COD_COMUNA'] = rural_village_data['COD_COMUNA'].astype(int)

# loop through each comuna

for comuna in names_comunas:
    # Filter by comuna
    rural_village_data_comuna = rural_village_data[rural_village_data['NOMBRE_COM'] == comuna].copy()
    # !!! cut number
    cut_number = rural_village_data_comuna['COD_COMUNA'][rural_village_data_comuna["NOMBRE_COM"] == comuna].unique().tolist()[0]
    # there is not need to calculate the total population in each zone
    # centroid of the entidades
    rural_village_data_comuna['centroid'] = rural_village_data_comuna['geometry'].centroid

    # convert the centroid csr to wgs84
    rural_village_data_comuna['centroid'] = rural_village_data_comuna['centroid'].to_crs(4326)

    rural_village_data_comuna["row"] = rural_village_data_comuna.apply(lambda x: f"{cut_number}0{x['CODIGO_LOC']}{x['CODIGO_ENT']},{x['centroid'].x},{x['centroid'].y}", axis=1)

    # save the row in a common csv file
    file_path = os.path.join(os.pardir, "local", "centroid", "villages.csv")
    with open(file_path, 'a') as f:
        for row in rural_village_data_comuna["row"]:
            f.write(f"{row}\n")
