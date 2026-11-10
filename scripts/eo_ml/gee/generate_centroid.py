"""
---- GENERATE CENTROID OF VILLAGES ----
this file uses generate_urban_centroid.py and generate_rural_centroid.py
to generate the centroid of the villages.
args: from_region to_region
"""

# Importing required libraries
import argparse
import subprocess
from multiprocessing import Process


def run_script(script, region):
    """Function to run a script with the region argument."""
    subprocess.run(["python", script, "-region", str(region)], check=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--from_region', type=int, help='Starting region number', required=True)
    parser.add_argument('-t', '--to_region', type=int, help='Ending region number', required=True)
    args = parser.parse_args()


    for region in range(args.from_region, args.to_region + 1):

        process_rural = Process(target=run_script, args=("generate_rural_centroid.py", region))
        process_urban = Process(target=run_script, args=("generate_urban_centroid.py", region))

        process_rural.start()
        process_rural.join()

        process_urban.start()
        process_urban.join()


if __name__ == "__main__":
    main()
