#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Script to extract shots from the GEDI L2L4A data tables located
#            within the EnMAP tiles bounding boxes and store them in CSV files

# %%
# # ----- Imports
from hybiomass.dataset_creation.finder import GediShotsFinder
from os import path
from hybiomass.dataset_creation.utils import get_root_dir, get_file_info, \
    start_script, stop_script, get_absolute_path
import logging
import os
import pandas as pd
from dask import delayed, compute
from tqdm.dask import TqdmCallback
from ruamel.yaml import YAML

# %%
# # ----- Define and create directories
ROOT_DIR = get_root_dir()
filepath, filename = get_file_info()

# %%
# # ----- Initialize script and logging
start_t = start_script(ROOT_DIR, filepath, filename, log=True, level=logging.DEBUG)

# %%
# # ----- Define the function to process a single tile
def process_tile(tile):
    """
    Processes a tile by using its bounding box to filter shots from the L2L4A data tables.

    This function performs the following steps:
    1. Extracts the bounding box and timestamp unique identifier from the tile metadata.
    2. Creates an instance of `GediShotsFinder` using the bounding box and timestamp.
    3. Constructs an output file path based on the timestamp.
    4. Retrieves and saves the filtered entries to the specified CSV file.

    Args:
        tile (dict): A dictionary containing metadata for the tile. The dictionary should include:
            - "min_lon" (float): Minimum longitude of the bounding box.
            - "min_lat" (float): Minimum latitude of the bounding box.
            - "max_lon" (float): Maximum longitude of the bounding box.
            - "max_lat" (float): Maximum latitude of the bounding box.
            - "timestamp_id" (str): The timestamp ID associated with the tile.

    Returns:
        None

    Notes:
        - The output CSV file is named using the timestamp ID and saved in the `output_dir` directory with a "_gedi_l2l4a.csv" suffix.
        - This function assumes the existence of `GediShotsFinder`, `l2l4a_dir`, and `output_dir` in the context where it is used.
    """
    bbox = [tile["min_lon"], tile["min_lat"], tile["max_lon"], tile["max_lat"]]
    timestamp = tile["timestamp_id"]
    finder = GediShotsFinder(l2l4a_dir, bbox, timestamp)
    output_path = os.path.join(output_dir, f"{finder.timestamp}_gedi_l2l4a.csv")
    finder.get_and_save_entries(output_path)

# %%
# # ----- Run code
try:
    ## Load the script configuration
    finder_config_path = path.join(ROOT_DIR, "config", "gedi", "shots_finder.yaml")
    with open(finder_config_path, "r") as f:
        yaml = YAML(typ="safe")
        finder_config = yaml.load(f)

    enmap_dir = get_absolute_path(ROOT_DIR, finder_config["enmap_dir"])
    l2l4a_dir = get_absolute_path(ROOT_DIR, finder_config["l2l4a_dir"])
    output_dir = get_absolute_path(ROOT_DIR, finder_config["output_dir"])
    gedi = finder_config["gedi"]
    region = finder_config["region"]
    num_workers = finder_config["num_workers"]

    ## Load the table containing the reprojected bounding boxes and timestamps of the EnMAP tiles
    tiles_filename = "enmap_coordinates_and_timestamps_all_tiles_reprojected.csv"
    tiles_path = os.path.join(enmap_dir, tiles_filename)
    df = pd.read_csv(tiles_path)
    if gedi and region=="EUR":
        df = df[(df["region"] == "EUR") & (df["gedi"] == "GEDI")]
        logging.info(
            f"Found {len(df)} EnMAP tiles in the DLR archive within the {region} and GEDI footprints"
        )
    elif gedi and region=="GLOBAL":
        df = df[df["gedi"] == "GEDI"]
        logging.info(
            f"Found {len(df)} EnMAP tiles in the DLR archive globally and GEDI footprints"
        )
    else:
        df = df[df["region"] == region]
        logging.info(
            f"Found {len(df)} EnMAP tiles in the DLR archive within the {region} footprint"
        )
    ## Filter out existing files
    existing_tiles = [t[:-15] for t in os.listdir(output_dir)]
    logging.info(f"Found {len(existing_tiles)} existing tiles")
    df = df[~df["timestamp_id"].isin(existing_tiles)]
    logging.info(f"Processing {len(df)} EnMAP tiles...")

    # %%
    ##### PARALLEL VERSION #####
    futures = [delayed(process_tile)(x) for _, x in df.iterrows()]
    with TqdmCallback(desc="compute"):
        results = compute(*futures, num_workers=num_workers)

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    print(f"An error occurred: {str(e)}")

# %%
# # ----- Finalize script and logging
finally:
    end_t = stop_script(start_t, filepath)
