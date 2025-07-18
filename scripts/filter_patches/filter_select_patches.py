#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Aaron Banze (aaron.banze@dlr.de)
# @License : (C)Copyright 2025, DLR
# @Desc    : Filter EnMAP & GEDI patches (e.g. quality metrics, spectral bands)

import logging
import multiprocessing
import os

import numpy as np
import pandas as pd
import rioxarray as rxr
from ruamel.yaml import YAML
from tqdm import tqdm

from hybiomass.dataset_creation.core import Core
from hybiomass.dataset_creation.utils import get_root_dir, get_file_info, start_script, \
    stop_script, get_absolute_path


# Define and create directories
ROOT_DIR = get_root_dir()
filepath, filename = get_file_info()

# Initialize script and logging
start_t = start_script(ROOT_DIR, filepath, filename, log=True, level=logging.INFO)

# Filter settings (set to False if filter should not be used):
L4A_HQFLAG = True  # l4_quality flag = 1
TREE_COVER = 10  # Min. tree cover in percent (tree cover >= x %) acc. to Hansen
LOSS_YEAR = 10
LOSS_YEAR_OPERATOR = "<" # To remove forest disturbances after 2010 (LOSSYEAR=10)
AGBD_UPPER_BOUND = 500  # Set to 500 to filter for value range [0, 500]
POWER_BEAM_ONLY = True  # Only use power beams (not coverage beams)
l2a_selalg_a0 = False  # If True, filter for l2a_selalg_a0 != 10
PFT_FOREST_ONLY = True

# Select settings
HYPERSPECTRAL = True  # If False, select RGB bands
MIN_GEDI_NOTNAN = 0.01  # fraction of values in patch, where GEDI values are not nan

# Define a function to append conditions to the gedi_conditions list
def append_condition(conditions, band, operator, value):
    """Append a condition to the gedi_conditions list."""
    condition = {
        "band": band,
        "operator": operator,
        "value": value,
    }
    conditions.append(condition)


def filter_select_save(args):
    enmap_path, gedi_path, output_path = args

    core = Core(enmap=enmap_path, gedi=gedi_path)

    # Filter the data based on conditions
    gedi_conditions = []

    # High quality AGBD estimate only
    if L4A_HQFLAG:
        append_condition(gedi_conditions, "l4a_hqflag", "==", True)
    # Selected algorithm != 10 only
    if l2a_selalg_a0:
        append_condition(gedi_conditions, "l2a_selalg_a0", "!=", 10)
    # Areas with tree cover (canopy closure >= 10% where vegetation >= 5m)
    if TREE_COVER:
        append_condition(gedi_conditions, "ls_treecov", ">=", TREE_COVER)  # ls_treecov, treecover2000
    # No forest disturbance since GEDI and EnMAP data acquisition (2019)
    if LOSS_YEAR:
        append_condition(gedi_conditions, "lossyear", LOSS_YEAR_OPERATOR, LOSS_YEAR)
    # Filter for power beams only (do not include coverage beams)
    # 4 out of 8 beams are power beams
    if POWER_BEAM_ONLY:
        append_condition(gedi_conditions, "beam", "==", 1)
    # Filter for AGBD values in range [0, 500] is recommended
    if AGBD_UPPER_BOUND:
        gedi_conditions.append({
            "band": "agbd_a0",
            "operator": "between",
            "lower_bound": 0,
            "upper_bound": AGBD_UPPER_BOUND,
            "left_closed": True,  # np.greater_equal if left_closed else np.greater
            "right_closed": True,  # np.less_equal if right_closed else np.less
        })

    # Filter by plant functional type (PFT); categories follow MODIS MCD12Q1v006
    if PFT_FOREST_ONLY:
        gedi_conditions.append({
            "band": "pft",
            "operator": "between",
            "lower_bound": 1,
            "upper_bound": 4,
            "left_closed": True,  # np.greater_equal if left_closed else np.greater
            "right_closed": True,  # np.less_equal if right_closed else np.less
        })

    # Select the bands to be included in the dataset
    enmap_bands = ["enmap_B11", "enmap_B29", "enmap_B44"]  # RGB
    # Hyperspectral bands except ones commonly removed
    if HYPERSPECTRAL:
        invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
                            137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
        valid_channels_ids = [c+1 for c in range(224) if c not in invalid_channels]
        enmap_bands = [f"enmap_B{i}" for i in valid_channels_ids]
        
    gedi_bands = ["agbd_a0", "region", "pft"]

    data = core.filter(gedi=gedi_conditions).select(enmap=enmap_bands,
                                                    gedi=gedi_bands)

    # Skip patches with less than a threshold of percentage of GEDI data not nan
    agbd_band_number = data.band_descriptions["agbd_a0"]
    fraction_notnan = data.sel(band=agbd_band_number).count() / data.sel(band=agbd_band_number).size
    if MIN_GEDI_NOTNAN:
        if fraction_notnan < MIN_GEDI_NOTNAN:
            return None  # skip this patch; dont save in final dataset
        
    # Save to the folder according to the region (0: Water)
    regions = {1: 'Europe', 2: 'North_Asia', 3: 'Australasia', 4: 'Africa',
               5: 'South_Asia', 6: 'South_America', 7: 'North_America'}
    
    # Iterate through each region
    region_band_number = data.band_descriptions["region"]
    dominant_region = None

    # Mask out water (region 0) from the data
    masked_data = data.sel(band=region_band_number).where(data.sel(band=region_band_number) != 0)

    for region_id, region_name in regions.items():
        # Calculate the fraction of pixels for the current region, excluding water
        fraction_region = (masked_data == region_id).sum(skipna=True).item() / \
            masked_data.count().item()

        # Check if the fraction is greater than 90 percent
        if fraction_region > 0.9:
            dominant_region = region_name
            break  # Exit the loop once a region meeting the criteria is found

    # Skip this patch if no region with > 90 % of pixels belonging to it
    if dominant_region is None:
        return None
    
    # Drop the region variable from the data array:
    data = data.sel(band=data['band']!= region_band_number)

    output_path_patch = os.path.join(output_path, dominant_region, enmap_path.split(os.sep)[-2])
    os.makedirs(output_path_patch, exist_ok=True)

    # Clip the EnMAP bands data
    enmap_band_names = [data.band_descriptions[key] for key in enmap_bands]
    data.loc[dict(band=enmap_band_names)] = data.sel(band=enmap_band_names).clip(0, 10000)

    # Save data as GeoTIFF
    data.rio.to_raster(os.path.join(output_path_patch, enmap_path.split(os.sep)[-1]))

    out_ndarray = data.to_numpy().astype(np.float32)  # convert to float32

    # Change extension from .TIF to .npy and save ndarray
    np.save(os.path.join(output_path_patch, enmap_path.split(os.sep)[-1][:-3]+'npy'),
            out_ndarray)

try:
    # Load the script configuration
    patchify_tiles_path = os.path.join(ROOT_DIR, "config", "filter_select",
                                       "filter_select_patches.yaml")
    with open(patchify_tiles_path, "r") as f:
        yaml = YAML(typ="safe")
        filter_select_config = yaml.load(f)

    patches_csv = get_absolute_path(ROOT_DIR, filter_select_config["patches_csv"])
    num_workers = filter_select_config["num_workers"]
    output_path = get_absolute_path(ROOT_DIR, filter_select_config["output_path"])

    # Read the csv file created while patchifying (contains paths to each patch)
    patches_df = pd.read_csv(patches_csv)
    
    # Multiprocessing pool
    pool = multiprocessing.Pool(num_workers)

    # Prepare arguments for the process_tile function
    enmap_paths = patches_df['EnMap Patch Path']
    gedi_paths = patches_df['Gedi Patch Path']
    args = [(enmap_paths[i], gedi_paths[i], output_path) for i in range(len(enmap_paths))]

    # Process tiles with tqdm for progress tracking
    for _ in tqdm(pool.imap(filter_select_save, args), total=len(args),
                  desc="Processing patches/tiles"):
        pass

    # Close the pool
    pool.close()
    pool.join()

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.exception(e)
    print(f"An error occurred: {str(e)}")

finally:
    # Finalize script and logging
    end_t = stop_script(start_t, filepath)
