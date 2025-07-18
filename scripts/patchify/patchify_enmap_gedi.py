#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Aaron Banze (aaron.banze@dlr.de), Nassim Ait Ali Braham
# @License : (C)Copyright 2025, DLR
# @Desc    : Script to create non-overlapping patches from EnMAP and GEDI tiles

import csv
from datetime import datetime
import logging
from os import path, listdir

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from rtree import index
from ruamel.yaml import YAML
from shapely.geometry import box
from tqdm import tqdm

from hybiomass.dataset_creation.enmap import extract_datetime
from hybiomass.dataset_creation.patchify import build_tile_rtree, update_patch_metadata, save_patch
from hybiomass.dataset_creation.utils import get_root_dir, get_file_info, start_script, \
    stop_script, get_absolute_path


# Define and create directories
ROOT_DIR = get_root_dir()
filepath, filename = get_file_info()

# Initialize script and logging
start_t = start_script(ROOT_DIR, filepath, filename, log=True, level=logging.INFO)


def create_patches_from_tiles(enmap_tiles_paths_dict, gedi_tiles_paths_dict, output_path, csv_path,
                              patch_size=128, stride=16, overlap_threshold=0.01):
    """Creates non-overlapping patches from EnMAP and GEDI tiles.

    One global R-tree is not enough, because of inaccuracy when transforming CRS. 
    Instead, uses UTM zone specific R-trees to check for overlap between patches.
    Because there can be overlap between patches from different UTM zones, a
    global R-tree is additionally used.
    The tile-level R-tree is used to check if the global R-Tree is necessary
    for any given tile, or not, because there is no overlap on a tile level.
    Saves EnMAP and GEDI tiles as separate TIFF files in specific directories.
    Also creates a CSV file with metadata of created patches.

    Args:
    enmap_tiles_paths_dict (dict): Mapping tile datetime IDs to EnMAP tile paths.
    gedi_tiles_paths_dict (dict): Mapping tile datetime IDs to GEDI tile paths.
    output_path (str): The output directory where the patches are saved to.
    csv_path (str): Path to output CSV containing metadata about created patches.
    patch_size (int): Size of each patch in pixels (default: 128).
    stride (int): Stride (i.e. step size) between patches in pixels (default: 16).
    overlap_threshold (float): Fraction of patch area as max overlap (default: 0.01).
    """

    # Create a dict to collect an R-Tree index per UTM-zone
    rtree_indices_utmzones = dict()
    # Initialize global R-Tree index that will be used on patch level
    global_rtree_index = index.Index()
    # Build tile-level R-Tree
    tile_rtree = build_tile_rtree(enmap_tiles_paths_dict)
    logging.info("Created global and tile-level R-Trees.")

    tile_idx = 0
    for tile_datetime_id in tqdm(enmap_tiles_paths_dict.keys()):
        logging.info(f"Next tiles to process; datetime ID: {tile_datetime_id}")

        # Open the TIFF file using rasterio
        with rasterio.open(enmap_tiles_paths_dict[tile_datetime_id]) as src,\
             rasterio.open(gedi_tiles_paths_dict[tile_datetime_id]) as gedi_src,\
             rasterio.open(enmap_tiles_paths_dict[tile_datetime_id][:-19]+
                           '-QL_QUALITY_CLOUD.TIF') as cloud_src,\
             rasterio.open(enmap_tiles_paths_dict[tile_datetime_id][:-19]+
                           '-QL_QUALITY_SNOW.TIF') as snow_src:

            # String of the crs for the tile: e.g. "EPSG:32632"
            epsg_string = str(src.crs)

            # Check if there is an R-Tree for the EPSG of the tile
            if epsg_string not in rtree_indices_utmzones:
                # Create an R-Tree index for new EPSG and new entry to dict
                rtree_indices_utmzones[epsg_string] = index.Index()
                logging.info(f"Created rtree index for {epsg_string}.")

            transform_to_latlon = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)

            tile_bounds_latlon = transform_to_latlon.transform_bounds(*src.bounds)
            check_global = len(list(tile_rtree.intersection(tile_bounds_latlon, objects=True))) > 1

            # Calculate the number of patches along each dimension
            patch_id = 0
            for j in range(0, src.height, stride):
                for i in range(0, src.width, stride):
                    # Boundary issue, to not go past the height or width of the tile
                    if i + patch_size > src.width or j + patch_size > src.height:
                        continue

                    # Define the window to read
                    patch_window = Window(i, j, patch_size, patch_size)
                    window_bounds = rasterio.windows.bounds(patch_window, src.transform)
                    patch_polygon = box(*window_bounds)
                    patch_area = patch_polygon.area

                    max_overlap_area = 0
                    window_bounds_latlon = transform_to_latlon.transform_bounds(*window_bounds)
                    patch_polygon_latlon = box(*window_bounds_latlon)
                    patch_area_latlon = patch_polygon_latlon.area

                    # Check overlap in the UTM zone specific R-tree
                    for item in rtree_indices_utmzones[epsg_string].intersection(window_bounds, objects=True):
                        overlap_box = box(*item.bbox)
                        max_overlap_area = max(max_overlap_area, patch_polygon.intersection(overlap_box).area)
                    # Skip this patch, if the fraction of overlap area is above threshold
                    if max_overlap_area / patch_area > overlap_threshold:
                        continue

                    # Reset the variable to zero, as value before was in m^2
                    max_overlap_area = 0

                    # Check global R-tree for patches from different UTM zones if tile level overlap was found
                    if check_global:
                        for item in global_rtree_index.intersection(window_bounds_latlon, objects=True):
                            if item.object != epsg_string:
                                overlap_box = box(*item.bbox)
                                max_overlap_area = max(max_overlap_area,
                                                       patch_polygon_latlon.intersection(overlap_box).area)

                        if max_overlap_area / patch_area_latlon > overlap_threshold:
                            continue

                    # Read the patch for all bands
                    try:
                        enmap_patch_data = src.read(window=patch_window)
                    except rasterio.errors.RasterioIOError as e:
                        logging.error(f"Failed to read EnMAP patch for {tile_datetime_id} at ({i}, {j}: {e})")
                        continue
                    
                    # Skip this patch if it contains no data values in band one
                    if np.any(enmap_patch_data[0, :, :] == src.nodata):
                        continue

                    cloud_patch_data = cloud_src.read(window=patch_window)
                    snow_patch_data = snow_src.read(window=patch_window)

                    # Calculate the percentage of clouds/snow in patch
                    cloud_perc = np.count_nonzero(cloud_patch_data == 1) / cloud_patch_data.size * 100
                    snow_perc = np.count_nonzero(snow_patch_data == 1) / snow_patch_data.size * 100

                    if cloud_perc > 1 or snow_perc > 10:
                        continue

                    # Read the GEDI patch data
                    try:
                        gedi_patch_data = gedi_src.read(window=patch_window)
                    except rasterio.errors.RasterioIOError as e:
                        logging.error(f"Failed to read GEDI patch for {tile_datetime_id} at ({i}, {j}): {e}")
                        continue

                    # Update the EnMAP and GEDI patch metadata
                    patch_meta_enmap, tags_enmap = update_patch_metadata(src, patch_size, patch_window, window_bounds)
                    patch_meta_gedi, tags_gedi = update_patch_metadata(gedi_src, patch_size, patch_window, window_bounds)

                    # Save the EnMAP patch as a new TIFF file:
                    enmap_patch_path = save_patch(output_path, enmap_patch_data, patch_meta_enmap, tags_enmap,
                                                  tile_idx, patch_id, subdir='enmap_patches')
                    gedi_patch_path = save_patch(output_path, gedi_patch_data, patch_meta_gedi, tags_gedi,
                                                 tile_idx, patch_id, subdir='gedi_patches')

                    with open(csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([tile_idx, f"{patch_id}", i, j, tile_datetime_id, enmap_patch_path,
                                         gedi_patch_path])

                    # Mark patch as used by inserting the bbox into the R-Tree index with unique id
                    unique_id = str(tile_idx).zfill(4) + str(patch_id).zfill(3)
                    rtree_indices_utmzones[epsg_string].insert(int(unique_id), window_bounds)
                    global_rtree_index.insert(int(unique_id), window_bounds_latlon)
                    patch_id += 1

        logging.info(f"Created {patch_id} patches.")
        tile_idx += 1


try:
    # Load the script configuration
    patchify_tiles_path = path.join(ROOT_DIR, "config", "patchify", "patchify_tiles.yaml")
    with open(patchify_tiles_path, "r") as f:
        yaml = YAML(typ="safe")
        patchify_config = yaml.load(f)

    from_enmap_archive = patchify_config["from_enmap_archive"]
    enmap_tiles_path = get_absolute_path(ROOT_DIR, patchify_config["enmap_tiles_path"])
    gedi_tiles_path = get_absolute_path(ROOT_DIR, patchify_config["gedi_tiles_path"])
    output_path = get_absolute_path(ROOT_DIR, patchify_config["output_path"])
    csv_path = get_absolute_path(ROOT_DIR, patchify_config["csv_path"])
    
    if from_enmap_archive:
        # Load the CSV file with paths to EnMAP tiles in EnMAP archive into a DataFrame
        df = pd.read_csv(path.join(ROOT_DIR, 'raw_data', 'enmap_tiles_with_index.csv'))
        enmap_tiles_paths = list(df['File Path'])
        # Create dict with extracted datetime ids as keys and enmap tile paths as values
        enmap_tiles_paths_dict = {extract_datetime(enmap_tile_path.split('/')[-1]):enmap_tile_path
                                  for enmap_tile_path in enmap_tiles_paths}

    elif not from_enmap_archive:
        # Create list of absolute paths for all EnMAP tiles
        enmap_tiles_paths = [path.abspath(path.join(enmap_tiles_path, f, f))
                             for f in listdir(enmap_tiles_path)
                             if path.isdir(path.join(enmap_tiles_path, f))]
        # Create dict with datetime ids as keys and EnMAP tile paths as values
        enmap_tiles_paths_dict = {extract_datetime(enmap_tile_path.split('/')[-1]):enmap_tile_path
                                  for enmap_tile_path in enmap_tiles_paths}
    else:
        raise ValueError(f"Unexpected string '{from_enmap_archive}' for '\
                         'variable in config file 'from_enmap_archive'.")
    
    # Create dict with datetime IDs as keys and gedi tile paths as values
    gedi_tiles_paths_dict = {extract_datetime(gedi_tile_fn):path.join(gedi_tiles_path, gedi_tile_fn)
                             for gedi_tile_fn in listdir(gedi_tiles_path)}

    # Filter EnMAP archive for GEDI & Europe tiles based on datetime ids
    enmap_tiles_paths_dict = {k: v for k, v in enmap_tiles_paths_dict.items()
                              if k in gedi_tiles_paths_dict.keys()}

    # Sort keys in dicts based on ascending datetime strings
    # to reduce the time-delta between GEDI shots and EnMAP scenes
    sorted_keys = sorted(enmap_tiles_paths_dict.keys(),
                         key=lambda key: datetime.strptime(key[:8], "%Y%m%d"))
    enmap_tiles_paths_dict = {key: enmap_tiles_paths_dict[key] for key in sorted_keys}
    gedi_tiles_paths_dict = {key: gedi_tiles_paths_dict[key] for key in sorted_keys}

    patch_size = 128  # Patch size in pixels
    stride = 16  # Stride in Pixels
    overlap_threshold = 0.01  # Maximum allowed overlap area as a fraction of patch area

    # Initialize CSV file with headers
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Tile Index", "Patch ID", "Column Offset", "Row Offset",
                         "Datetime ID", "EnMap Patch Path", "Gedi Patch Path"])

    create_patches_from_tiles(enmap_tiles_paths_dict, gedi_tiles_paths_dict, output_path, csv_path,
                              patch_size, stride, overlap_threshold)

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.exception(e)
    print(f"An error occurred: {str(e)}")

# Finalize script and logging
finally:
    end_t = stop_script(start_t, filepath)
