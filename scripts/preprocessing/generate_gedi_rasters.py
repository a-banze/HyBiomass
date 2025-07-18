#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Script to create sparse rasters of GEDI L2L4A data matching the
#            EnMAP tiles spatial dimensions.

# %%
# # ----- Imports
from tqdm import tqdm
import multiprocessing
from os import path, listdir
import logging

import pandas as pd
import rasterio as rio
import rioxarray as rxr
from ruamel.yaml import YAML

from hybiomass.dataset_creation.raster import SparseRasterCreator
from hybiomass.dataset_creation.enmap import extract_datetime
from hybiomass.dataset_creation.gedi import timestamp_to_datetime, add_delta_time
from hybiomass.dataset_creation.utils import get_root_dir, get_file_info, start_script, stop_script, get_absolute_path

# %%
# # ----- Define and create directories
ROOT_DIR = get_root_dir()
filepath, filename = get_file_info()

# %%
# # ----- Initialize script and logging
start_t = start_script(ROOT_DIR, filepath, filename, log=True, level=logging.INFO)

# %%
## Define the function to process a single tile
def process_tile(args):
    id, gedi_extracted_csvs_dir, enmap_tiles_paths_dict, gedi_tiles_output_dir, data_vars = args
    """
    Processes a single tile by reading the corresponding table, updating the raster with data from the table, 
    and saving the updated raster to a specified output path.

    This function performs the following steps:
    1. Constructs file paths for the table and tile based on the provided `id`.
    2. Reads the table data into a DataFrame and preprocesses it by adding delta time and sorting.
    3. Opens the raster tile using rasterio.
    4. Checks if the table is empty and logs a warning if it is.
    5. Creates an instance of `SparseRasterCreator` with the raster and table data.
    6. Writes data from the table to the raster.
    7. Saves the updated raster to an output file.
    
    Args:
        id (int): The timestamp_id used as unique identifier to match tiles and tables.
        gedi_extracted_csvs_dir (str): The directory where table CSV files are stored.
        enmap_tiles_paths_dict (dict): Keys: Timstamps as id; Values: Paths to corresponding EnMAP tiles.
        gedi_tiles_output_dir (str): The directory where GEDI rasters are saved to.
        data_vars (list of str): List of data variables to be used from the table.

    Returns:
        SparseRasterCreator: An instance of `SparseRasterCreator` containing the updated raster data.

    Notes:
        - The output raster is saved with a filename that includes "-GEDI.TIF" suffix in the same directory as the input tile.
    """
    table_path = path.join(gedi_extracted_csvs_dir, tables[id])
    table = pd.read_csv(table_path)
    
    if table.empty:
        logging.warning(f"Table {table_path} is empty.")
        return None
    
    else:
        dt = timestamp_to_datetime(id)
        table = add_delta_time(table, dt)
        table.sort_values("abs_delta_t_days", inplace=True)
        tile = rxr.open_rasterio(enmap_tiles_paths_dict[id] + "-SPECTRAL_IMAGE.TIF")
        creator = SparseRasterCreator(tile, table, data_vars, id)
        creator.write_data()
        
        output_path = path.join(gedi_tiles_output_dir, enmap_tiles_paths_dict[id].split('/')[-1] + "-GEDI.TIF")
        creator.save_raster(output_path)
        return creator


# %%
# # ----- Run code
try:
    ## Load the script configuration
    generate_gedi_rasters_config_path = path.join(ROOT_DIR, "config", "gedi", "generate_gedi_rasters.yaml")
    with open(generate_gedi_rasters_config_path, "r") as f:
        yaml = YAML(typ="safe")
        generate_gedi_rasters_config = yaml.load(f)
    enmap_dir = get_absolute_path(ROOT_DIR, generate_gedi_rasters_config["enmap_dir"])
    from_enmap_archive = generate_gedi_rasters_config["from_enmap_archive"]
    gedi_extracted_csvs_dir = get_absolute_path(ROOT_DIR, generate_gedi_rasters_config["gedi_extracted_csvs_dir"])
    gedi_tiles_output_dir = get_absolute_path(ROOT_DIR, generate_gedi_rasters_config["gedi_tiles_output_dir"])
    num_workers = generate_gedi_rasters_config["num_workers"]

    ## Read the data_vars from gedi_cube_config.yaml
    gedi_cube_config_path = path.join(ROOT_DIR, "config", "gedi", "gedi_cube_config.yaml")
    with open(path.join(gedi_cube_config_path), "r") as file:
        yaml = YAML(typ="safe")
        gedi_cube_config = yaml.load(file)
    data_vars = gedi_cube_config["data_vars"]

    if from_enmap_archive:
        ## Load the CSV file with paths to EnMAP tiles in EnMAP archive into a DataFrame
        df = pd.read_csv(path.join(ROOT_DIR, "raw_data", "enmap_tiles_with_index.csv"))
        enmap_tiles_paths = list(df['File Path'])
        ## Create dict with extracted datetimes as keys and enmap tile paths as values; [:-19] to remove '.SEPCTRAL-TIF' extension
        enmap_tiles_paths_dict = {extract_datetime(enmap_tile_path.split('/')[-1]):enmap_tile_path[:-19] for enmap_tile_path in enmap_tiles_paths}

    elif not from_enmap_archive:
        ## Create list of absolute paths for all EnMAP tiles
        enmap_tiles_paths = [path.abspath(path.join(enmap_dir, f, f)) for f in listdir(enmap_dir) if path.isdir(path.join(enmap_dir, f))]
        ## Create dict with datetimes as keys and EnMAP tile paths as values
        enmap_tiles_paths_dict = {extract_datetime(enmap_tile_path.split('/')[-1]):enmap_tile_path for enmap_tile_path in enmap_tiles_paths}
    else:
        raise ValueError(f"Unexpected string value '{from_enmap_archive}' for variable in config file 'from_enmap_archive'.")

    ## List the matching tiles and tables to process (intersection of both minus existing files)
    logging.info("Listing the tiles and tables to process...")
    tables = {ta[:16]:ta for ta in listdir(gedi_extracted_csvs_dir)}
  
    logging.info(f"Found {len(enmap_tiles_paths_dict)} tiles")
    ids = list(set.intersection(set(enmap_tiles_paths_dict.keys()), set(tables.keys())))
    logging.info(f"Found {len(ids)} matching tables and tiles")
    logging.info(f"Looking for existing data...")
    ids = [id for id in ids if not path.exists(path.join(gedi_tiles_output_dir, enmap_tiles_paths_dict[id].split('/')[-1] + "-GEDI.TIF"))]
    logging.info(f"Found {len(ids)} tile-table matches still to process.")
    
    ## Multiprocessing pool
    pool = multiprocessing.Pool(num_workers)

    ## Prepare arguments for the process_tile function
    args = [(id, gedi_extracted_csvs_dir, enmap_tiles_paths_dict, gedi_tiles_output_dir, data_vars) for id in ids]

    ## Process tiles with tqdm for progress tracking
    for _ in tqdm(pool.imap(process_tile, args), total=len(args), desc="Processing tiles"):
        pass

    ## Close the pool
    pool.close()
    pool.join()
    
except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.exception(e)
    print(f"An error occurred: {str(e)}")
# %%
# # ----- Finalize script and logging
finally:
    end_t = stop_script(start_t, filepath)
