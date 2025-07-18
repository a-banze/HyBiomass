#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Script to append Hansen Global Forest Change 2023 (v1.11) data
#            to the GEDI L2L4A data tables

# %%
# # ----- Imports
from os import path, makedirs, listdir
from hybiomass.dataset_creation.utils import get_root_dir, get_file_info, start_script, stop_script
import logging
from tqdm import tqdm
from ruamel.yaml import YAML
from hybiomass.dataset_creation.hansen import HansenProcessor
import multiprocessing


# %%
# # ----- Define and create directories
ROOT_DIR = get_root_dir()
filepath, filename = get_file_info()

# %%
# # ----- Initialize script and logging
start_t = start_script(ROOT_DIR, filepath, filename, log=True, level=logging.INFO)

# %%
# # ----- Define the function to process a single tile
def process_table(args):
    """
    Function for the multiprocessing pool to process a single table
    """
    l2l4a_dir, table_name, hansen_dir, products, output_dir = args
    
    processor = HansenProcessor(l2l4a_dir, table_name, hansen_dir, products)
    processor.append_hansen_to_table()
    processor.save_output(output_dir)

# %%
# # ----- Run code
try:
    ## Load the script configuration
    processor_config_path = path.join(ROOT_DIR, "config", "hansen", "processor.yaml")
    with open(processor_config_path, "r") as f:
        yaml = YAML(typ="safe")
        processor_config = yaml.load(f)
    products = processor_config["products"]
    hansen_dir = processor_config["hansen_dir"]
    l2l4a_dir = processor_config["l2l4a_dir"]
    output_dir = processor_config["output_dir"]
    if not path.exists(output_dir):
        makedirs(output_dir)
    num_workers = processor_config["num_workers"]

    ## List input tables
    tables = listdir(l2l4a_dir)
    tables = [table for table in tables if table.endswith(".zip")]
    logging.info(f"Found {len(tables)} gedi l2l4a tables in the input directory {l2l4a_dir}")

    ## List existing output table
    existing_tables = listdir(output_dir)
    existing_tables = [table for table in existing_tables if table.endswith(".zip")]
    logging.info(f"Found {len(existing_tables)} existing tables in the output directory {output_dir}") 
    
    ## Filter out tables that have already been processed
    tables = [table for table in tables if table not in existing_tables]
    logging.info(f"Processing {len(tables)} l2l4a tables...")

    ## Multiprocessing pool
    pool = multiprocessing.Pool(num_workers)
    
    ## Prepare the arguments for the pool
    args = [(l2l4a_dir, table_name, hansen_dir, products, output_dir) for table_name in tables]

    ## Process tiles with tqdm for progress tracking
    for _ in tqdm(pool.imap(process_table, args), total=len(args), desc="Processing tables"):
        pass

    ## Close the pool
    pool.close()
    pool.join()

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    print(f"An error occurred: {str(e)}")
# %%
# # ----- Finalize script and logging
finally:
    end_t = stop_script(start_t, filepath)
