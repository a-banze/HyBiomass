#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz-potsdam.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Script to prepare the table of EnMAP tiles, including coordinates
#            reprojection and timestamp formatting

# %%
# # ----- Imports
import pandas as pd
import geopandas as gpd
from hybiomass.utils import reproject_bbox
from hybiomass.dataset_creation.enmap import format_timestamp
from os import path
from hybiomass.utils import (
    get_root_dir,
    get_file_info,
    start_script,
    stop_script,
    bbox_to_geojson_feature,
)
import os
import geojson
import numpy as np
import logging
from ruamel.yaml import YAML

# %%
# # ----- Define and create directories
ROOT_DIR = get_root_dir()
filepath, filename = get_file_info()

# %%
# # ----- Initialize script and logging
start_t = start_script(ROOT_DIR, filepath, filename, log=True, level=logging.DEBUG)

# %%
# # ----- Run code
try:
    ## Load the script configuration
    enmap_table_config_path = path.join(ROOT_DIR, "config", "enmap", "enmap_table.yaml")
    with open(enmap_table_config_path, "r") as f:
        yaml = YAML(typ="safe")
        enmap_table_config = yaml.load(f)

    enmap_dir = os.path.join(ROOT_DIR, enmap_table_config["enmap_dir"])
    input_file_path = os.path.join(enmap_dir, enmap_table_config["input_filename"])
    output_dir = os.path.join(ROOT_DIR, enmap_table_config["output_dir"])
    output_file_path = os.path.join(output_dir, enmap_table_config["output_filename"])

    ## Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    ## Format the Timestamp as unique tile identifier
    df["timestamp_id"] = df["Timestamp"].apply(format_timestamp)

    ## Apply the reproject function to each row in the DataFrame
    out_crs = "EPSG:4326"
    reprojected_coords = df.apply(
        lambda row: reproject_bbox(
            row["Min X"], row["Min Y"], row["Max X"], row["Max Y"], row["CRS"], out_crs
        ),
        axis=1,
    )

    ## Split the reprojected coordinates into separate columns
    df[["min_lon", "min_lat", "max_lon", "max_lat"]] = pd.DataFrame(
        reprojected_coords.tolist(), index=df.index
    )

    ## Label the EnMAP tiles that are within the bounding box of Europe
    europe_geojson = os.path.join(ROOT_DIR, "raw_data", "europe.geojson")
    europe = gpd.read_file(europe_geojson)
    eu_bbox = europe.geometry.total_bounds
    eu_min_lon, eu_min_lat, eu_max_lon, eu_max_lat = eu_bbox

    EUR_condition = (
        (df["min_lat"] >= eu_min_lat)
        & (df["min_lon"] >= eu_min_lon)
        & (df["max_lat"] <= eu_max_lat)
        & (df["max_lon"] <= eu_max_lon)
    )
    df["region"] = np.where(EUR_condition, "EUR", "Non-EUR")

    # %%
    ## Label the EnMAP tiles that are within the GEDI footprint
    GEDI_condition = (df["min_lat"] >= -51.6) & (df["max_lat"] <= 51.6)
    df["gedi"] = np.where(GEDI_condition, "GEDI", "Non-GEDI")

    ## Drop the original coordinate columns if desired
    # df.drop(['Min X', 'Min Y', 'Max X', 'Max Y'], axis=1, inplace=True)

    ## Save the DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

    logging.info(
        f"Reprojection complete. Results ({len(df)} entries) saved to: {output_file_path}"
    )
    print(
        f"Reprojection complete. Results ({len(df)} entries) saved to:",
        output_file_path,
    )
    ## Create GeoJSON FeatureCollection
    bounding_boxes_filename = "bounding_boxes.geojson"
    bounding_boxes_filepath = os.path.join(output_dir, bounding_boxes_filename)

    features = [
        bbox_to_geojson_feature(
            row["min_lat"], row["min_lon"], row["max_lat"], row["max_lon"]
        )
        for index, row in df.iterrows()
    ]
    feature_collection = geojson.FeatureCollection(features)

    # Save to GeoJSON file
    with open(bounding_boxes_filepath, "w") as f:
        geojson.dump(feature_collection, f)

    logging.info(
        f"GeoJSON complete FeatureCollection saved to: {bounding_boxes_filepath}"
    )
    print(f"GeoJSON complete FeatureCollection saved to:", bounding_boxes_filepath)
    # %%

except Exception as e:
    print(f"An error occurred: {str(e)}")
# %%
# # ----- Finalize script and logging
finally:
    end_t = stop_script(start_t, filepath)
