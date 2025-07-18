#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz-potsdam.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Class to find tabular data entries within a bounding box

import numpy as np
import re
import os
import pandas as pd
from datetime import datetime
import logging


class Finder:
    """
    A class to find tabular data entries within a specified bounding box.

    This class provides functionality to filter tabular data based on a bounding box".

    Attributes:
        data_dir (str): The directory path where the tabular data is stored.
        bbox (tuple): A tuple of four floats representing the bounding box coordinates
                      in the format (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        timestamp (str, optional): An optional timestamp associated with the EnMAP tile.
        min_lon (float): Minimum longitude of the bounding box.
        min_lat (float): Minimum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.

    Args:
        data_dir (str): The directory containing the tabular data.
        bbox (tuple): The bounding box coordinates (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        timestamp (str, optional): The timestamp of the EnMAP tile. Defaults to None.
    """

    def __init__(self, data_dir, bbox, timestamp=None):
        """
        Initializes the Finder with the specified data directory, bounding box, and optional timestamp.

        Args:
            data_dir (str): The directory containing the tabular data.
            bbox (tuple): The bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
            timestamp (str, optional): The timestamp of the EnMAP tile. Defaults to None.
        """
        self.data_dir = data_dir
        self.bbox = bbox
        self.timestamp = timestamp
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.bbox
        logging.info(f"Initializing a data Finder for EnMAP tile {self.timestamp}...")

    def filter_entries(self, table):
        """
        Filters entries in a tabular file based on the specified bounding box.

        This method loads a CSV file containing tabular data, filters the entries
        based on the bounding box coordinates, and returns a DataFrame with only
        the entries within the bounding box.

        Args:
            table (str): The name of the CSV file containing the table to be filtered.
                        The file should be located in the directory specified by `self.data_dir`.

        Returns:
            pd.DataFrame: A DataFrame containing only the entries that fall within the bounding box.

        Raises:
            FileNotFoundError: If the specified file does not exist in the directory.
            KeyError: If the required columns ('lon_lm_a0' or 'lat_lm_a0') are not found in the table.
        """
        # Construct the file path
        file_path = os.path.join(self.data_dir, table)

        # Load the table as a DataFrame
        try:
            table_df = pd.read_csv(file_path)
        except FileNotFoundError:
            logging.error(f"The file {file_path} does not exist.")
            raise

        # Check for required columns
        if not all(col in table_df.columns for col in ["lon_lm_a0", "lat_lm_a0"]):
            missing_cols = [
                col for col in ["lon_lm_a0", "lat_lm_a0"] if col not in table_df.columns
            ]
            logging.error(f"Missing required columns: {', '.join(missing_cols)}")
            raise KeyError(f"Missing required columns: {', '.join(missing_cols)}")

        # Filter the entries based on the bounding box
        filtered = table_df[
            (table_df["lon_lm_a0"] >= self.min_lon)
            & (table_df["lon_lm_a0"] <= self.max_lon)
            & (table_df["lat_lm_a0"] >= self.min_lat)
            & (table_df["lat_lm_a0"] <= self.max_lat)
        ]

        return filtered

    def filter_all_entries(self):
        """
        Filters entries from all tables based on the specified bounding box and combines them into a single DataFrame.

        This method iterates over each table listed in `self.tables`, applies the `filter_entries` method to filter
        the data based on the bounding box, and then concatenates the filtered results from all tables into a single
        DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all filtered entries from the specified tables that fall within the bounding box.
        """
        all_filtered = []

        # Filter entries for each table
        for table in self.filtered_tables:
            filtered = self.filter_entries(table)
            all_filtered.append(filtered)

        # Concatenate all filtered entries into a single DataFrame
        all_filtered_df = pd.concat(all_filtered, ignore_index=True)

        return all_filtered_df

    def get_entries(self):
        """
        Retrieves and filters all entries based on the specified bounding box.

        This method calls `filter_all_entries` to get the filtered entries and assigns them
        to `self.all_filtered_entries`. It logs the process and handles any unexpected errors
        that may occur during filtering.

        Returns:
            None
        """
        logging.info(f"Finding entries matching EnMAP tile {self.timestamp}...")
        try:
            self.all_filtered_entries = self.filter_all_entries()
        except:
            logging.error(
                f"Could not find entries matching EnMAP tile {self.timestamp}."
            )

    def save_entries(self, output_path):
        """
        Saves the filtered entries to a CSV file.

        This method attempts to save the `self.all_filtered_entries` DataFrame to the specified
        file path as a CSV file. It logs the success or failure of the operation.

        Args:
            output_path (str): The path where the CSV file will be saved.

        Returns:
            None
        """
        try:
            self.all_filtered_entries.to_csv(output_path, index=False)
            logging.info(
                f"Saved entries matching EnMAP tile {self.timestamp} to {output_path}"
            )
        except:
            logging.error(
                f"Could not save entries matching EnMAP tile {self.timestamp} to {output_path}"
            )

    def get_and_save_entries(self, output_path):
        """
        Retrieves filtered entries and saves them to a CSV file.

        This method first retrieves filtered entries by calling `get_entries()`, and then
        saves the resulting entries to the specified CSV file using `save_entries(output_path)`.

        Args:
            output_path (str): The path where the CSV file will be saved.

        Returns:
            None
        """
        self.get_entries()
        self.save_entries(output_path)


class GediShotsFinder(Finder):
    """
    A specialized Finder class for finding GEDI shots from the L2L4A archive (Burns et al.) within a bounding box.

    This class extends the `Finder` class to specifically handle GEDI shots. It initializes
    the necessary parameters, such as the directory containing GEDI shot tables and the
    bounding box for filtering. Additionally, it calculates latitude and longitude ranges
    based on the bounding box coordinates.

    Args:
        data_dir (str): The directory containing the GEDI shot tables.
        bbox (tuple): A tuple of four floats representing the bounding box coordinates
                      in the format (min_lon, min_lat, max_lon, max_lat).
        timestamp (str, optional): The timestamp of the GEDI shot data. Defaults to None.

    Attributes:
        tables (list): A list of table file names containing GEDI shots, obtained from
                       `list_gedi_shots_tables(data_dir)`.
        lat_range (tuple): A tuple of two integers representing the range of latitudes,
                           calculated as (floor(min_lat), ceil(max_lat)).
        lon_range (tuple): A tuple of two integers representing the range of longitudes,
                           calculated as (floor(min_lon), ceil(max_lon)).
    """

    def __init__(self, data_dir, bbox, timestamp=None):
        """
        Initializes the GediShotsFinder with the specified data directory, bounding box, and optional timestamp.

        This constructor initializes the base `Finder` class, calculates latitude and longitude
        ranges, and retrieves the list of GEDI shot tables.

        Args:
            data_dir (str): The directory containing the GEDI shot tables.
            bbox (tuple): The bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
            timestamp (str, optional): The timestamp of the GEDI shot data. Defaults to None.
        """
        super().__init__(data_dir, bbox, timestamp)
        self.tables = list_gedi_shots_tables(data_dir)
        self.lat_range = (int(np.floor(self.min_lat)), int(np.ceil(self.max_lat)))
        self.lon_range = (int(np.floor(self.min_lon)), int(np.ceil(self.max_lon)))
        self.filtered_tables = self.filter_gedi_tables()

    def filter_gedi_tables(self):
        """
        Filters GEDI data tables based on their geographic coordinates and the specified latitude and longitude ranges.

        This method iterates over the list of GEDI data tables, extracts the geographic coordinates from each table's name
        using a regular expression, adjusts these coordinates based on their directional indicators, and checks if they
        fall within the specified latitude and longitude ranges. Only the tables that meet these criteria are included
        in the filtered list.

        Returns:
            list: A list of table names that intersect with the specified latitude and longitude ranges.

        Notes:
            - The regular expression `r"g(\d{3})([ew])(\d{2})([ns])"` is used to parse the table names to extract
            longitude and latitude degrees along with their directional indicators.
            - Longitude and latitude are adjusted according to their directional indicators:
                - Longitude: "e" (east) remains positive, "w" (west) is converted to negative.
                - Latitude: "n" (north) remains positive, "s" (south) is converted to negative.
            - The filtered tables are those that have coordinates within the specified `lat_range` and `lon_range`.
        """
        filtered_tables = []

        for table in self.tables:
            match = re.search(r"g(\d{3})([ew])(\d{2})([ns])", table)
            # Regular Expression: The regular expression r'g(\d{3})([ew])(\d{2})([ns])' is used to extract the
            # longitude and latitude values along with their directions (E/W for longitude and N/S for latitude).
            if match:
                lon_deg, lon_dir, lat_deg, lat_dir = match.groups()
                lon_deg = int(lon_deg)
                lat_deg = int(lat_deg)

                # Direction Adjustment: Depending on the direction (east or west for longitude, north or south for latitude),
                # the extracted values are converted to their corresponding positive or negative values.
                if lon_dir == "w":
                    lon_deg = -lon_deg
                if lat_dir == "s":
                    lat_deg = -lat_deg

                # Range Checking: The script checks if the extracted latitude and longitude values fall within the specified ranges.
                if (
                    self.lat_range[0] <= lat_deg < self.lat_range[1]
                    and self.lon_range[0] <= lon_deg < self.lon_range[1]
                ):
                    filtered_tables.append(table)
        logging.info(
            f"Found {len(filtered_tables)} GEDI data table(s) intersecting EnMAP tile {self.timestamp}"
        )
        return filtered_tables


class NfiPlotsFinder(Finder):
    ### TO DO ####
    def __init__(self, data_dir, bbox, timestamp=None):
        super().__init__(data_dir, bbox, timestamp)
        self.tables = list_nfi_plots_tables(data_dir)

    def filter_nfi_tables(self):
        # Filter the tables (i.e. countries) intersecting the bounding box
        pass


def list_gedi_shots_tables(data_dir):
    """
    Lists the GEDI L2L4A shot tables from Burns et al. available in a specified directory.

    This function scans the specified directory for files that match the GEDI shot table naming pattern and have a `.zip` extension.
    It returns a list of filenames that meet these criteria.

    Args:
        data_dir (str): The directory path to scan for GEDI shot table files.

    Returns:
        list: A list of filenames (as strings) that end with ".zip" and start with "gediv002_l2l4a_va_".
    """
    tables = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".zip") and f.startswith("gediv002_l2l4a_va_")
    ]
    return tables


def list_nfi_plots_tables(data_dir):
    ### TO DO ###
    tables = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and f.startswith("nfi_agb_")
    ]
    return tables
