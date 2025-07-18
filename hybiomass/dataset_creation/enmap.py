#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Functions to process EnMAP data

import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def list_enmap_tiles(enmap_dir):
    tiles_list = os.listdir(os.path.join(enmap_dir))
    return tiles_list


def extract_datetime(enmap_tile):
    """
    Extracts the part of the string that matches the pattern '_YYYYMMDDTHHMMSSZ_'.

    Args:
    s (str): The input string.

    Returns:
    str: The extracted part if found, otherwise None.
    """
    # Regex pattern to match the desired part
    pattern = r"_(\d{8}T\d{6}Z)_"

    # Search for the pattern in the string
    match = re.search(pattern, enmap_tile)

    # Extract the matched part if found
    if match:
        datetime_part = match.group(1)
        return datetime_part
    else:
        return None


def format_timestamp(dt):
    """
    Formats a timestamp from ISO 8601 format to a custom string format used as tile identifier.

    This function converts a timestamp in ISO 8601 format (`"%Y-%m-%dT%H:%M:%S.%fZ"`)
    into a more compact format (`"%Y%m%dT%H%M%SZ"`).

    Args:
        dt (str): The timestamp as a string in ISO 8601 format, e.g., `"2024-07-29T12:34:56.789Z"`.

    Returns:
        str: The formatted timestamp as a string in the format `"%Y%m%dT%H%M%SZ"`, e.g., `"20240729T123456Z"`.
    """
    return datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y%m%dT%H%M%SZ")


def plot_rgb(ds):
    """
    Plots an RGB composite image from the given EnMAP tile loaded as a xarray DataArray.

    This function creates an RGB composite image from the given EnMAP tile loaded as a xarray DataArray.

    Args:
        ds (xarray.DataArray): The input DataArray containing the bands for Red, Green, and Blue channels.

    Returns:
        None
    """
    # Filter out _fillValue and negative values
    data = ds.where((ds != -32768) & (ds >= 0))
    # Select the specific bands for R, G, B
    red_band = data.sel(band=48)  # red
    green_band = data.sel(band=30)  # green
    blue_band = data.sel(band=16)  # blue

    red_band_norm = 7 * (red_band - red_band.min()) / (red_band.max() - red_band.min())
    green_band_norm = (
        7 * (green_band - green_band.min()) / (green_band.max() - green_band.min())
    )
    blue_band_norm = (
        7 * (blue_band - blue_band.min()) / (blue_band.max() - blue_band.min())
    )

    rgb = np.stack([red_band_norm, green_band_norm, blue_band_norm], axis=-1)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("RGB True Composite Image")
    plt.axis("off")
    plt.show()


def plot_false_rbg(ds):
    """
    Plots a false-color RGB composite image from the given EnMAP tile loaded as a xarray DataArray.

    This function creates a false-color RGB composite image from the given EnMAP tile loaded as a xarray DataArray.

    Args:
        ds (xarray.DataArray): The input DataArray containing the bands for NIR, Red, and Green channels.

    Returns:
        None
    """
    # Filter out _fillValue and negative values
    data = ds.where(ds >= 0)
    # Select the specific bands for NIR, R, G visualization
    nir_band = data.sel(band=60)  # nir
    red_band = data.sel(band=48)  # red
    green_band = data.sel(band=30)  # green

    nir_band_norm = 7 * (nir_band - nir_band.min()) / (nir_band.max() - nir_band.min())
    red_band_norm = 7 * (red_band - red_band.min()) / (red_band.max() - red_band.min())
    green_band_norm = (
        7 * (green_band - green_band.min()) / (green_band.max() - green_band.min())
    )

    rgb = np.stack([nir_band_norm, red_band_norm, green_band_norm], axis=-1)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("RGB False Composite Image")
    plt.axis("off")
    plt.show()
