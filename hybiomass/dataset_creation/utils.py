#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : General utilities for the hyperamplifai package

from os import path
from pyproj import Transformer, CRS
import logging
from pathlib import Path
import torch
import psutil
import datetime
import inspect
import geojson


def get_root_dir(verbose=True):
    # SET YOUR ROOT DIR HERE
    root_dir = ""
    if verbose:
        print(f"ROOT_DIR set to: {root_dir}")
    return root_dir


def get_absolute_path(root_path, rel_or_abs_path):
    # Make sure that the returned path is absolute, when given "rel_or_abs_path" can be both relative or absolute
    return rel_or_abs_path if path.isabs(rel_or_abs_path) else path.join(root_path, rel_or_abs_path)


def get_sysinfo():
    """Get the GPU availability, and the number of CPUs and RAM."""
    return f'cuda: {torch.cuda.is_available()}, cpu: {psutil.cpu_count()}, ram: {str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"}'


def get_file_info():
    """Get the file path and name of the caller"""
    # Get the stack frame of the caller
    frame = inspect.stack()[1]
    # Get the file path of the caller
    caller_filepath = frame.filename
    # Create a Path object from the file path
    filepath = Path(caller_filepath)
    # Extract the filename without the extension
    filename = filepath.stem
    return filepath, filename


def get_time(print_to_console=True):
    """Get the current time"""
    time = datetime.datetime.now()
    if print_to_console:
        print(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    return time


def start_script(ROOT_DIR="", filepath="", filename="", log=True, level=logging.DEBUG):
    if not ROOT_DIR:
        ROOT_DIR = get_root_dir()
    if not filepath or not filename:
        filepath, filename = get_file_info()
    if log:
        log_dir = path.join(ROOT_DIR, "logs")
        logging.basicConfig(
            filename=path.join(log_dir, filename + ".log"),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z",
            level=level,
        )
        logging.info(f"==/\/\/\== START of the {filepath.name} script ==/\/\/\==")
        logging.info(f"system details: {get_sysinfo()}")
        logging.info(f"ROOT_DIR: {ROOT_DIR}")
    print(f"==/\/\/\== START of the {filepath.name} script ==/\/\/\==")
    print(f"system details: {get_sysinfo()}")
    start_t = get_time()
    return start_t


def stop_script(start_t, filepath="", log=True):
    if not filepath:
        filepath, filename = get_file_info()
    end_t = get_time()
    if log:
        logging.info(f"Script duration: {end_t - start_t}")
        logging.info(f"==/\/\/\=== END of the {filepath.name} script ==/\/\/\===")
    print(f"Script duration: {end_t - start_t}")
    print(f"==/\/\/\=== END of the {filepath.name} script ==/\/\/\===")
    return end_t


def reproject_coords(lon, lat, in_crs, out_crs):
    in_crs = CRS.from_user_input(in_crs)
    out_crs = CRS.from_user_input(out_crs)
    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)

    lon, lat = transformer.transform(lon, lat)

    return lon, lat


def reproject_bbox(min_x, min_y, max_x, max_y, in_crs, out_crs):
    in_crs = CRS.from_user_input(in_crs)
    out_crs = CRS.from_user_input(out_crs)
    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)

    min_lon, min_lat = transformer.transform(min_x, min_y)
    max_lon, max_lat = transformer.transform(max_x, max_y)

    return min_lon, min_lat, max_lon, max_lat


def bbox_to_geojson_feature(min_lat, min_lon, max_lat, max_lon):
    coordinates = [
        [
            [min_lon, min_lat],
            [min_lon, max_lat],
            [max_lon, max_lat],
            [max_lon, min_lat],
            [min_lon, min_lat],
        ]
    ]
    return geojson.Feature(geometry=geojson.Polygon(coordinates))
