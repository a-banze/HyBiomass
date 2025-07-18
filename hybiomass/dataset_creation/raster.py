#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz-potsdam.de)
# @License : (C)Copyright 2025, GFZ Potsdam

# %%
import numpy as np
from hybiomass.dataset_creation.utils import reproject_bbox, reproject_coords
from hybiomass.dataset_creation.enmap import list_enmap_tiles, extract_datetime
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import logging
# %%
class SparseRasterCreator():
    """
    A class for creating sparse rasters with geolocated data (e.g., GEDI shots)
    from a table and with the same spatial dimensions as the input DataArray (e.g., EnMAP tile).

    This class initializes with a data array, a table, and other relevant parameters. It sets up the raster's 
    coordinate reference system (CRS), processes the data to create a mask, and prepares an empty raster 
    based on the input data array. The class provides methods to write the data from the table to the empty raster.

    Args:
        in_da (xarray.DataArray): The input data array containing raster data, typically an EnMAP GEOTIF loaded with rxr.open_rasterio().
        table (pd.DataFrame): The table containing geolocated data to write as a sparse raster.
        data_vars (list of str): List of data variables to be used from the table.
        timestamp_id (str, optional): An optional timestamp identifier for the raster, defaults to None.
        table_crs (str, optional): The coordinate reference system of the table data, defaults to "EPSG:4326".

    Attributes:
        raster_crs (str): The coordinate reference system of the input data array.
        mask (xarray.DataArray): A mask array indicating valid data (not equal to -32768) in the input data array.
        empty_da (xarray.DataArray): An empty raster data array created based on the input data array.
    """
    def __init__(self, in_da, table, data_vars, timestamp_id = None, table_crs="EPSG:4326"):
        """
        Initializes the SparseRasterCreator with the given parameters.

        Args:
            in_da (xarray.DataArray): The input data array containing raster data, typically an EnMAP GEOTIF loaded with rxr.open_rasterio().
            table (pd.DataFrame): The table containing geolocated data to write as a sparse raster
            data_vars (list of str): List of data variables to be used from the table.
            timestamp_id (str, optional): An optional timestamp identifier for the raster, defaults to None.
            table_crs (str, optional): The coordinate reference system of the table data, defaults to "EPSG:4326".

        Attributes:
            raster_crs (str): The coordinate reference system of the input data array.
            mask (xarray.DataArray): A mask array indicating valid data (not equal to -32768) in the input data array.
            empty_da (xarray.DataArray): An empty raster data array created based on the input data array.
        """
        self.in_da = in_da
        self.raster_crs = in_da.rio.crs
        self.table = table
        self.data_vars = data_vars
        self.timestamp_id = timestamp_id
        self.table_crs = table_crs
        self.mask = self.in_da.isel(band=0) != -32768
        self.empty_da = self.create_empty_raster()

    def create_empty_raster(self):
        """
        Creates an empty raster with the same spatial dimensions as the input data array.

        This method generates a new `xarray.DataArray` with empty (NaN) values for each band, 
        matching the dimensions and coordinate system of the input data array (`in_da`). 
        The empty raster is initialized with specific attributes and metadata, including 
        band descriptions and fill values. The raster is masked based on the input data array's 
        mask to set areas outside the valid data to a specified no-data value.

        Returns:
            xarray.DataArray: An empty raster data array with dimensions corresponding to the 
                              number of data variables and the spatial dimensions of the input data array.
                              The empty raster is initialized with NaN values and has additional attributes 
                              and metadata, including band descriptions and no-data values.

        Notes:
            - The new raster will have dimensions ("band", "y", "x") where "band" corresponds to the 
              number of data variables, and "y" and "x" correspond to the spatial dimensions of the 
              input data array.
            - The `empty_da` DataArray is initialized with NaN values, but valid data areas are set 
              to -32768.0 based on the mask.
            - The `_FillValue` attribute is set to -32768 and `no_data` attribute is set to NaN.
            - This method uses the coordinate values ("y", "x", "spatial_ref") from the input data array.
        """
        empty_data = np.full((len(self.data_vars), self.in_da.shape[1], self.in_da.shape[2]), np.nan)
        empty_da = xr.DataArray(
            empty_data,
            dims=["band", "y", "x"],
            coords={
                "band": np.arange(len(self.data_vars)),
                "y": self.in_da.coords["y"],
                "x": self.in_da.coords["x"],
                "spatial_ref": self.in_da.coords["spatial_ref"],
            },
            attrs=self.in_da.attrs,
            )
        empty_da.attrs.update({"band_descriptions": self.data_vars})
        empty_da.attrs.update({"_FillValue": -32768, "no_data": np.nan})
        empty_da = empty_da.where(self.mask, -32768.0)
        return empty_da
    
    def write_data(self):
        """
        Writes data from the table to the raster, updating the empty raster with values from the table.

        This method iterates through each entry in the table, reprojects coordinates from the table's CRS to
        the raster's CRS, and updates the corresponding pixels in the raster. It handles cases where pixels
        have already been updated or where new data needs to be written. It tracks and logs the number of 
        entries that were skipped due to having the same pixel index as a previously updated entry.

        Updates:
            - `self.out_da` (xarray.DataArray): The raster data array with updated values from the table.
            - `self.skipped_pixels` (int): The number of pixels that were skipped because they were already
              updated by another entry or falling within the mask.

        Returns:
            xarray.DataArray: The updated raster data array with values from the table.

        Notes:
            - The method uses `reproject_coords` to convert coordinates from the table's CRS to the raster's CRS.
            - It uses `nearest` interpolation to find the nearest raster grid point for each entry.
            - Pixels that were already filled with valid data (i.e. not NaN) or falling within the mask (i.e., equal to -32768.0) are skipped.
            - The method logs the progress of data writing and the percentage of skipped entries.
        """
        coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
        power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']

        self.out_da = self.empty_da.copy()
        logging.info(f"Writing sparse data to raster for tile {self.timestamp_id}...")
        masked_counts = 0
        repeat_counts = 0
        # for _, entry in tqdm(self.table.iterrows(), total=len(self.table)):
        for _, entry in self.table.iterrows():
            _lon = entry["lon_lm_a0"]
            _lat = entry["lat_lm_a0"]
            data = entry[self.data_vars].to_list()
            xx, yy = reproject_coords(_lon, _lat, self.table_crs, self.raster_crs)
            nearest_point = self.out_da.sel(x=xx, y=yy, method="nearest")
            exact_x = nearest_point.coords["x"].values.item()
            exact_y = nearest_point.coords["y"].values.item()

            if np.all(np.isclose(self.out_da.loc[dict(x=exact_x, y=exact_y)], -32768.0)):
                masked_counts += 1
            elif not np.all(np.isnan(self.out_da.loc[dict(x=exact_x, y=exact_y)])):
                repeat_counts += 1
            else:
                # Including the 'beam' variable returns an error, because it's values are strings
                # Add an exception for 'beam' variable coding coverage and power beams as 0 and 1
                beam_index = self.data_vars.index('beam') if 'beam' in self.data_vars else None
                if beam_index is not None:
                    beam = data[beam_index]
                    if beam in coverage_beams:
                        data[beam_index] = 0
                    elif beam in power_beams:
                        data[beam_index] = 1
                    else:
                        logging.warning(f"Unknown beam value: {beam}")
                        data[beam_index] = np.nan
                # All other variables with numeric values
                self.out_da.loc[dict(x=exact_x, y=exact_y)] = data

        self.masked_counts = masked_counts
        self.repeat_counts = repeat_counts               
        logging.info(f"{self.timestamp_id}: number of shots skipped: {self.masked_counts} ({self.masked_counts/len(self.table)*100:.2f}%) (masked) and {self.repeat_counts} ({self.repeat_counts/len(self.table)*100:.2f}%) (repeat) out of {len(self.table)}")
        return self.out_da
    
        
    def save_raster(self, output_path):
        """
        Saves the raster data array to a specified file and writes the number of repeated pixels to a separate file.

        This method saves the raster data (`self.out_da`) to the provided output path using the `rio.to_raster` method. 

        Args:
            output_path (str): The file path where the raster data should be saved. The file extension should be appropriate 
                               for raster formats (e.g., ".TIF").

        Notes:
            - The method assumes that `self.out_da` is a valid `xarray.DataArray` that has been prepared and updated.

            - The method logs success and error messages indicating whether the raster was saved successfully.
        """
        try:
            self.out_da.rio.to_raster(output_path)
            logging.info(
                f"Saved sparse raster for tile {self.timestamp_id} to {output_path}"
            )
        except:
            logging.error(
                f"Could not save sparse raster for tile {self.timestamp_id} to {output_path}"
            )
