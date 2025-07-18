#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz-potsdam.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : A class to load, filter and select data from the L1 and L2 Core Datasets

import xarray as xr
import rioxarray as rxr
import numpy as np
import ast
import re
from typing import Optional, Union, List


class Core:
    """
    Core class for loading, filtering and selecting data from the L1 and L2 Core Datasets.

    This class initializes with optional ENMAP and GEDI data arrays, loading them and setting up
    corresponding masks for further processing.

    Args:
        enmap (xr.DataArray, str, optional): An xarray DataArray containing EnMAP data,
                                             or a string containing the path to EnMAP data.
                                             If not provided, will be set to None.
        gedi (xr.DataArray, str, optional): An xarray DataArray containing GEDI data,
                                            or a string containing the path to GEDI data.
                                            If not provided, will be set to None.

    Attributes:
        enmap (xr.DataArray): The loaded EnMAP dataset.
        gedi (xr.DataArray): The loaded GEDI dataset.
        enmap_mask (np.ndarray): A boolean mask for the EnMAP dataset, initialized to
                                 ones if the dataset is provided, otherwise None.
        gedi_mask (np.ndarray): A boolean mask for the GEDI dataset, initialized to
                                ones if the dataset is provided, otherwise None.
    """

    def __init__(
        self,
        *,
        enmap: Optional[Union[xr.DataArray, str]] = None,
        gedi: Optional[Union[xr.DataArray, str]] = None,
    ):
        """
        Initializes the Core class with optional EnMAP and GEDI datasets.

        This constructor loads the provided datasets using the `_load_dataset` method, initializes
        boolean masks for both datasets, and handles cases where the datasets may not be provided.

        Args:
            enmap (xr.DataArray, str, optional): An xarray DataArray containing EnMAP data,
                                                 or a string containing the path to EnMAP data.
                                                 Default is None.
            gedi (xr.DataArray, str, optional): An xarray DataArray containing GEDI data,
                                                or a string containing the path to GEDI data.
                                                Default is None.

        Raises:
            ValueError: If the provided datasets do not meet expected criteria.
        """
        # Load datasets
        self.enmap = self._load_dataset(enmap, "enmap")
        self.gedi = self._load_dataset(gedi, "gedi")

        # Initialize masks
        self.enmap_mask = (
            np.ones(self.enmap.isel(band=0).shape, dtype=bool)
            if self.enmap is not None
            else None
        )
        self.gedi_mask = (
            np.ones(self.gedi.isel(band=0).shape, dtype=bool)
            if self.gedi is not None
            else None
        )

    def _load_dataset(self, dataset, name: str) -> Optional[xr.DataArray]:
        """
        Loads a dataset and validates its type.

        This private method takes a dataset as input, which can be either a file path (string), an
        xarray DataArray, or None. It attempts to load the dataset if it's a file path and performs
        type validation. Additionally, it assigns appropriate band descriptions based on the type of
        dataset being loaded (either "enmap" or "gedi").

        Args:
            dataset (Optional[Union[str, xr.DataArray]]): The dataset to load. This can be:
                - A string representing the file path to the raster data.
                - An xarray DataArray containing the raster data.
                - None, in which case the method returns None.
            name (str): A string indicating the type of dataset being loaded, used for logging
                        and band description assignment (e.g., "enmap", "gedi").

        Returns:
            Optional[xr.DataArray]: The loaded xarray DataArray if valid, or None if the input dataset is None.

        Raises:
            ValueError: If the dataset is not a string, an xarray DataArray, or None.
        """
        if dataset is None:
            return None
        if isinstance(dataset, str):
            dataset = rxr.open_rasterio(dataset, masked=True)
        elif not isinstance(dataset, xr.DataArray):
            raise ValueError(
                f"{name} must be either a string or an xarray DataArray or None"
            )

        band_descriptions = (
            {f"{name}_B{b}": b for b in dataset.coords["band"].values}
            if name == "enmap"
            else {
                b: i + 1
                for i, b in enumerate(ast.literal_eval(dataset.band_descriptions))
            }
        )
        return dataset.assign_attrs({"band_descriptions": band_descriptions})

    def select(
        self, *, enmap: Optional[List[str]] = None, gedi: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Selects specified bands from the EnMAP and GEDI datasets and combines them into a single DataArray.

        This method validates the presence of the specified bands in the datasets and processes the selection
        of the desired bands. The selected bands from ENMAP and GEDI are combined into a single DataArray.

        Args:
            enmap (Optional[List[str]]): A list of band names to select from the ENMAP dataset.
                                        If None, ENMAP data will not be included.
            gedi (Optional[List[str]]): A list of band names to select from the GEDI dataset.
                                        If None, GEDI data will not be included.

        Returns:
            xr.DataArray: A combined DataArray containing the selected bands from ENMAP and GEDI.

        Raises:
            ValueError: If the specified bands are requested but the corresponding dataset is not available.

        """
        self._validate_presence(enmap, self.enmap, "enmap")
        self._validate_presence(gedi, self.gedi, "gedi")

        enmap_data = (
            self._process_selection("enmap", enmap, self.enmap_mask) if enmap else None
        )
        gedi_data = (
            self._process_selection("gedi", gedi, self.gedi_mask) if gedi else None
        )

        return self._combine_datasets(enmap_data, gedi_data)

    def _process_selection(self, dataset_name: str, bands, mask):
        """
        Processes the selection of specified bands from a given dataset.

        This method selects the specified bands from the dataset, applies a mask if provided,
        and reassigns the band coordinates based on the selected bands.

        Args:
            dataset_name (str): The name of the dataset being processed ("enmap" or "gedi").
            bands (Optional[List[str]]): A list of band names to select.
            mask (Optional[np.ndarray]): A boolean mask to apply to the selected data.

        Returns:
            xr.DataArray: The selected and masked DataArray with updated band coordinates.

        """
        dataset = getattr(self, dataset_name)
        selected_data = self._select_bands(dataset, bands, mask)
        selected_data = reassign_band_coords(selected_data, bands)
        return selected_data

    def _validate_presence(self, band_list, dataset, name: str):
        """
        Validates the presence of the specified bands and dataset.

        This method checks whether the specified bands are requested and whether the corresponding dataset is available.
        Raises a ValueError if the dataset is not provided while bands are specified.

        Args:
            band_list (Optional[List[str]]): A list of bands that are requested for selection.
            dataset (Optional[xr.DataArray]): The dataset associated with the bands.
            name (str): A string indicating the type of dataset being checked (e.g., "enmap" or "gedi").

        Raises:
            ValueError: If the specified bands are requested but the corresponding dataset is not available.
        """

        if band_list is not None and dataset is None:
            raise ValueError(
                f"{name} must be provided when initializing Core instance."
            )

    def _select_bands(
        self,
        dataset: Optional[xr.DataArray],
        bands: Optional[List[str]],
        mask: Optional[np.ndarray],
    ) -> Optional[xr.DataArray]:
        """
        Selects specified bands from a given dataset and applies a mask.

        This method extracts the specified bands from the dataset and applies a mask if provided.

        Args:
            dataset (Optional[xr.DataArray]): The dataset from which to select bands.
            bands (Optional[List[str]]): A list of band names to select from the dataset.
            mask (Optional[np.ndarray]): A boolean mask to apply to the selected data.

        Returns:
            Optional[xr.DataArray]: The selected DataArray with the specified bands. Returns None
                                    if the dataset or bands are not provided.
        """
        if dataset is None or bands is None:
            return None
        band_coi = [dataset.band_descriptions[b] for b in bands]
        selected_data = dataset.sel(band=band_coi)
        if mask is not None:
            selected_data = selected_data.where(mask)
        return selected_data

    def _combine_datasets(
        self,
        enmap_data: Optional[xr.DataArray],
        gedi_data: Optional[xr.DataArray],
    ) -> xr.DataArray:
        """
        Combines ENMAP and GEDI datasets into a single DataArray.

        This method concatenates the provided ENMAP and GEDI DataArrays along the 'band' dimension,
        and reassigns the band coordinates.

        Args:
            enmap_data (Optional[xr.DataArray]): The ENMAP dataset to be combined.
            gedi_data (Optional[xr.DataArray]): The GEDI dataset to be combined.

        Returns:
            xr.DataArray: The combined DataArray containing both ENMAP and GEDI data.

        Raises:
            ValueError: If both datasets are None or if dimensions and coordinates do not match.

        """
        if enmap_data is None and gedi_data is None:
            raise ValueError(
                "At least one of enmap or gedi must be provided for selection."
            )
        das = []
        das_band_list = []
        for da in [enmap_data, gedi_data]:
            if da is not None:
                das.append(da)
                das_band_list.extend(list(da.band_descriptions.keys()))

        if len(das) > 1:
            self._validate_dimensions(das)

        combined_data = xr.concat(das, dim="band")
        combined_data = reassign_band_coords(combined_data, das_band_list)

        return combined_data

    def _validate_dimensions(self, datasets: List[xr.DataArray]):
        """
        Validates that the dimensions and coordinates of the provided datasets are compatible.

        This method checks if the datasets have the same dimensions and coordinates. Raises a ValueError
        if they do not match, ensuring that combined datasets can be concatenated without issues.

        Args:
            datasets (List[xr.DataArray]): A list of DataArrays to validate.

        Raises:
            ValueError: If the dimensions or coordinates of the datasets do not match.

        """
        dims_equal = all([datasets[0].dims == ds.dims for ds in datasets])
        coords_equal = all(
            [
                np.array_equal(datasets[0].coords["x"], ds.coords["x"])
                and np.array_equal(datasets[0].coords["y"], ds.coords["y"])
                for ds in datasets
            ]
        )
        if not (dims_equal and coords_equal):
            raise ValueError(
                "enmap and gedi must have the same dimensions and coordinates."
            )

    def filter(
        self, *, enmap: Optional[List[str]] = None, gedi: Optional[List[str]] = None
    ) -> "Core":
        """
        Filters the ENMAP and GEDI datasets based on specified conditions.

        This method applies filters to the ENMAP and/or GEDI datasets based on the conditions provided.
        It validates the presence of the datasets and applies the filtering criteria to generate masks
        that can be used for further processing.

        Args:
            enmap (Optional[List[str]]): A list of conditions to filter the ENMAP dataset. Each condition
                                            should specify the band and the filtering criteria.
            gedi (Optional[List[str]]): A list of conditions to filter the GEDI dataset. Each condition
                                            should specify the band and the filtering criteria.

        Returns:
            Core: The instance of the Core class with applied filters.

        Raises:
            ValueError: If specified conditions refer to bands that are not present in the corresponding dataset.

        """
        if enmap:
            self._validate_presence(enmap, self.enmap, "enmap")
            self._apply_filter("enmap", enmap)

        if gedi:
            self._validate_presence(gedi, self.gedi, "gedi")
            self._apply_filter("gedi", gedi)

        return self

    def _apply_filter(self, dataset_name: str, conditions: List[str]):
        """
            Applies specified filtering conditions to a dataset and updates the corresponding mask.

            This private method processes the given conditions for the specified dataset (ENMAP or GEDI)
            and updates the mask for that dataset based on the filtering criteria.

            Args:
                dataset_name (str): The name of the dataset being filtered ("enmap" or "gedi").
                conditions (List[str]): A list of conditions to apply for filtering the dataset.

            Returns:
                None

            Example:
                >>> self._apply_filter("enmap", [{
            "band": "agbd_a0",
            "operator": "between",
            "lower_bound": 100,
            "upper_bound": 200,
            "left_closed": True,
            "right_closed": True,
        }])
        """
        dataset = getattr(self, dataset_name)
        mask_name = f"{dataset_name}_mask"
        mask = np.ones(dataset.isel(band=0).shape, dtype=bool)
        mask = apply_conditions(dataset, mask, conditions)
        setattr(self, mask_name, mask)


def reassign_band_coords(da, bands):
    """
    Reassigns band coordinates and updates band descriptions for a given DataArray.

    This function takes an xarray DataArray and reassigns the band coordinates to a new range,
    starting from 1 to the number of bands provided. It also updates the attributes of the DataArray
    to include a dictionary of band descriptions based on the provided band names.

    Args:
        da (xr.DataArray): The DataArray for which to reassign band coordinates and descriptions.
        bands (list of str): A list of band names that correspond to the bands in the DataArray.

    Returns:
        xr.DataArray: The updated DataArray with reassigned band coordinates and updated band descriptions.

    Example:
        >>> da = xr.DataArray(...)  # An existing DataArray with band dimension
        >>> bands = ['band1', 'band2', 'band3']
        >>> updated_da = reassign_band_coords(da, bands)
        >>> print(updated_da)
    """

    da = da.assign_coords(band=np.arange(1, len(bands) + 1, 1))
    da = da.assign_attrs({"band_descriptions": {b: i + 1 for i, b in enumerate(bands)}})
    return da


def apply_conditions(da, mask, conditions):
    """
    Applies filtering conditions to a DataArray and updates the mask.

    This function processes a list of conditions, applying them to the provided DataArray to
    update the mask according to the specified criteria.

    Args:
        da (xr.DataArray): The DataArray on which to apply the conditions.
        mask (np.ndarray): The boolean mask to update based on the conditions.
        conditions (List[dict]): A list of dictionaries, each specifying a filtering condition.
                                 Each dictionary should include the band name and the operator
                                 to be applied.

    Returns:
        np.ndarray: The updated mask after applying the specified conditions.

    Raises:
        ValueError: If a specified band is not found in the dataset or if an unsupported operator is provided.
    """

    operator_map = {
        ">=": np.greater_equal,
        ">": np.greater,
        "<=": np.less_equal,
        "<": np.less,
        "==": np.equal,
        "!=": np.not_equal,
        "in": np.isin,
    }

    for condition in conditions:
        band_name = condition.get("band")
        if band_name not in da.band_descriptions:
            raise ValueError(f"Band {band_name} not found in dataset.")

        selected_band = da.sel(band=da.band_descriptions[band_name])
        operator = condition.get("operator")

        if operator == "between":
            lower_bound = condition.get("lower_bound")
            upper_bound = condition.get("upper_bound")
            left_closed = condition.get("left_closed", True)
            right_closed = condition.get("right_closed", True)

            lower_op = np.greater_equal if left_closed else np.greater
            upper_op = np.less_equal if right_closed else np.less
            mask &= lower_op(selected_band, lower_bound) & upper_op(
                selected_band, upper_bound
            )

        elif operator == "in":
            values = condition.get("values")
            mask &= np.isin(selected_band, list(values))

        elif operator in operator_map:
            value = condition.get("value")
            mask &= operator_map[operator](selected_band, value)

        else:
            raise ValueError(f"Unsupported operator: {operator}")

    return mask
