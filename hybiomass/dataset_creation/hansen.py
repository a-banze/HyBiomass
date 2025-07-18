from hybiomass.dataset_creation.utils import reproject_coords
import pandas as pd
import rioxarray as rxr
import xarray as xr
from os import path
import logging
import zipfile

def extract_hansen_coords(filename):
    """
    Extracts latitude and longitude coordinates from a Hansen file name.

    This function parses a filename to extract latitude and longitude coordinates
    based on a specific naming convention.

    Args:
        filename (str): The name of the Hansen file from which to extract coordinates.
                        The filename is expected to be formatted such that the latitude
                        and longitude information is located in specific positions.

    Returns:
        tuple: A tuple containing four elements:
            - lat (int): Latitude in degrees.
            - lat_bear (str): Latitude hemisphere indicator ('N' or 'S').
            - lon (int): Longitude in degrees.
            - lon_bear (str): Longitude hemisphere indicator ('E' or 'W').

    Example:
        >>> filename = " Hansen_GFC-2023-v1.11_treecover2000_50N_030E.tif"
        >>> extract_hansen_coords(filename)
        (50, 'N', 30, 'E')
    """
    lat_str = filename.split('_')[-2]
    lon_str = filename.split('_')[-1]
    lat = int(lat_str[:2])
    lat_bear = lat_str[2]
    lon = int(lon_str[:3])
    lon_bear = lon_str[3]
    return (lat, lat_bear, lon, lon_bear)
    
def extract_l2l4a_coords(filename):
    """
    Extracts latitude and longitude coordinates from a GEDI L2L4A table name.

    This function parses a filename to extract latitude and longitude coordinates
    based on a specific naming convention.

    Args:
        filename (str): The name of the L2L4A file from which to extract coordinates.
                        The filename is expected to be formatted such that latitude
                        and longitude information is located in the last segment.

    Returns:
        tuple: A tuple containing four elements:
            - lat (int): Latitude in degrees.
            - lat_bear (str): Latitude hemisphere indicator ('N' or 'S').
            - lon (int): Longitude in degrees.
            - lon_bear (str): Longitude hemisphere indicator ('E' or 'W').

    Example:
        >>> filename = "gediv002_l2l4a_va_g014e35n.zip"
        >>> extract_l2l4a_coords(filename)
        (35, 'N', 14, 'E')
    """
    lon_str = filename.split('_')[-1]
    lat_str = filename.split('_')[-1]
    lon = int(lon_str[2:4])
    lon_bear = str.capitalize(lon_str[4])
    lat = int(lat_str[5:7])
    lat_bear = str.capitalize(lat_str[7])
    return (lat, lat_bear, lon, lon_bear)

def match_l2l4a_csv_to_hansen_raster(csv_coords):
    """
    Matches the latitude and longitude coordinates from a GEDI L2L4A table to the
    corresponding Hansen granule coordinates.

    Args:
        csv_coords (tuple): A tuple containing four elements:
            - csv_lat (int): Latitude in degrees from the table.
            - csv_lat_bearing (str): Latitude hemisphere indicator ('N' or 'S').
            - csv_lon (int): Longitude in degrees from the table.
            - csv_lon_bearing (str): Longitude hemisphere indicator ('E' or 'W').

    Returns:
        tuple: A tuple containing four elements:
            - raster_lat (int): Rounded latitude in degrees for the granule.
            - raster_lat_bearing (str): Adjusted latitude hemisphere indicator ('N' or 'S').
            - raster_lon (int): Rounded longitude in degrees for the granule.
            - raster_lon_bearing (str): Adjusted longitude hemisphere indicator ('E' or 'W').

    Example:
        >>> csv_coords = (23, 'N', 45, 'E')
        >>> match_l2l4a_csv_to_hansen_raster(csv_coords)
        (30, 'N', 40, 'E')
    """
    csv_lat, csv_lat_bearing, csv_lon, csv_lon_bearing = csv_coords
    # Determine the correct RASTER latitude
    if csv_lat_bearing == 'N':
        raster_lat = ((csv_lat - 1) // 10 + 1) * 10  # Round up to the next decade
    else:  # S
        raster_lat = (csv_lat // 10) * 10  # Round down to the previous decade
    
    # Determine the correct RASTER longitude
    if csv_lon_bearing == 'E':
        raster_lon = (csv_lon // 10) * 10  # Round down to the previous decade
    else:  # W
        raster_lon = ((csv_lon - 1) // 10 + 1) * 10  # Round up to the next decade

    # Handle special cases where 0°S and 0°W do not exist
    if raster_lat == 0 and csv_lat_bearing == 'S':
        raster_lat_bearing = 'N'
    else:
        raster_lat_bearing = csv_lat_bearing

    if raster_lon == 0 and csv_lon_bearing == 'W':
        raster_lon_bearing = 'E'
    else:
        raster_lon_bearing = csv_lon_bearing

    raster_coords = (raster_lat, raster_lat_bearing, raster_lon, raster_lon_bearing)
    
    return raster_coords

def generate_hansen_raster_filenames(raster_coords, products):
    """
    Generates a list of filenames for Hansen raster files based on the provided
    coordinates and product names.

    This function constructs filenames for Hansen raster files by formatting
    latitude and longitude coordinates and combining them with specified product names.
    The filenames follow a specific naming convention used for Hansen data.

    Args:
        raster_coords (tuple): A tuple containing four elements:
            - raster_lat (int): Latitude in degrees for the granule.
            - raster_lat_bearing (str): Latitude hemisphere indicator ('N' or 'S').
            - raster_lon (int): Longitude in degrees for the granule.
            - raster_lon_bearing (str): Longitude hemisphere indicator ('E' or 'W').

        products (list of str): A list of product names to be included in the filenames.
                                Each product will be used to generate a separate filename.

    Returns:
        list of str: A list of filenames for the Hansen raster files,
                     each formatted according to the specified naming convention.

    Example:
        >>> raster_coords = (10, 'N', 20, 'E')
        >>> products = ['lossyear', 'datamask']
        >>> generate_hansen_raster_filenames(raster_coords, products)
        ['Hansen_GFC-2023-v1.11_lossyear_10N_020E.tif', 'Hansen_GFC-2023-v1.11_datamask_10N_020E.tif']
    """
    raster_lat, raster_lat_bearing, raster_lon, raster_lon_bearing = raster_coords
    # Ensure that latitude and longitude are formatted as two-digit numbers
    lat_str = f"{abs(raster_lat):02d}{raster_lat_bearing}"
    lon_str = f"{abs(raster_lon):03d}{raster_lon_bearing}"
    
    # Construct the filename
    filenames = [f"Hansen_GFC-2023-v1.11_{p}_{lat_str}_{lon_str}.tif" for p in products]
    return filenames

def extract_values_from_da(df, da, bands, df_crs = "EPSG:4326"):
    """
    Extracts raster values for specified bands from a DataArray (typically a
    raster image loaded with rioxarray) at coordinates provided in a DataFrame.

    This function retrieves values from a raster DataArray (`da`) based on coordinates (longitude and latitude) 
    provided in a DataFrame (`df`). The coordinates are reprojected to match the CRS of the DataArray. Values 
    are extracted for each band in the DataArray and appended as new columns in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing longitude and latitude columns used to extract values from the DataArray.
            Expected columns are:
            - "lon_lm_a0" (float): Longitude coordinates.
            - "lat_lm_a0" (float): Latitude coordinates.
        
        da (xr.DataArray): The DataArray containing raster data from which values will be extracted. Must have CRS 
            information set via rasterio.

        bands (list of str): List of band names that correspond to the bands in the DataArray. The number of bands 
            should match the number of dimensions in the DataArray.

        df_crs (str, optional): The coordinate reference system (CRS) of the coordinates in the DataFrame. Default 
            is "EPSG:4326".

    Returns:
        pd.DataFrame: The input DataFrame with additional columns for each band. Each new column contains the extracted 
            raster values for the corresponding band.

    Example:
        >>> df = pd.DataFrame({
        >>>     "lon_lm_a0": [10.0, 20.0],
        >>>     "lat_lm_a0": [50.0, 60.0]
        >>> })
        >>> da = xr.DataArray(...)  # A DataArray with raster data
        >>> bands = ["band1", "band2"]
        >>> updated_df = extract_values_from_da(df, da, bands)
        >>> print(updated_df)
           lon_lm_a0  lat_lm_a0  band1  band2
        0       10.0       50.0   0.25   0.30
        1       20.0       60.0   0.45   0.50
    """
    # Get the raster CRS
    da_crs = da.rio.crs
    
    # Extract longitude and latitude arrays from the DataFrame
    lon_arr = df["lon_lm_a0"].values
    lat_arr = df["lat_lm_a0"].values
    
    # Reproject all coordinates at once from EPSG:4326 to the da CRS
    xx, yy = reproject_coords(lon_arr, lat_arr, df_crs, da_crs)
    
    # Initialize an empty list to store the band values
    values = []
    
    # Loop over each pair of coordinates
    for x, y in zip(xx, yy):
        # Select the nearest point for the current band and coordinate pair
        nearest_point = da.sel(x=x, y=y, method="nearest")
        
        # Extract the value for the nearest point
        value = nearest_point.values  # Convert from array to scalar
        
        # Append the value to the list
        values.append(value)
    
    # Write the values to the DataFrame, one column per band
    for i, b in enumerate(bands):
        df[b] = [arr[i] for arr in values]

    return df

class HansenProcessor:
    """
    Processes Hansen Global Forest Change data and appends it to a table.

    This class handles the integration of Hansen Global Forest Change (GFC) raster data with 
    a provided table of data. It performs the following operations:
    1. Loads Hansen GFC raster data based on raster filenames generated from coordinates.
    2. Loads a CSV table of data.
    3. Extracts values from the raster data for each entry in the table.
    4. Saves the updated table with Hansen GFC data to a specified output directory.

    Args:
        table_dir (str): The directory containing the table CSV or ZIP file.
        table_name (str): The name of the table CSV or ZIP file.
        hansen_dir (str): The directory containing Hansen GFC raster files.
        products (list of str): List of Hansen GFC product names that correspond to the bands in the raster data.

    Attributes:
        table_dir (str): Directory where the table CSV file is located.
        table_name (str): Name of the table CSV file.
        hansen_dir (str): Directory where Hansen GFC raster files are located.
        table_path (str): Full path to the table CSV file.
        products (list of str): List of Hansen GFC product names.
        csv_coords (tuple): Coordinates extracted from the table name.
        raster_coords (tuple): Coordinates used to match Hansen GFC raster files.
        raster_names (list of str): List of matching Hansen GFC raster filenames.
        out_df (pd.DataFrame): DataFrame containing the original table data appended with Hansen GFC values.
    """
    def __init__(self, table_dir, table_name, hansen_dir, products):
        self.table_dir = table_dir
        self.table_name = table_name
        self.hansen_dir = hansen_dir
        self.table_path = path.join(table_dir, table_name)
        self.products = products
        self.csv_coords = extract_l2l4a_coords(self.table_name)
        self.raster_coords = match_l2l4a_csv_to_hansen_raster(self.csv_coords)
        self.raster_names = generate_hansen_raster_filenames(self.raster_coords, self.products)
    
    def load_hansen_rasters(self):
        """
        Loads Hansen GFC raster data from files and concatenates them into a single DataArray.

        This method reads raster files based on the filenames generated from the coordinates, 
        and combines them into a single `xarray.DataArray` with a 'band' dimension.

        Returns:
            xr.DataArray: Combined DataArray containing all Hansen GFC raster bands.
        """
        das = []
        for i, raster in enumerate(self.raster_names):
            das.append(rxr.open_rasterio(path.join(self.hansen_dir, raster)).assign_coords(band=[i+1]))
        da = xr.concat(das, dim='band')
        return da
    
    def load_l2l4a_table(self):
        """
        Loads the GEDI L2L4A table data from a CSV or ZIP file.

        This method reads the table into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the table data.
        """
        try:
            df = pd.read_csv(self.table_path)
            return df
        except zipfile.BadZipFile:
            # Log the error and continue with the next file
            logging.error(f"BadZipFile error: {self.table_path} is not a zip file or it is corrupted.")
        
    def append_hansen_to_table(self):
        """
        Appends Hansen GFC data to the L2L4A table.

        This method extracts raster values for each entry in the table using the loaded Hansen GFC rasters 
        and adds these values as new columns to the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with Hansen GFC values appended to the original table data.
        """
        logging.info(f"Appending Hansen GFC data ({len(self.raster_names)} layers) from {self.raster_coords} granule to the table {self.table_name}")
        df = self.load_l2l4a_table()
        da = self.load_hansen_rasters()
        self.out_df = extract_values_from_da(df, da, self.products)
        return self.out_df
    
    def save_output(self, out_dir):
        """
        Saves the updated table with Hansen GFC values to a CSV file.

        Args:
            out_dir (str): The directory where the output CSV file will be saved.

        Returns:
            str: The full path to the saved output CSV file.
        """
        
        out_path = path.join(out_dir, self.table_name)
        self.out_df.to_csv(out_path, index=False)
        logging.info(f"Output table {self.table_name} with {self.out_df.shape} shape saved to {out_path}")
        return out_path
    