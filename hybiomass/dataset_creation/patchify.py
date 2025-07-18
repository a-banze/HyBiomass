# Functions to create EnMAP and GEDI patches from tiles

from os import path, makedirs

from pyproj import Transformer
import rasterio
from rtree import index


def build_tile_rtree(enmap_tiles_paths_dict):
    """Builds the tile-level R-Tree index for all tiles.

    This R-Tree is used to check if a tile has any overlapping tile(s).
    While iterating through the tiles, if a tile does not overlap with any tile,
    checking the global patch-level R-Tree can be skipped to save time.

    Args:
        enmap_tiles_paths_dict (dict): keys are datetime_ids, values are paths

    Returns:
        rtree.index.Index: Tile-level R-tree for detecting overlaps between tiles.
    """
    tile_rtree = index.Index()

    for datetime_id, tif_file in enmap_tiles_paths_dict.items():
        with rasterio.open(tif_file) as src:
            bounds = src.bounds
            crs_string = src.crs.to_string()
            transformer_to_latlon = Transformer.from_crs(src.crs, 'EPSG:4326',
                                                         always_xy=True)
            bounds_latlon = transformer_to_latlon.transform_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top)
            unique_id = int(datetime_id[:8]+datetime_id[9:15])
            tile_rtree.insert(unique_id, bounds_latlon, obj=crs_string)

    return tile_rtree


def update_patch_metadata(src, patch_size, patch_window, window_bounds):
    patch_meta = src.meta.copy()
    patch_meta.update({
        'height': patch_size,
        'width': patch_size,
        'transform': src.window_transform(patch_window),
        'bounds': window_bounds})

    # Some meta data, e.g. band_descriptions related to GEDI are saved as tags
    # and are missing after saving the patches if not explicitly copied
    tags = src.tags()

    return patch_meta, tags


def save_patch(output_path, patch_data, patch_meta, patch_tags, tile_idx,
               patch_id, subdir):
    # Save the EnMAP patch as a new TIFF file:
    patch_path = path.join(output_path, subdir,
                           str(tile_idx).zfill(5),
                           f"{str(patch_id).zfill(3)}.tif")
    # Create dir if it does not exist
    if not path.exists(path.dirname(patch_path)):
        makedirs(path.dirname(patch_path))
    with rasterio.open(patch_path, 'w', **patch_meta) as dst:
        dst.write(patch_data)
        dst.update_tags(**patch_tags)

    return patch_path
