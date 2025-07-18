from typing import Callable, Optional, List

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from torch import Tensor, is_tensor

from torchgeo.datasets import NonGeoDataset


class GediEnmapDataset(NonGeoDataset):
    """GediEnmap Dataset for Aboveground Biomass (AGB) estimation.

    Short summary of the dataset and link to its homepage.

    Dataset features:

    * EnMAP hyperspectral spaceborne imagery
    * Global areas covered

    Dataset format:

    * what file format and shape the input data comes in
    * what file format and shape the target data comes in
    * possible metadata files

    If you use this dataset in your research, please cite the following paper:

    * URL of publication or citation information

    .. versionadded: next TorchGeo minor release version, e.g., 1.0
    """

    valid_splits = ('train', 'val', 'test')
    valid_sensors = ('enmap')

    rgb_indices = {"enmap": [43, 28, 10]}

    def __init__(
        self,
        dataset_path: str,
        sensor: str = 'enmap',
        split: str = 'train',
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 202,
        filepaths_split: List[str] = None,
    ) -> None:
        """Initialize the dataset.

        The init parameters can include additional arguments, such as an option to
        select specific image bands, data modalities, or other arguments that give
        greater control over data loading. They should all have reasonable defaults.

        Args:
            dataset_path: absolute path to the dataset
            sensor: sensor used in the dataset ('enmap')
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            num_bands: number of spectral bands
            filepaths_split: list of filepaths for the split
        """
        self.dataset_path = dataset_path
        self.sensor = sensor
        assert (split in self.valid_splits), f"Choose one of the valid splits: {self.valid_splits}."
        self.split = split

        self.transforms = transforms
        self.num_bands = num_bands
        self.filepaths_split = filepaths_split

    def __len__(self) -> int:
        """The length of the dataset.

        This is the total number of samples per epoch, and is used to define the
        maximum allow index that can be passed to `__getitem__`.
        """

        return len(self.filepaths_split)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """A single sample from the dataset.

        Load a single input image and target label or mask, and return it in a
        dictionary.
        """

        if is_tensor(idx):
            idx = idx.tolist()

        npy_path = self.filepaths_split[idx]
        ndarray = np.load(npy_path)
        enmap = Tensor(ndarray[:-2, :, :])
        agbd = Tensor(ndarray[-2, :, :])
        
        sample = {"image": enmap, "mask": agbd}

        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample

    def plot(
            self,
            sample: dict[str, Tensor],
            ) -> plt.Figure:
        """Plot a sample of the dataset for visualization purposes.

        This might involve selecting the RGB bands, using a colormap to display a mask,
        adding a legend with class labels, etc.

        Creates a plot with subpots:
            1) EnMAP RGB patch
            2) GEDI shots (labels)
            
        If sample dict includes overlayed with predictions if available
            - Overlay subplot 2 with AGB predictions
            - Add third subplot: Scatter plot with AGB labels vs predictions
        """
        ncols=2

        image = sample['image']
        mask = sample['mask']

        # Select RGB bands and improve contrast
        image = image[self.rgb_indices[self.sensor]].numpy()
        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        showing_predictions = "prediction" in sample
        if showing_predictions:
            y_pred = sample["prediction"]#.squeeze(0)
            ncols = 3

        fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=ncols)
        
        # Subplot 1: EnMAP RGB patch
        ax[0].set_title("Input RGB EnMAP patch")
        ax[0].imshow(image, interpolation='none')
        ax[0].axis("off")

        # Subplot 2: GEDI shots (labels), overlayed with predictions if available
        ax[1].set_title("GEDI shots (labels)")
        if showing_predictions:
            ax[1].set_title("GEDI shots overlayed with AGB predictions")
            #TODO: test if need to transpose y_pred?
            im1 = ax[1].imshow(y_pred, cmap='YlGn', interpolation='none', vmin=0)
            divider = make_axes_locatable(ax[1])
            cax1 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax1, orientation='vertical', label='Predicted AGB [Mg/ha]')
        
        gedi = ax[1].imshow(mask, interpolation='none', vmin=0)
        ax[1].axis("off")

        # Subplot 3: Scatter plot with AGB labels vs predictions
        if showing_predictions:
            #TODO: Mask label and y_pred
            ax[2].scatter(mask, y_pred, c=mask, cmap='viridis', vmin=0)
            ax[2].axline((0, 0), slope=1, color='grey', ls="--")  # Add identity line
            divider = make_axes_locatable(ax[2])
            cax2 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(gedi, cax=cax2, orientation='vertical',
                         label='Reference AGB [Mg/ha]')
            ax[2].set_ylabel("Model prediction")
            ax[2].set_xlabel("Reference value (GEDI)")

        plt.tight_layout()
        return fig
    

if __name__ == "__main__":
    dataset_train = GediEnmapDataset(split="train")
    dataset_val = GediEnmapDataset(split="val")
    dataset_test = GediEnmapDataset(split="test")

    print(len(dataset_train), len(dataset_val), len(dataset_test))

    # Retrieve a sample and check some of the desired properties
    x = dataset_train[0]
    assert isinstance(x, dict)
    assert isinstance(x['image'], Tensor)
    assert isinstance(x['mask'], Tensor)

    # Test the plotting method through something like the following
    x = dataset_train[0].copy()
    x['prediction'] = x['mask'].clone()
    dataset_train.plot(x)
    plt.savefig('test_plot.png')
    plt.close()
