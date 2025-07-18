import glob
from typing import Any, Union, List, Optional

import kornia.augmentation as K
from kornia.constants import DataKey, Resample
import numpy as np
from sklearn.model_selection import train_test_split
from torch import tensor, Tensor
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datamodules.utils import MisconfigurationException
from torchgeo.samplers.utils import _to_tuple

from ..datasets.enmap_gedi import GediEnmapDataset


class EnMAPGEDIDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EnMAP-GEDI dataset."""

    valid_regions = ('Africa', 'Australasia', 'Europe', 'North_America',
                     'North_Asia', 'South_America', 'South_Asia')

    def __init__(
        self,
        seed: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size: Union[int, tuple[int, int]] = 128,
        dataset_path: str = None,
        regions: List[str] = None,
        num_bands: int = 202,
        split_type: str = 'random',
        test_region: str = None,
        val_size: float = 0.2,
        test_size: float = 0.1,
        use_dofa_norm_values: bool = False,
        use_panopticon_norm_values: bool = False,
        use_n_samples: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize a new EnMAP-GEDI instance.

        Args:
            seed: Random seed for reproducibility (fixed split).
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            patch_size: Size of each patch
            dataset_path: absolute path to the dataset
            regions: list of regions to include in the dataset
            num_bands: number of bands in the dataset
            split_type: type of split to use (random or region-based)
            test_region: region to use for testing (if split_type is region-based)
            val_size: proportion of data to use for validation (if split_type is random)
            test_size: proportion of data to use for testing (if split_type is random)
            use_n_samples: number of samples to use for training (if split_type is random)
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NewDataset`.
        """
        # in the init method of the base class the dataset will be instantiated with **kwargs
        super().__init__(GediEnmapDataset, batch_size, num_workers, **kwargs)

        self.seed = seed
        self.patch_size = _to_tuple(patch_size)
        self.dataset_path = dataset_path
        self.regions = regions if regions else self.valid_regions
        assert all(region in self.valid_regions for region in regions),\
            f"Invalid region(s) in {regions}. Must be one of {self.valid_regions}."
        self.num_bands = num_bands
        self.split_type = split_type
        if test_region:
            assert (test_region in self.valid_regions),\
                f"Invalid test region '{test_region}'. Must be one of {self.valid_regions}."
        self.test_region = test_region
        self.val_size = val_size
        self.test_size = test_size
        self.use_n_samples = use_n_samples

        self.aug = None

        if use_dofa_norm_values:
            try:
                mean = tensor(np.load('data/statistics/dofa_mu.npy'))
                std = tensor(np.load('data/statistics/dofa_sigma.npy'))
            except FileNotFoundError:
                raise MisconfigurationException(f"Missing statistics!")
        elif use_panopticon_norm_values:
            try:
                mean = tensor(np.load('data/statistics/panopticon_mu.npy'))
                std = tensor(np.load('data/statistics/panopticon_sigma.npy'))
            except FileNotFoundError:
                raise MisconfigurationException(f"Missing statistics!")
        else:
            try:
                mean = tensor(np.load('data/statistics/spectralearth_mu.npy'))
                std = tensor(np.load('data/statistics/spectralearth_sigma.npy'))
            except FileNotFoundError:
                raise MisconfigurationException(f"Missing statistics!")


        # Series of Kornia augmentations will be applied to a batch of data in
        # `on_after_batch_transfer`
        self.train_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.4, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.Normalize(mean, std),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.val_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            K.Normalize(mean=mean, std=std),
            data_keys=None,
            keepdim=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            K.Normalize(mean=mean, std=std),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self.split_type == 'random':
            all_patches = []
            for region in self.regions:
                patches_region_paths = glob.glob(f"{self.dataset_path}/{region}/**/*.npy")
                all_patches.extend(patches_region_paths)

            if self.use_n_samples:
                np.random.seed(self.seed)
                all_patches = np.random.choice(all_patches, self.use_n_samples, replace=False)

            train_val_patches, test_patches = train_test_split(
                all_patches, test_size=self.test_size, random_state=self.seed)
            train_patches, val_patches = train_test_split(
                train_val_patches, test_size=self.val_size / (1 - self.test_size), random_state=self.seed)

            if stage in ['fit', 'validate']:
                self.train_dataset = GediEnmapDataset(
                    self.dataset_path, self.regions, 'train',
                    filepaths_split=train_patches, num_bands=self.num_bands)
                self.val_dataset = GediEnmapDataset(
                    self.dataset_path, self.regions, 'val',
                    filepaths_split=val_patches, num_bands=self.num_bands)
            if stage in ['test']:
                self.test_dataset = GediEnmapDataset(
                    self.dataset_path, self.regions, 'test',
                    filepaths_split=test_patches, num_bands=self.num_bands)

        elif self.split_type == 'region-based':
            if stage in ['fit', 'validate']:
                train_val_patches = []
                for region in self.regions:
                    if region != self.test_region:
                        patches_region_paths = glob.glob(f"{self.dataset_path}/{region}/**/*.npy")
                        train_val_patches.extend(patches_region_paths)

                train_patches, val_patches = train_test_split(
                    train_val_patches, test_size=self.val_size, random_state=self.seed)
                self.train_dataset = GediEnmapDataset(
                    self.dataset_path, self.regions, 'train',
                    filepaths_split=train_patches, num_bands=self.num_bands)
                self.val_dataset = GediEnmapDataset(
                    self.dataset_path, self.regions, 'val',
                    filepaths_split=val_patches, num_bands=self.num_bands)
            if stage in ['test']:
                test_patches = glob.glob(f"{self.dataset_path}/{self.test_region}/**/*.npy")
                self.test_dataset = GediEnmapDataset(
                    self.dataset_path, self.regions, 'test',
                    filepaths_split=test_patches, num_bands=self.num_bands)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations after transferring batch to device.

        Args:
            batch: A batch of data that needs to be augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            Augmented batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing or self.trainer.predicting:
                aug = self.test_aug
            else:
                raise NotImplementedError("Unknown trainer state")

            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = batch["image"].to(batch["mask"].device)

        return batch
