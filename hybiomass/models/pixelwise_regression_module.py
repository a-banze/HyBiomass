import os
import math
from typing import Any, Optional, Dict, Tuple, Union, List
import timm
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

from ..backbones.dofa_encoder import DofaBasePatch16
from ..backbones.spec_resnet import SpecResNet50
from ..backbones.spec_vit import SpecViTSmall, SpecViTBase, SpecViTLarge, \
    SpecViTHuge, SpecViTGiant
from ..baselines.unet import UNet
from ..decoders.upernet import RegUPerNet


# Model registry mapping model names to their respective classes
MODEL_REGISTRY = {
    "spec_resnet50": SpecResNet50,
    "spec_vit_small": SpecViTSmall,
    "spec_vit_base": SpecViTBase,
    "spec_vit_large": SpecViTLarge,
    "spec_vit_huge": SpecViTHuge,
    "spec_vit_giant": SpecViTGiant,
}


class ConvHead(nn.Module):
    def __init__(self, embedding_size: int = 384, num_classes: int = 1, patch_size: int = 4):
        super(ConvHead, self).__init__()

        # Ensure patch_size is a positive power of 2
        if not (patch_size > 0 and ((patch_size & (patch_size - 1)) == 0)):
            raise ValueError("patch_size must be a positive power of 2.")

        num_upsampling_steps = int(math.log2(patch_size))

        # Determine the initial number of filters (maximum 128 or embedding_size)
        initial_filters = 128

        # Generate the sequence of filters: 128, 64, 32, ..., down to num_classes
        filters = [initial_filters // (2 ** i) for i in range(num_upsampling_steps - 1)]
        filters.append(num_classes)  # Ensure the last layer outputs num_classes channels

        layers = []
        in_channels = embedding_size

        for i in range(num_upsampling_steps):
            out_channels = filters[i]

            # Upsampling layer
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

            # Convolutional layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            # Apply BatchNorm and ReLU only if not the last layer
            if i < num_upsampling_steps - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels  # Update in_channels for the next iteration

        self.segmentation_conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.segmentation_conv(x)


class ViTSegmentor(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int = 4):
        super(ViTSegmentor, self).__init__()
        self.encoder = backbone
        self.num_classes = num_classes  # Add a class for background if needed
        self.embedding_size = backbone.num_features
        self.patch_size = patch_size
        self.head = ConvHead(self.embedding_size, self.num_classes, self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        x = self.encoder.get_intermediate_layers(x, norm=True)[0]
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        x = self.head(x)
        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        inter_channels = in_channels // 8
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, out_channels, 1),
        ]
        super().__init__(*layers)


class FCNRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, regressor: nn.Module) -> None:
        super().__init__()
        self.encoder = backbone
        self.regressor = regressor

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        x = self.regressor(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear",
                          align_corners=False)
        return x


def create_fcn_model(
        backbone: nn.Module, num_classes: int, embedding_size: int
        ) -> FCNRegressor:
    """Creates an FCN regression model with given backbone."""
    # Create the FCN classifier head
    regressor = FCNHead(embedding_size, num_classes)

    # Return the full segmentation model
    return FCNRegressor(backbone, regressor)


class UPerNetRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, regressor: nn.Module,
                 output_layers: Union[int, List[int]], backbone_name: str,
                 output_shape: Tuple[int, int]
                 ) -> None:
        super().__init__()
        self.encoder = backbone
        self.regressor = regressor
        self.output_layers = output_layers
        self.backbone_name = backbone_name
        self.output_shape = output_shape

    def forward(self, x: Tensor) -> Tensor:
        if self.backbone_name in MODEL_REGISTRY:
            intermediate_features = self.encoder.get_intermediate_layers(
                x = x,
                n = self.output_layers,
                reshape=True
            )
            # Convert tuple of tensors to list of tensors
            x = list(intermediate_features)
        elif self.backbone_name == 'dofa_encoder':
            x = self.encoder.forward(x)
        elif self.backbone_name == 'panopticon':
            # reshape argument is not implemented corrected, therefore do it here
            blocks = self.encoder.get_intermediate_layers(x, n=self.output_layers, return_class_token=True)
            outputs = [blk[0] for blk in blocks]  # patch tokens, shape (N, (224/14)**2, 768)
            w, h = (224, 224)
            patch_size = 14
            # Reshape: list(N, (224/14)**2, 768) -> list(N, 768, 224/16, 224/16)
            x = [
                out.reshape(outputs[0].shape[0],  # batch size
                            w // patch_size,
                            h // patch_size,
                            -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        return self.regressor(x, output_shape=self.output_shape)


def create_upernet_model(
        backbone: nn.Module, output_layers: Union[int, List[int]],
        in_channels: int, pyramid_output: bool, backbone_name: str,
        output_shape: Tuple[int, int], upscale_method: str,
        ) -> RegUPerNet:
    """Creates a regression model with given backbone and UPerNet decoder."""

    regressor = RegUPerNet(
        channels=512,
        output_layers=output_layers,
        pool_scales=(1, 2, 3, 6),
        feature_multiplier=1,
        in_channels=[in_channels] * len(output_layers),
        pyramid_output=pyramid_output,
        upscale_method=upscale_method,
    )

    # Return the full segmentation model
    return UPerNetRegressor(backbone, regressor, output_layers, backbone_name, output_shape)


class PixelwiseRegressionModule(LightningModule):
    def __init__(
        self,
        decoder: str,
        backbone: str,
        num_classes: int,
        in_channels: int = 202,
        img_size: int = 128,
        pretrained_weights: Optional[str] = None,
        token_patch_size: int = 4,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        loss_name: str = 'mse',
        pyramid_output: bool = False,
        output_dim: int = None,
        output_layers: Optional[List[int]] = None,
        upscale_method: str = 'bilinear_interpolation',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 10,
        eta_min: float = 1e-5,

        wave_list: Optional[List] = None,
        input_bands: Optional[int] = None
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Args:
            decoder (str): Name of the regression model type to use.
            backbone (str): Name of the timm backbone to use.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-4.
            t_max (int, optional): T_max parameter for learning rate scheduler. Defaults to 10.
            eta_min (float, optional): Minimum learning rate for scheduler. Defaults to 1e-5.

        Raises:
            ValueError: if ignore_index is not an int or None.
        """
        super().__init__()

        # Assign hyperparameters as explicit attributes
        self.decoder = decoder
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.pretrained_weights = pretrained_weights
        self.token_patch_size = token_patch_size
        self.freeze_backbone = freeze_backbone
        self.finetune_adapter = finetune_adapter
        self.loss_name = loss_name
        self.pyramid_output = pyramid_output
        self.output_dim = output_dim
        self.output_layers = output_layers
        self.upscale_method = upscale_method
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min
        self.wave_list = wave_list
        self.input_bands = input_bands

        # Configure the task (initialize model, loss, etc.)
        self._config_task()

        # Initialize metrics
        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.best_val_loss = float('inf')
        self.best_epoch = None

        self.max_val_metrics = {}
        for key in self.val_metrics.keys():
            if key in ["val_RMSE", "val_MAE"]:
                self.max_val_metrics['min_' + key] = float('inf')
            elif key == "val_R2":
                self.max_val_metrics['max_' + key] = float('-inf')

    def _config_task(self) -> None:
        """Configures the task by initializing the model, loss function, and
        freezing layers if necessary."""
        if self.backbone_name in ['unet']:
            self.model = UNet(n_channels=self.in_channels)
        else:
            # Create the backbone based on model_type
            backbone = self._initialize_backbone()

            # Load pretrained weights if specified, Panopticon weights already loaded
            if not self.backbone_name=='panopticon':
                self._load_pretrained_weights(backbone)

            # Initialize the segmentation model based on model_type
            self._initialize_model(backbone)

            # Freeze backbone if specified
            if self.freeze_backbone:
                self._freeze_encoder()

            # Unfreeze adapter layers if finetuning is enabled
            if self.finetune_adapter:
                self._unfreeze_adapter_layers()

        # Set the loss function
        if self.loss_name == "mse":
            self.loss = nn.MSELoss()
        elif self.loss_name == "mae":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Loss type '{self.loss_name}' is not valid.")

    def _initialize_backbone(self) -> nn.Module:
        """Initializes the backbone based on the model type.

        Returns:
            nn.Module: Initialized backbone module.
        """
        if self.backbone_name in MODEL_REGISTRY:
            if self.decoder == "conv_head":
                backbone = MODEL_REGISTRY[self.backbone_name](
                    num_classes=self.num_classes,
                    token_patch_size=self.token_patch_size,
                )
            elif self.decoder == "fcn":
                backbone = MODEL_REGISTRY[self.backbone_name](
                    num_classes=self.num_classes,
                    replace_stride_with_dilation=[True, True, True, True],
                    return_features=True,
                )
            elif self.decoder == "upernet":
                backbone = MODEL_REGISTRY[self.backbone_name](
                    num_classes=self.num_classes,
                    token_patch_size=self.token_patch_size,
                )
        elif self.backbone_name =='dofa_encoder':
            backbone = DofaBasePatch16(
                encoder_weights=self.pretrained_weights,
                input_bands=self.input_bands,
                input_size=self.img_size,
                output_layers=self.output_layers,
                wave_list=self.wave_list,
                use_norm=False,
            )
        elif self.backbone_name =='panopticon':
            backbone = torch.hub.load('Panopticon-FM/panopticon','panopticon_vitb14')

        else:
            # Use timm
            if "vit" in self.backbone_name:
                backbone = timm.create_model(
                    self.backbone_name,
                    pretrained=False,
                    in_chans=self.in_channels,
                    img_size=self.img_size,
                    patch_size=self.token_patch_size,
                    num_classes=self.num_classes,
                )
            elif "resnet" in self.backbone_name:
                backbone = timm.create_model(
                    self.backbone_name,
                    pretrained=False,
                    in_chans=self.in_channels,
                    num_classes=self.num_classes,
                )
            else:
                raise ValueError("Backbone not supported.")

        return backbone

    def _load_pretrained_weights(self, backbone: nn.Module) -> None:
        """Loads pretrained weights into the backbone if a path is provided.

        Args:
            backbone (nn.Module): The backbone module to load weights into.

        Raises:
            FileNotFoundError: If the pretrained_weights path does not exist.
        """
        if self.pretrained_weights:
            if not os.path.exists(self.pretrained_weights):
                raise FileNotFoundError(f"Pretrained weights not found at {self.pretrained_weights}")
            state_dict = torch.load(self.pretrained_weights)
            msg = backbone.load_state_dict(state_dict, strict=False)
            print("Encoder weights loaded:", msg)

    def _initialize_model(self, backbone: nn.Module) -> None:
        """Initializes the segmentation model based on the model type.

        Args:
            backbone (nn.Module): The initialized backbone module.
        """
        if self.decoder == "fcn":
            self.model = create_fcn_model(
                backbone=backbone,
                num_classes=self.num_classes,
                embedding_size=backbone.num_features
            )
        
        elif self.decoder == 'upernet':
            self.model = create_upernet_model(
                backbone=backbone,
                output_layers=self.output_layers,
                in_channels=self.output_dim,
                pyramid_output=self.pyramid_output,
                backbone_name=self.backbone_name,
                output_shape=(self.img_size, self.img_size),
                upscale_method=self.upscale_method,
            )
        elif self.decoder == 'conv_head':
            self.model = ViTSegmentor(
                num_classes=self.num_classes,
                backbone=backbone,
                patch_size=self.token_patch_size,
            )

    def _unfreeze_adapter_layers(self) -> None:
        """Unfreezes the adapter layers."""
        try:
            for param in self.model.encoder.spectral_adapter.parameters():
                param.requires_grad = True
        except AttributeError:
            raise AttributeError("The backbone does not have 'spectral_adapter' attributes.")

    def _freeze_encoder(self) -> None:
        """Freezes the encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        """Unfreezes the encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Model output.
        """
        if self.backbone_name=='panopticon':
            x = {
                'imgs': x,
                'chn_ids': torch.tensor([self.wave_list['enmap']]).repeat(x.shape[0], 1).to(x.device)
            }
            return self.model(x)
        else:
            return self.model(x)
    
    def model_step(
        self, batch: Dict[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input image and labels as tensors

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x = batch['image']
        y = batch['mask']
        y_pred = self.forward(x)
        y_pred_flattened = y_pred.squeeze(dim=1)  # squeeze channel dim
        mask = ~torch.isnan(y)
        y_pred_flattened_masked = y_pred_flattened[mask]
        y_masked = y[mask]
        loss = self.loss(y_pred_flattened_masked, y_masked)
        return loss, y_pred_flattened_masked, y_masked, y, y_pred

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch (Dict[str, Tensor]): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Training loss.
        """
        # Skip if the (last) batch has only one sample, that would lead to an
        # Error in UPerNet Batch Normalization
        if batch['image'].shape[0] == 1:
            return None
        loss, y_pred_flattened_masked, y_masked, _, _ = self.model_step(batch)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.train_metrics(y_pred_flattened_masked, y_masked)

        return loss

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss and log metrics.

        Args:
            batch (Dict[str, Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        loss, y_pred_flattened_masked, y_masked, _, _ = self.model_step(batch)

        loss = self.loss(y_pred_flattened_masked, y_masked)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics(y_pred_flattened_masked, y_masked)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute test loss and log metrics.

        Args:
            batch (Dict[str, Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        loss, y_pred_flattened_masked, y_masked, _, _ = self.model_step(batch)

        loss = self.loss(y_pred_flattened_masked, y_masked)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics(y_pred_flattened_masked, y_masked)

    def on_validation_epoch_end(self) -> None:
        """Logs epoch-level validation metrics."""
        val_metrics = self.val_metrics.compute()
        for key, value in val_metrics.items():
            if key in ["val_RMSE", "val_MAE"]:
                if value < self.max_val_metrics.get('min_' + key, float('inf')):
                    self.max_val_metrics['min_' + key] = value
            elif key == "val_R2":
                if value > self.max_val_metrics.get('max_' + key, float('-inf')):
                    self.max_val_metrics['max_' + key] = value
        self.log_dict(val_metrics)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        for key, value in self.max_val_metrics.items():
            self.log(key, value)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
