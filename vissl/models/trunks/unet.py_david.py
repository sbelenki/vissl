"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from vissl.config import AttrDict
from vissl.models.trunks import register_model_trunk
from vissl.models.model_helpers import (
    Flatten,
    get_trunk_forward_outputs,
    transform_model_input_data_type
)

# Unet from FastMRI
@register_model_trunk("unet")
class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        model_config: AttrDict,
        model_name: str,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.model_config = model_config
        trunk_config = model_config.TRUNK.TRUNK_PARAMS.UNET
        self.in_chans = trunk_config.IN_CHANNELS
        self.out_chans = trunk_config.OUT_CHANNELS
        self.chans = trunk_config.get("CHANNELS", 32)
        self.num_pool_layers = trunk_config.get("NUM_POOLS_LAYERS", 4)
        self.drop_prob =  trunk_config.get("DROP_PROBABILITY", 0.0)
        
        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )
        
        self.down_sample_layers = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        ch = self.chans
        for _ in range(self.num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, self.drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, self.drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(self.num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, self.drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, self.drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        
        # we mapped the layers of resnet model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by requested_feat_keys argument in the
        # forward() call.
        feature_blocks_mapping = []
        for i in range(self.down_sample_layers.__len__()):
            feature_blocks_mapping.append((f"downlayer{i+1}", self.down_sample_layers[i] ))
        
        feature_blocks_mapping.append( ('conv', self.conv) )
        
        for i in range(self.up_transpose_conv.__len__()):
            feature_blocks_mapping.append((f"uptranconvlayer{i+1}", self.up_transpose_conv[i]))
        
        for i in range(self.up_conv.__len__()):
            feature_blocks_mapping.append((f"upconvlayer{i+1}", self.up_conv[i]))
        
        feature_blocks_mapping.append( ("flatten",Flatten(1)) )
        
        self._feature_blocks = nn.ModuleDict(feature_blocks_mapping)
        
        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "conv1": "conv1_relu",
            "res1": "maxpool",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
            "res5avg": "avgpool",
            "flatten": "flatten",
        }
        
    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        feat = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            feature_mapping=self.feat_eval_mapping,
            use_checkpointing=self.use_checkpointing,
            checkpointing_splits=self.num_checkpointing_splits,
        )

    # def forward(self, image: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         image: Input 4D tensor of shape `(N, in_chans, H, W)`.

    #     Returns:
    #         Output tensor of shape `(N, out_chans, H, W)`.
    #     """
    #     stack = []
    #     output = image

    #     # apply down-sampling layers
    #     for layer in self.down_sample_layers:
    #         output = layer(output)
    #         stack.append(output)
    #         output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

    #     output = self.conv(output)

    #     # apply up-sampling layers
    #     for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
    #         downsample_layer = stack.pop()
    #         output = transpose_conv(output)

    #         # reflect pad on the right/botton if needed to handle odd input dimensions
    #         padding = [0, 0, 0, 0]
    #         if output.shape[-1] != downsample_layer.shape[-1]:
    #             padding[1] = 1  # padding right
    #         if output.shape[-2] != downsample_layer.shape[-2]:
    #             padding[3] = 1  # padding bottom
    #         if torch.sum(torch.tensor(padding)) != 0:
    #             output = F.pad(output, padding, "reflect")

    #         output = torch.cat([output, downsample_layer], dim=1)
    #         output = conv(output)

    #     return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
