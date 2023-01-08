from collections import OrderedDict
from typing import List, Optional

from timm import create_model
from timm.models.layers.eca import EcaModule
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.nn.functional import interpolate


class MultilayerPerceptron(Module):
    """
    Multilayer perceptron with one hidden layer
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        """
        Sets up the modules

        Args:
            in_dim (int): Number of input features
            hidden_dim (int): Number of hidden features
            out_dim (int): Number of output features
        """
        super().__init__()

        self.linear_1 = Linear(
            in_features=in_dim,
            out_features=hidden_dim,
        )
        self.relu = ReLU()
        self.linear_2 = Linear(
            in_features=hidden_dim,
            out_features=out_dim,
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        output = self.linear_1(input)
        output = self.relu(output)
        output = self.linear_2(output)
        return output


class PatchSpatialInteraction(Module):
    """
    Patch spatial interaction (PSI)
    """

    def __init__(
        self,
        in_dim: int,
        input_size: int,
    ) -> None:
        """
        Sets up the modules and parameters

        Args:
            in_dim (int): Number of input features
            input_size (int): Height/width of the input
        """
        super().__init__()

        self.n_height_patches = 16

        if input_size % self.n_height_patches != 0:
            self.n_height_patches = input_size

        patch_size = input_size // self.n_height_patches
        n_patches = self.n_height_patches**2

        self.channel_patch_squeeze = Conv2d(
            in_channels=in_dim,
            out_channels=1,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.flatten = Flatten(
            start_dim=1,
            end_dim=3,
        )

        self.multilayer_perceptron = MultilayerPerceptron(
            in_dim=n_patches,
            hidden_dim=n_patches // 2,
            out_dim=n_patches,
        )
        self.sigmoid = Sigmoid()

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        batch_size, in_dim, height, width = input.shape

        squeezed = self.channel_patch_squeeze(input)
        squeezed = self.flatten(squeezed)

        weights = self.multilayer_perceptron(squeezed)
        weights = self.sigmoid(weights)

        weights = weights.reshape(
            batch_size, 1, self.n_height_patches, self.n_height_patches
        )

        weights = interpolate(
            input=weights,
            size=height,
            mode="nearest",
        )

        output = input * weights
        return output


class ConvolutionBatchNormReLU(Module):
    """
    Convolution, optionally followed by batch norm and ReLU
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        has_batch_norm: bool = True,
        has_relu: bool = True,
    ) -> None:
        """
        Sets up the modules

        Args:
            in_dim (int): Number of input features
            out_dim (int): Number of output features
            kernel_size (int): Kernel size.
            Default is 3
            stride (int): Stride.
            Default is 1
            has_batch_norm (bool): Whether to have batch norm.
            Default is True
            has_relu (bool): Whether to have ReLU.
            Default is True
        """
        super().__init__()

        padding = kernel_size // 2
        bias = not has_batch_norm

        self.convolution = Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        if has_batch_norm:
            self.batch_norm = BatchNorm2d(
                num_features=out_dim,
            )

        else:
            self.batch_norm = None

        if has_relu:
            self.relu = ReLU()

        else:
            self.relu = None

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        output = self.convolution(input)

        if self.batch_norm:
            output = self.batch_norm(output)

        if self.relu:
            output = self.relu(output)

        return output


class PSIResidualBottleneck(Module):
    """
    Residual bottleneck layer with PSI
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_size: int,
        stride: int = 1,
        has_efficient_channel_attention: bool = False,
    ) -> None:
        """
        Sets up the modules

        Args:
            in_dim (int): Number of input features
            hidden_dim (int): Number of hidden features
            out_dim (int): Number of output features
            input_size (int): Height/width of the input
            stride (int): Stride.
            Default is 1
            has_efficient_channel_attention (bool): Whether to apply efficient
            channel attention after PSI.
            Default is False
        """
        super().__init__()

        self.shrink = ConvolutionBatchNormReLU(
            in_dim=in_dim,
            out_dim=hidden_dim,
            kernel_size=1,
            stride=1,
            has_batch_norm=True,
            has_relu=True,
        )

        self.bottleneck = ConvolutionBatchNormReLU(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            kernel_size=3,
            stride=stride,
            has_batch_norm=True,
            has_relu=True,
        )

        self.expand = ConvolutionBatchNormReLU(
            in_dim=hidden_dim,
            out_dim=out_dim,
            kernel_size=1,
            stride=1,
            has_batch_norm=True,
            has_relu=False,
        )

        self.patch_spatial_interaction = PatchSpatialInteraction(
            in_dim=out_dim,
            input_size=input_size,
        )

        if has_efficient_channel_attention:
            self.efficient_channel_attention = EcaModule(
                channels=out_dim,
            )

        else:
            self.efficient_channel_attention = None

        if (in_dim != out_dim) or (stride != 1):
            self.downsample = ConvolutionBatchNormReLU(
                in_dim=in_dim,
                out_dim=out_dim,
                kernel_size=1,
                stride=stride,
                has_batch_norm=True,
                has_relu=False,
            )

        else:
            self.downsample = None

        self.relu = ReLU()

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        residual = input

        output = self.shrink(input)
        output = self.bottleneck(output)
        output = self.expand(output)

        output = self.patch_spatial_interaction(output)

        if self.efficient_channel_attention:
            output = self.efficient_channel_attention(output)

        if self.downsample:
            residual = self.downsample(residual)

        output = output + residual
        output = self.relu(output)
        return output


class PSIResidualBottleneckStage(Module):
    """
    Stage composed of a stack of residual bottleneck layers with PSI
    """

    def __init__(
        self,
        n_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_size: int,
        stride: int = 1,
        has_efficient_channel_attention: bool = False,
    ) -> None:
        """
        Sets up the modules

        Args:
            n_layers (int): Number of layers in the stage
            in_dim (int): Number of input features
            hidden_dim (int): Number of hidden features
            out_dim (int): Number of output features
            input_size (int): Height/width of the input
            stride (int): Stride.
            Default is 1
            has_efficient_channel_attention (bool): Whether to apply efficient
            channel attention after PSI.
            Default is False
        """
        super().__init__()

        input_size = input_size // stride

        residual_bottleneck_layer_with_psi = PSIResidualBottleneck(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_size=input_size,
            stride=stride,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )
        residual_bottleneck_layer_with_psi = (
            f"residual_bottleneck_layer_with_psi_1",
            residual_bottleneck_layer_with_psi,
        )

        residual_bottleneck_layers_with_psi = [
            residual_bottleneck_layer_with_psi,
        ]

        for i in range(n_layers - 1):
            residual_bottleneck_layer_with_psi = PSIResidualBottleneck(
                in_dim=out_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                input_size=input_size,
                stride=1,
                has_efficient_channel_attention=has_efficient_channel_attention,
            )

            residual_bottleneck_layer_with_psi = (
                f"residual_bottleneck_layer_with_psi_{i+2}",
                residual_bottleneck_layer_with_psi,
            )

            residual_bottleneck_layers_with_psi.append(
                residual_bottleneck_layer_with_psi
            )

        residual_bottleneck_layers_with_psi = OrderedDict(
            residual_bottleneck_layers_with_psi
        )
        self.residual_bottleneck_layers_with_psi = Sequential(
            residual_bottleneck_layers_with_psi
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        output = self.residual_bottleneck_layers_with_psi(input)
        return output


class ResNetStem(Module):
    """
    ResNet's stem
    """

    def __init__(
        self,
    ) -> None:
        """
        Sets up the modules
        """
        super().__init__()

        self.convolution_batch_norm_relu = ConvolutionBatchNormReLU(
            in_dim=3,
            out_dim=64,
            kernel_size=7,
            stride=2,
            has_batch_norm=True,
            has_relu=True,
        )
        self.max_pool = MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        output = self.convolution_batch_norm_relu(input)
        output = self.max_pool(output)
        return output


class Head(Module):
    """
    Head for convolutional neural networks for classification
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
    ) -> None:
        """
        Sets up the modules

        Args:
            in_dim (int): Number of input features
            n_classes (int): Number of classes
        """
        super().__init__()

        self.adaptive_average_pooling = AdaptiveAvgPool2d(
            output_size=1,
        )
        self.linear = Linear(
            in_features=in_dim,
            out_features=n_classes,
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        output = self.adaptive_average_pooling(input)
        output = output.squeeze(2).squeeze(2)
        output = self.linear(output)
        return output


class PSIResNet(Module):
    """
    ResNet with PSI
    """

    def __init__(
        self,
        stages_n_layers: List[int],
        input_size: int,
        n_classes: int,
        has_efficient_channel_attention: bool = False,
    ) -> None:
        """
        Sets up the modules

        Args:
            stages_n_layers (List[int]): List containing the number of layers for
            each stage
            input_size (int): Height/width of the input
            n_classes (int): Number of classes
            has_efficient_channel_attention (bool): Whether to apply efficient
            channel attention after PSI.
            Default is False
        """
        super().__init__()

        input_size //= 4

        self.stem = ResNetStem()

        self.stage_1 = PSIResidualBottleneckStage(
            n_layers=stages_n_layers[0],
            in_dim=64,
            hidden_dim=64,
            out_dim=256,
            input_size=input_size,
            stride=1,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )

        self.stage_2 = PSIResidualBottleneckStage(
            n_layers=stages_n_layers[1],
            in_dim=256,
            hidden_dim=128,
            out_dim=512,
            input_size=input_size,
            stride=2,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )

        input_size = input_size // 2

        self.stage_3 = PSIResidualBottleneckStage(
            n_layers=stages_n_layers[2],
            in_dim=512,
            hidden_dim=256,
            out_dim=1024,
            input_size=input_size,
            stride=2,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )

        input_size = input_size // 2

        self.stage_4 = PSIResidualBottleneckStage(
            n_layers=stages_n_layers[3],
            in_dim=1024,
            hidden_dim=512,
            out_dim=2048,
            input_size=input_size,
            stride=2,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )

        self.head = Head(
            in_dim=2048,
            n_classes=n_classes,
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """
        Passes the input through the block

        Args:
            input (Tensor): Input

        Returns (Tensor): Output of the block
        """
        output = self.stem(input)

        output = self.stage_1(output)
        output = self.stage_2(output)
        output = self.stage_3(output)
        output = self.stage_4(output)

        output = self.head(output)
        return output


def get_psi_resnet14(
    input_size: int,
    n_classes: int,
    has_efficient_channel_attention: bool = False,
) -> PSIResNet:
    """
    Gets a PSI-ResNet-14

    Args:
        input_size (int): Height/width of the input
        n_classes (int): Number of classes
        has_efficient_channel_attention (bool): Whether to apply efficient
        channel attention after PSI.
        Default is False

    Returns (PSIResNet): PSI-ResNet-14, possibly with ECA
    """
    psi_resnet15 = PSIResNet(
        stages_n_layers=[1, 1, 1, 1],
        input_size=input_size,
        n_classes=n_classes,
        has_efficient_channel_attention=has_efficient_channel_attention,
    )
    return psi_resnet15


def get_psi_resnet26(
    input_size: int,
    n_classes: int,
    has_efficient_channel_attention: bool = False,
) -> PSIResNet:
    """
    Gets a PSI-ResNet-26

    Args:
        input_size (int): Height/width of the input
        n_classes (int): Number of classes
        has_efficient_channel_attention (bool): Whether to apply efficient
        channel attention after PSI.
        Default is False

    Returns (PSIResNet): PSI-ResNet-26, possibly with ECA
    """
    psi_resnet26 = PSIResNet(
        stages_n_layers=[2, 2, 2, 2],
        input_size=input_size,
        n_classes=n_classes,
        has_efficient_channel_attention=has_efficient_channel_attention,
    )
    return psi_resnet26


def get_psi_resnet50(
    input_size: int,
    n_classes: int,
) -> PSIResNet:
    """
    Gets a PSI-ResNet-50

    Args:
        input_size (int): Height/width of the input
        n_classes (int): Number of classes
        has_efficient_channel_attention (bool): Whether to apply efficient
        channel attention after PSI.
        Default is False

    Returns (PSIResNet): PSI-ResNet-50, possibly with ECA
    """
    psi_resnet50 = PSIResNet(
        stages_n_layers=[3, 4, 6, 3],
        input_size=input_size,
        n_classes=n_classes,
    )
    return psi_resnet50


def get_psi_resnet(
    model_name: str,
    input_size: int,
    n_classes: int,
) -> PSIResNet:
    """
    Gets the corresponding PSI-ResNet from a string name

    Args:
        model_name (str): Name of PSI-ResNet
        input_size (int): Height/width of the input
        n_classes (int): Number of classes

    Returns (PSIResNet): Corresponding PSI-ResNet, possibly with ECA
    """
    has_efficient_channel_attention = model_name.startswith("eca")

    if model_name.endswith("psi_resnet14"):
        psi_resnet = get_psi_resnet14(
            input_size=input_size,
            n_classes=n_classes,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )

    elif model_name.endswith("psi_resnet26"):
        psi_resnet = get_psi_resnet26(
            input_size=input_size,
            n_classes=n_classes,
            has_efficient_channel_attention=has_efficient_channel_attention,
        )

    elif model_name.endswith("psi_resnet50"):
        psi_resnet = get_psi_resnet50(
            input_size=input_size,
            n_classes=n_classes,
        )

    else:
        psi_resnet = None

    return psi_resnet


def get_model(
    model_name: str,
    n_classes: int,
) -> Module:
    """
    Gets the corresponding model from a string name, supporting both timm
    models and PSI-ResNets

    Args:
        model_name (str): Name of PSI-ResNet
        n_classes (int): Number of classes

    Returns (Module): Corresponding model
    """
    if model_name.count("psi"):
        model = get_psi_resnet(
            model_name=model_name,
            input_size=256,
            n_classes=n_classes,
        )

    else:
        model = create_model(
            model_name=model_name,
            num_classes=n_classes,
        )

    return model
