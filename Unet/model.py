import torch.nn as nn
import segmentation_models_pytorch as smp


class FloodwaterSegmentationModel(nn.Module):

    def __init__(self,
                 encoder_name="resnet34",
                 encoder_weights="imagenet",
                 in_channels=2,
                 classes=2):
        """
        初始化 FloodwaterSegmentationModel 类。

        Args:
            encoder_name (str): 使用的编码器名称。默认为 'resnet34'。
            encoder_weights (str): 编码器的预训练权重。默认为 'imagenet'。
            in_channels (int): 输入通道数。对于 Sentinel-1 数据，默认为 2(VV 和 VH)。
            classes (int): 输出类别数。对于二分类问题，默认为 2。
        """
        super(FloodwaterSegmentationModel, self).__init__()
        self.model = smp.Unet(encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              in_channels=in_channels,
                              classes=classes)

    def forward(self, x):
        """
        定义模型的前向传播。

        Args:
            x (torch.Tensor): 输入数据。

        Returns:
            torch.Tensor: 模型的输出。
        """
        return self.model(x)
