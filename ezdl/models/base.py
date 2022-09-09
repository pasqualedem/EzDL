import torch
import math

from super_gradients.training.utils import HpmStruct
from super_gradients.training import utils as sg_utils
from torch import nn
from ezdl.models.backbones import *
from ezdl.models.layers import trunc_normal_


class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', input_channels: int = 3, backbone_pretrained: bool = False) -> None:
        super().__init__()
        self.backbone_pretrained = backbone_pretrained
        self.backbone = self.eval_backbone(backbone, input_channels, pretrained=backbone_pretrained)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """
        freeze_pretrained = sg_utils.get_param(training_params, 'freeze_pretrained', False)
        if self.backbone_pretrained and freeze_pretrained:
            return [{'named_params': list(filter(lambda x: not x[0].startswith('backbone'), list(self.named_parameters())))}]
        return [{'named_params': self.named_parameters()}]

    @classmethod
    def eval_backbone(cls, backbone: str, input_channels: int, n_blocks:int = 4, pretrained: bool = False) -> nn.Module:
        backbone, variant = backbone.split('-')
        return eval(backbone)(variant, input_channels, n_blocks=n_blocks, pretrained=pretrained)