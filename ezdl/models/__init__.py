from ezdl.models.regseg import RegSeg48
from ezdl.models.backbones import ResNet
from ezdl.models.segnet import SegNet
from ezdl.models.dummy import Random, SingleConv
from ezdl.models.deeplab import deeplabv3_resnet50

MODELS = {
    'segnet': SegNet,
    'regseg48': RegSeg48,
    'random': Random,
    'conv': SingleConv,
    'resnet': ResNet,
    'deeplabv3_resnet50': deeplabv3_resnet50
}
