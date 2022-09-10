from ezdl.models.regseg import RegSeg48
from ezdl.models.backbones import ResNet
from ezdl.models.segnet import SegNet
from ezdl.models.random import Random

MODELS = {
    'segnet': SegNet,
    'regseg48': RegSeg48,
    'random': Random,
    'resnet': ResNet
}
