from ezdl.models.regseg import RegSeg48
from ezdl.models.backbones import ResNet
from ezdl.models.segnet import SegNet
from ezdl.models.random import Random
from ezdl.models.lawin import Lawin, Laweed, DoubleLawin, DoubleLaweed, SplitLawin, SplitLaweed

MODELS = {
    'segnet': SegNet,
    'regseg48': RegSeg48,
    'random': Random,
    'lawin': Lawin,
    'laweed': Laweed,
    'doublelawin': DoubleLawin,
    "doublelaweed": DoubleLaweed,
    'splitlaweed': SplitLaweed,
    'splitlawin': SplitLawin,
    'resnet': ResNet
}
