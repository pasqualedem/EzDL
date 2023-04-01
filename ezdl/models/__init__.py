from collections import namedtuple
from ezdl.models.regseg import RegSeg48
from ezdl.models.backbones import ResNet, MiT
from ezdl.models.segnet import SegNet
from ezdl.models.dummy import Random, SingleConv
from ezdl.models.deeplab import deeplabv3_resnet50
from ezdl.models.lawin import Lawin, SplitLawin, DoubleLawin
from ezdl.models.base import WrappedModel
from super_gradients.training.models.kd_modules.kd_module import KDOutput

from ezdl.models.kd.feature import \
    FeatureDistillationModule, \
    FeatureDistillationConvAdapter, \
    VariationalInformationDistillation
    
from ezdl.models.kd.logits import LogitsDistillationModule

MODELS = {
    'conv': SingleConv,
    'resnet': ResNet,
    'mit': MiT,

    'segnet': SegNet,
    'regseg48': RegSeg48,
    'random': Random,
    'deeplabv3_resnet50': deeplabv3_resnet50,
    'lawin': Lawin,
    'splitlawin': SplitLawin,
    'doublelawin': DoubleLawin,
}

KD_MODELS = {
    "feature_distillation": FeatureDistillationModule,
    "variational_information_distillation": VariationalInformationDistillation,
    "feature_distillation_conv_adapter": FeatureDistillationConvAdapter,
    "logits_distillation": LogitsDistillationModule,
}


ComposedOutput = namedtuple('ComposedOutput', ['main',
                                    'aux'])