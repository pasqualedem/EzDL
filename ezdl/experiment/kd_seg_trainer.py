from typing import Mapping

from super_gradients import KDTrainer
from super_gradients.common import StrictLoad
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model

from ezdl.experiment.seg_trainer import SegmentationTrainer
from ezdl.models.kd.feature import FeatureDistillationModule

logger = get_logger(__name__)


class KDSegTrainer(SegmentationTrainer, KDTrainer):
    def init_model(self, params: Mapping, resume: bool, checkpoint_path: str = None):
        logger.info("Initializing teacher model")
        self.teacher_architecture, _ = self._load_model({**params, "model": params["kd"]["teacher"]})
        self.net = self.teacher_architecture
        teacher_checkpoint = params.get("kd", {}).get("teacher", {}).get("checkpoint_path", None)
        if teacher_checkpoint is not None:
            load_checkpoint_to_model(params['kd']['teacher']['checkpoint_path'],
                                     load_backbone=False,
                                     net=self.teacher_architecture,
                                     strict=StrictLoad.ON.value,
                                     load_weights_only=False)
        else:
            logger.warning("No teacher checkpoint provided, using random weights")
        self._net_to_device()
        logger.info("Initializing student model")
        super().init_model(params, resume, checkpoint_path)

        self.net = FeatureDistillationModule(self.arch_params, student=self.net, teacher=self.teacher_architecture)

