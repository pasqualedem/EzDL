from typing import Mapping

from super_gradients import KDTrainer
from super_gradients.training.models.kd_modules.kd_module import KDModule

from ezdl.experiment.seg_trainer import SegmentationTrainer


class KDSegTrainer(SegmentationTrainer, KDTrainer):
    def init_model(self, params: Mapping, resume: bool, checkpoint_path: str = None):
        self.teacher_architecture = self._load_model({**params, "model": params["kd"]["teacher"]})
        super().init_model(params, resume, checkpoint_path)

        self.net = KDModule(self.arch_params, student=self.net, teacher=self.teacher_architecture)
