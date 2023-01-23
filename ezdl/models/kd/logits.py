from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.utils import HpmStruct


class LogitsDistillationModule(KDModule):
    """
    Logits Distillation Module
    """

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        if hasattr(self.student.module, "initialize_param_groups"):
            return self.student.module.initialize_param_groups(lr, training_params)
        else:
            return [{"named_params": self.student.named_parameters()}]