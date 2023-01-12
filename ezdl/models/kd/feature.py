import torch
from super_gradients.training.models import SgModule
from super_gradients.training.models.kd_modules.kd_module import KDModule, KDOutput
from super_gradients.training.utils import HpmStruct
from collections import namedtuple

from torch import nn

FDOutput = namedtuple('KDFOutput', ['student_features',
                                    'student_output',
                                    'teacher_features',
                                    'teacher_output'])

from ezdl.models.layers.common import ConvModule


class FeatureDistillationModule(KDModule):
    def __init__(self, arch_params: HpmStruct, student: SgModule, teacher: torch.nn.Module, run_teacher_on_eval=False):
        super().__init__(arch_params=arch_params,
                         student=student, teacher=teacher,
                         run_teacher_on_eval=run_teacher_on_eval)
        if hasattr(self.student, "module") and not hasattr(self.student, "stepped_forward"):
            self.student.stepped_forward = self.student.module.stepped_forward
        if hasattr(self.teacher, "module") and not hasattr(self.teacher, "stepped_forward"):
            self.teacher.stepped_forward = self.teacher.module.stepped_forward

    def forward(self, x):
        student_features, student_output = self.student.stepped_forward(x)
        if self.teacher_input_adapter is not None:
            teacher_features, teacher_output = self.teacher.stepped_forward(self.teacher_input_adapter(x))
        else:
            teacher_features, teacher_output = self.teacher.stepped_forward(x)
        return FDOutput(
            student_features=student_features,
            student_output=student_output,
            teacher_features=teacher_features,
            teacher_output=teacher_output
        )

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        if hasattr(self.student.module, "initialize_param_groups"):
            return self.student.module.initialize_param_groups(lr, training_params)
        else:
            return [{"named_params": self.student.named_parameters()}]


class VariationalInformationDistillation(FeatureDistillationModule):
    def __init__(self, arch_params: HpmStruct, student: SgModule, teacher: torch.nn.Module, run_teacher_on_eval=False,
                 epsilon=1e-6):
        super().__init__(arch_params=arch_params,
                         student=student, teacher=teacher,
                         run_teacher_on_eval=run_teacher_on_eval)
        self.mu_networks = [ConvModule(inp, out, 3) for inp, out in
                            zip(self.student.module.encoder_maps_sizes, self.teacher.module.encoder_maps_sizes)]
        self.alphas = [torch.nn.Parameter(torch.rand(1)) for _ in self.mu_networks]
        self.sigmas = [SigmaVariance(epsilon) for _ in self.mu_networks]

    def forward(self, x):
        fd_output = super().forward(x)
        mu = [mu_network(fd_output.student_features) for mu_network in self.mu_networks]
        sigmas = [sigma() for sigma in self.sigmas]
        return FDOutput(
            student_features=(mu, sigmas),
            student_output=fd_output.student_output,
            teacher_features=fd_output.teacher_features,
            teacher_output=fd_output.teacher_output
        )


class SigmaVariance(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self):
        return torch.nn.functional.softplus(self.sigma) + self.epsilon
