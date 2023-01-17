from ezdl.models.kd.feature import FDOutput
from ezdl.utils.utilities import substitute_values

from super_gradients.training.losses.kd_losses import KDklDivLoss
from torch.nn import CrossEntropyLoss, Module, MSELoss
import torch.nn.functional as F
import torch.nn as nn
import torch


def get_reduction(reduction: str):
    if reduction == 'none':
        return lambda x: x
    elif reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")


class CELoss(CrossEntropyLoss):
    def __init__(self, *args, weight=None, **kwargs):
        if weight:
            weight = torch.tensor(weight)
        super().__init__(*args, weight=weight, **kwargs)


class FocalLoss(Module):
    def __init__(self, gamma: float = 0, weight=None, reduction: str = 'mean', **kwargs):
        super().__init__()
        self.weight = None
        if weight:
            self.weight = torch.tensor(weight)
        self.gamma = gamma

        self.reduction = get_reduction(reduction)

    def __call__(self, x, target, **kwargs):
        ce_loss = F.cross_entropy(x, target, reduction='none', **kwargs)
        pt = torch.exp(-ce_loss)
        if self.weight is not None:
            self.weight = self.weight.to(x.device)
            wtarget = substitute_values(target, self.weight, unique=torch.arange(len(self.weight), device=target.device))
            focal_loss = torch.pow((1 - pt), self.gamma) * wtarget * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)


class VisklDivLoss(nn.KLDivLoss):
    """ KL divergence wrapper for Computer Vision tasks."""
    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__(reduction='none')
        self.macro_reduction = get_reduction(reduction)

    def forward(self, student_output, teacher_output):
        return self.macro_reduction(
            super().forward(torch.log_softmax(student_output, dim=1),
                                                torch.softmax(teacher_output, dim=1)).sum(dim=1)
        )


class KDFeatureLogitsLoss(nn.Module):
    name = "KDFLLoss"
    def __init__(self, task_loss_fn, feature_loss_fn=MSELoss(), distillation_loss_fn=VisklDivLoss(),
                 dist_feats_loss_coeffs=(0.2, 0.4, 0.4), feats_loss_reduction='mean'):
        super().__init__()
        self.__class__.__name__ = self.name
        self.task_loss_fn = task_loss_fn
        self.feature_loss_fn = feature_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.dist_feats_loss_coeff = dist_feats_loss_coeffs
        self.feats_loss_reduction = get_reduction(feats_loss_reduction)

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return [self.name, 
        self.task_loss_fn.__class__.__name__, 
        self.distillation_loss_fn.__class__.__name__, 
        self.feature_loss_fn.__class__.__name__
        ]

    def forward(self, kd_output: FDOutput, target):
        logits_loss = self.distillation_loss_fn(kd_output.student_output, kd_output.teacher_output)
        feats_loss = self.feats_loss_reduction(torch.tensor([
            self.feature_loss_fn(student_feat, teacher_feat)
            for student_feat, teacher_feat in zip(kd_output.student_features, kd_output.teacher_features
                                                  )], device=kd_output.student_output.device))
        task_loss = self.task_loss_fn(kd_output.student_output, target)

        loss = task_loss * self.dist_feats_loss_coeff[0] + \
               logits_loss * self.dist_feats_loss_coeff[1] + \
               feats_loss * self.dist_feats_loss_coeff[2]

        return loss, torch.cat((loss.unsqueeze(0),
                                task_loss.unsqueeze(0),
                                logits_loss.unsqueeze(0),
                                feats_loss.unsqueeze(0)
                                )).detach()


class VariationalInformationLoss(nn.Module):
    def forward(self, student_feats, teacher_feats, **kwargs):
        mu, sigma = student_feats
        return torch.sum(
            torch.log(sigma) + torch.div(torch.square(teacher_feats - mu), 2 * torch.square(sigma))
        )


class VariationalInformationLossMean(nn.Module):
    def forward(self, student_feats, teacher_feats, **kwargs):
        mu, sigma = student_feats
        return torch.mean(
            torch.log(sigma) + torch.div(torch.square(teacher_feats - mu), 2 * torch.square(sigma))
        )


class VariationalInformationLossScaled(VariationalInformationLoss):
    def __init__(self, scale_factor=1e-6, *args, **krwargs) -> None:
        super().__init__(*args, **krwargs)
        self.factor = scale_factor

    def forward(self, student_feats, teacher_feats, **kwargs):
        return super().forward(student_feats, teacher_feats, **kwargs) * self.factor


class VariationalInformationLogitsLoss(KDFeatureLogitsLoss):
    name = "VILLoss"
    variants = {
        'standard': VariationalInformationLoss,
        'mean': VariationalInformationLossMean,
        'scaled': VariationalInformationLossScaled,
    }
    def __init__(self, task_loss_fn, distillation_loss_fn=VisklDivLoss(),
                 dist_feats_loss_coeffs=(0.2, 0.4, 0.4), feats_loss_reduction='mean',
                 variant='standard', scale_factor=1e-6):
        params = {'scale_factor': scale_factor} if variant == 'scaled' else {}
        feature_loss_fn = self.variants[variant](**params)
        super().__init__(task_loss_fn=task_loss_fn,
                         feature_loss_fn=feature_loss_fn,
                         distillation_loss_fn=distillation_loss_fn,
                         dist_feats_loss_coeffs=dist_feats_loss_coeffs,
                         feats_loss_reduction=feats_loss_reduction)

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        comps = super().component_names
        comps[0] = self.name
        return comps


LOSSES = {
    'cross_entropy': CELoss,
    'focal': FocalLoss,
    'variational_information_loss': VariationalInformationLoss,
    'mean_variational_information_loss': VariationalInformationLossMean,
    'scaled_variational_information_loss': VariationalInformationLossScaled,
    "vis_kldiv_loss": VisklDivLoss,

    # KD composed losses
    'kd_feature_logits_loss': KDFeatureLogitsLoss,
    'variational_information_logits_loss': VariationalInformationLogitsLoss
}


def instiantiate_loss(loss_name, params):
    """
    Instantiate a loss function from a string name.
    Args:
        loss_name (str): Name of the loss function.
        params (dict): Parameters for the loss function.
    Returns:
        loss_fn (nn.Module): Loss function.
    """
    loss_cls_names = {loss_cls.__name__: loss_cls for loss_cls in LOSSES.values()}
    if loss_name in LOSSES:
        return LOSSES[loss_name](**params)
    elif loss_name in loss_cls_names:
        return loss_cls_names[loss_name](**params)
    elif loss_name in nn.__dict__:
        return nn.__dict__[loss_name](**params)
    else:
        raise ValueError(f'Loss {loss_name} not found. Available losses: {list(LOSSES.keys())}')
