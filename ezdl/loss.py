from ezdl.models.kd.feature import FDOutput

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
        ce_loss = F.cross_entropy(x, target.float(), reduction='none', **kwargs)
        pt = torch.exp(-ce_loss)
        if self.weight is not None:
            wtarget = self.weight[(...,) + (None,) * (len(target.shape) - 1)] \
                          .moveaxis(0, 1).to(target.device) \
                      * target
            wtarget = wtarget[wtarget != 0].reshape(pt.shape)
            focal_loss = torch.pow((1 - pt), self.gamma) * wtarget * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)


class KDFeatureLogitsLoss(nn.Module):
    def __init__(self, task_loss_fn, feature_loss_fn=MSELoss(), distillation_loss_fn=KDklDivLoss(),
                 dist_feats_loss_coeffs=(0.2, 0.4, 0.4), feats_loss_reduction='mean'):
        super().__init__()
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
        return ["KDFeatureLogitsLoss", "Task Loss", "Distillation Loss", "Feature Loss"]

    def forward(self, kd_output: FDOutput, target):
        logits_loss = self.distillation_loss_fn(kd_output.student_output, kd_output.teacher_output)
        feats_loss = self.feats_loss_reduction(torch.tensor([
            self.feature_loss_fn(student_feat, teacher_feat)
            for student_feat, teacher_feat in zip(kd_output.student_features, kd_output.teacher_features
                                                  )]))
        task_loss = self.task_loss_fn(kd_output.student_output, target)

        loss = task_loss * self.dist_feats_loss_coeff[0] + \
               logits_loss * self.dist_feats_loss_coeff[1] + \
               feats_loss * self.dist_feats_loss_coeff[2]

        return loss, torch.cat((loss.unsqueeze(0),
                                task_loss.unsqueeze(0),
                                logits_loss.unsqueeze(0),
                                feats_loss.unsqueeze(0)
                                )).detach()


class VariationalInformationLogitsLoss(KDFeatureLogitsLoss):
    def __init__(self, task_loss_fn, distillation_loss_fn=KDklDivLoss(),
                 dist_feats_loss_coeffs=0.8, feats_loss_reduction='mean'):
        super().__init__(task_loss_fn=task_loss_fn,
                         feature_loss_fn=VariationalInformationLoss(),
                         distillation_loss_fn=distillation_loss_fn,
                         dist_feats_loss_coeffs=dist_feats_loss_coeffs,
                         feats_loss_reduction=feats_loss_reduction)


class VariationalInformationLoss(nn.Module):
    def __call__(self, student_feats, teacher_feats, **kwargs):
        mu, sigma = student_feats
        return torch.sum(
            torch.log(sigma) + torch.div(torch.square(teacher_feats - mu), 2 * torch.square(sigma))
        )


LOSSES = {
    'cross_entropy': CELoss,
    'focal': FocalLoss,
    'kd_feature_logits_loss': KDFeatureLogitsLoss
}
