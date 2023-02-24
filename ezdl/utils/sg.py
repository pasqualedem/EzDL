import torch
from ezdl.models import WrappedModel

def unwrap_model_from_parallel(model, return_was_wrapped=False):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper
    :param model: the model
    :return: the unwrapped model
    """
    if isinstance(
        model,
        (
            torch.nn.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
            WrappedModel,
        ),
    ):
        if return_was_wrapped:
            return model.module, True
        return model.module
    else:
        if return_was_wrapped:
            return model, False
        return model