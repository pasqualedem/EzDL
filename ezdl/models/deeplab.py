from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50 as dls50


def deeplabv3_resnet50(arch_params):
    args = dls50.__code__.co_varnames
    reduced_keys = set(arch_params.keys()).intersection(args)
    reduced_args = {k: arch_params[k] for k in reduced_keys if k in arch_params}
    return dls50(**reduced_args)
