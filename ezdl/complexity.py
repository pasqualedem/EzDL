import numpy as np
import torchvision.models as models
import inspect
import torch

from inspect import signature
from ptflops import get_model_complexity_info
from ezdl.models import MODELS
from ezdl.utils.utilities import load_yaml


def seg_model_flops(model, size, verbose=False, per_layer_stats=False):
    n_channels, w, h = size
    macs, params = get_model_complexity_info(model, (n_channels, w, h), as_strings=True,
                                             print_per_layer_stat=per_layer_stats, verbose=verbose)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def model_parameters(model): 
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(f'{total_params:,} total parameters.')


def seg_inference_throughput(model, size, batch_size, device):
    n_channels, w, h = size
    dummy_input = torch.randn(batch_size, n_channels, w, h, dtype=torch.float).to(device)
    repetitions = 100
    warmup = 50
    total_time = 0
    with torch.no_grad():
        for rep in range(warmup):
            _ = model(dummy_input)
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (repetitions * batch_size) / total_time
    print('Final Throughput:', throughput)


def seg_inference_inference_per_second(model, size, batch_size, device, model_args={}):
    n_channels, w, h = size
    dummy_input = torch.randn(batch_size, n_channels, w, h, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(50):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f"Mean inference time: {mean_syn} ms")
    print(f"Time per example {mean_syn / batch_size} ms")


def complexity(file):
    params = load_yaml(file)
    models = [
        # ('lawin', 1, {'backbone_pretrained': True}),
        # ('lawin', 2, {'backbone_pretrained': True}),
        # ('lawin', 3, {'backbone_pretrained': True}),
        # ('lawin', 4, {'backbone_pretrained': True, 'main_pretrained': ['R', 'G', 'G', 'G']}),
        #
        # ('lawin', 1, {'backbone': 'MiT-B1'}),
        # ('lawin', 2, {'backbone': 'MiT-B1'}),
        # ('lawin', 3, {'backbone': 'MiT-B1'}),
        # ('lawin', 4, {'backbone': 'MiT-B1'}),
        #
        # ('laweed', 1, {'backbone_pretrained': True}),
        # ('laweed', 2, {'backbone_pretrained': True}),
        # ('laweed', 3, {'backbone_pretrained': True}),
        # ('laweed', 4, {'backbone_pretrained': True, 'main_pretrained': ['R', 'G', 'G', 'G']}),
        #
        # ('laweed', 1, {'backbone_pretrained': True, 'backbone': 'MiT-B1'}),
        # ('laweed', 2, {'backbone_pretrained': True, 'backbone': 'MiT-B1'}),
        # ('laweed', 3, {'backbone_pretrained': True, 'backbone': 'MiT-B1'}),
        # ('laweed', 4, {'backbone_pretrained': True, 'main_pretrained': ['R', 'G', 'G', 'G'], 'backbone': 'MiT-B1'}),
        #
        #
        # ('splitlawin', 3, {'main_channels': 2}),
        # ('splitlawin', 4, {'main_channels': 2}),
        # ('splitlawin', 3, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        # ('splitlawin', 4, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        #
        # ('splitlaweed', 3, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        # ('splitlaweed', 4, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        #
        # ('doublelawin', 3, {'main_channels': 2}),
        # ('doublelawin', 4, {'main_channels': 2}),
        # ('doublelawin', 3, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        # ('doublelawin', 4, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        #
        # ('doublelaweed', 3, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        # ('doublelaweed', 4, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        # ('segnet', 1, {}),
        # ('segnet', 2, {}),
        # ('segnet', 3, {}),
        # ('segnet', 4, {}),
        # ('resnet', 3, {"model_name": "50"}),
        # ('deeplabv3_resnet50', 3, {}, 6),
        # ('splitlawin', 5, {'main_channels': 3, 'backbone': 'MiT-B0'}),
        # ('splitlawin', 5, {'main_channels': 3, 'backbone': 'MiT-B1'}),
        # ('lawin', 5, {'backbone': 'MiT-B0'}, 16),
        # ('lawin', 5, {'backbone': 'MiT-B1'}, 8),
        # ('lawin', 5, {'backbone': 'MiT-LD'}, 32),
        # ('lawin', 5, {'backbone': 'MiT-L0'}, 48),
        # ('lawin', 5, {'backbone': 'MiT-L1'}, 64),
        # ('mit', 5, {'model_name': 'B0'}),
        # ('mit', 5, {'model_name': 'B1'}),
        # ('mit', 5, {'model_name': 'LD'}),
        # ('mit', 5, {'model_name': 'LT'}),
        # ('mit', 5, {'model_name': 'L0'}),
        # ('mit', 5, {'model_name': 'L1'}),
    ]
    per_layer_stats = params.get('per_layer_stats', False)
    verbose = params.get('verbose', False)
    models = params['models']
    wh = params.get("size") or [256, 256]
    default_batch_size = params.get('batch_size', 1)
    defualt_channels = params.get('in_channels', 3)
    default_num_classes = params.get('num_classes', 3)

    for params in models:
        model = params['name']
        batch_size = params.get('batch_size', default_batch_size)
        in_channels = params.get('in_channels', defualt_channels)
        num_classes = params.get('num_classes', default_num_classes)
        params = params.get('params') or {}

        with torch.cuda.device(0):
            if inspect.isclass(MODELS[model]):
                model_signature = signature(MODELS[model].__init__).parameters.keys()
            else:
                model_signature = signature(MODELS[model]).parameters.keys()
            in_params = {'input_channels': in_channels, 'num_classes': num_classes, "output_channels": num_classes, **params}
            if 'arch_params' in model_signature:
                net = MODELS[model](in_params).to('cuda')
            else:
                actual_args = {k: v for k, v in in_params.items() if k in model_signature}
                discard_args = {k: v for k, v in in_params.items() if k not in model_signature}
                print(f"Discarded args: {discard_args}")
                net = MODELS[model](**actual_args).to('cuda')
            size = [in_channels] + wh
            print(f"Model: {model}")
            print(f"args: {params}")
            print(f"Batches: {batch_size}")
            print(f"Size: {size}")
            seg_model_flops(net, size, verbose, per_layer_stats)
            model_parameters(net)
            # seg_inference_throughput(model, channels, batch_size, 'cuda', args)
            seg_inference_inference_per_second(net, size, batch_size, 'cuda')
            torch.cuda.empty_cache()
            print()
        del model


if __name__ == '__main__':
    file = "complexity.yaml"
    complexity(file)