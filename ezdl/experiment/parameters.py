import numpy as np
import torch
from super_gradients.training.utils.callbacks import Phase
from super_gradients.training.utils.early_stopping import EarlyStop

from ezdl.loss import LOSSES as LOSSES_DICT
from ezdl.metrics import metrics_factory
from ezdl.utils.utilities import recursive_get


def parse_params(params: dict) -> (dict, dict, dict, list):
    # Set Random seeds
    torch.manual_seed(params['train_params']['seed'])
    np.random.seed(params['train_params']['seed'])

    # Instantiate loss
    input_train_params = params['train_params']
    loss_params = params['train_params']['loss']
    loss = LOSSES_DICT[loss_params['name']](**loss_params['params'])

    # metrics
    train_metrics = metrics_factory(params['train_metrics'])
    test_metrics = metrics_factory(params['test_metrics'])

    # dataset params
    dataset_params = params['dataset']

    if input_train_params.get('metric_to_watch') == 'loss':
        input_train_params['metric_to_watch'] = loss.__class__.__name__

    train_params = {
        **input_train_params,
        "train_metrics_list": list(train_metrics.values()),
        "valid_metrics_list": list(test_metrics.values()),
        "loss": loss,
        "loss_logging_items_names": ["loss"],
        "sg_logger": params['experiment']['logger'],
        'sg_logger_params': {
            'entity': params['experiment']['entity'],
            'tags': params['tags'],
            'project_name': params['experiment']['name'],
        }
    }

    test_params = {
        "test_metrics": test_metrics,
    }

    # callbacks
    train_callbacks = add_phase_in_callbacks(params.get('train_callbacks') or {}, "train")
    test_callbacks = add_phase_in_callbacks(params.get('test_callbacks') or {}, "test")
    val_callbacks = add_phase_in_callbacks(params.get('val_callbacks') or {}, "validation")

    # early stopping
    if params.get('early_stopping'):
        val_callbacks['early_stopping'] = params.get('early_stopping')

    if recursive_get(val_callbacks, 'early_stopping', 'monitor') == 'loss':
        val_callbacks['early_stopping']['monitor'] = loss.__class__.__name__

    return train_params, test_params, dataset_params, (train_callbacks, val_callbacks, test_callbacks)


def add_phase_in_callbacks(callbacks, phase):
    """
    Add default phase to callbacks
    :param callbacks: dict of callbacks
    :param phase: "train", "validation" or "test"
    :return: dict of callbacks with phase
    """
    for callback in callbacks.values():
        if callback.get('phase') is None:
            callback['phase'] = phase
    return callbacks
