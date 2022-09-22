import numpy as np
import torch
from super_gradients.training.utils.callbacks import Phase
from super_gradients.training.utils.early_stopping import EarlyStop

from ezdl.learning.wandb_logger import WandBSGLogger
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
    if recursive_get(params, 'early_stopping', 'params', 'monitor') == 'loss':
        params['early_stopping']['params']['monitor'] = loss.__class__.__name__
    train_params = {
        **input_train_params,
        "train_metrics_list": list(train_metrics.values()),
        "valid_metrics_list": list(test_metrics.values()),
        "loss": loss,
        "loss_logging_items_names": ["loss"],
        "sg_logger": WandBSGLogger,
        'sg_logger_params': {
            'entity': params['experiment']['entity'],
            'tags': params['tags'],
            'project_name': params['experiment']['name'],
        }
    }

    test_params = {
        "test_metrics": test_metrics,
    }

    # early stopping
    early_stop = [EarlyStop(Phase.VALIDATION_EPOCH_END, **params['early_stopping']['params'])] \
        if params['early_stopping']['enabled'] else []

    return train_params, test_params, dataset_params, early_stop
