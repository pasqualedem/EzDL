import importlib
import os
from typing import Mapping

import pandas as pd
import wandb
from super_gradients.common import MultiGPUMode

from ezdl.utils.utilities import get_module_class_from_path
from piptools.scripts.sync import _get_installed_distributions

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers import SG_LOGGERS, BaseSGLogger
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.training import StrictLoad, Trainer

from super_gradients.training.params import TrainingParams
from super_gradients.training.utils.callbacks import Phase, PhaseContext, CallbackHandler
from super_gradients.training import utils as core_utils
from super_gradients.training import models

import torch
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from tqdm import tqdm

from ezdl.learning.wandb_logger import WandBSGLogger
from ezdl.callbacks import SegmentationVisualizationCallback
from ezdl.models import MODELS as MODELS_DICT
from ezdl.learning.wandb_logger import BaseSGLogger as BaseLogger

logger = get_logger(__name__)


class SegmentationTrainer(Trainer):
    def __init__(self, ckpt_root_dir=None, model_checkpoints_location: str = 'local', **kwargs):
        self.run_id = None
        self.train_initialized = False
        self.model_checkpoints_location = model_checkpoints_location
        super().__init__(ckpt_root_dir=ckpt_root_dir, **kwargs)

    def init_dataset(self, dataset_interface, dataset_params):
        if isinstance(dataset_interface, str):
            try:
                module, dataset_interface_cls = get_module_class_from_path(dataset_interface)
                dataset_module = importlib.import_module(module)
                dataset_interface_cls = getattr(dataset_module, dataset_interface_cls)
            except AttributeError:
                raise AttributeError("No interface found!")
            dataset_interface = dataset_interface_cls(dataset_params)
        elif isinstance(dataset_interface, type):
            pass
        else:
            raise ValueError("Dataset interface should be str or class")
        data_loader_num_workers = dataset_params['num_workers']
        self.connect_dataset_interface(dataset_interface, data_loader_num_workers)
        return self.dataset_interface

    def connect_dataset_interface(self, dataset_interface, data_loader_num_workers):
        self.dataset_interface = dataset_interface
        self.train_loader, self.valid_loader, self.test_loader, self.classes = \
            self.dataset_interface.get_data_loaders(batch_size_factor=self.num_devices,
                                                    num_workers=data_loader_num_workers,
                                                    distributed_sampler=self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL)

        self.dataset_params = self.dataset_interface.get_dataset_params()

    def init_model(self, params: Mapping, resume: bool, checkpoint_path: str = None):
        # init model
        model_params = params['model']
        input_channels = len(params['dataset']['channels'])
        output_channels = params['dataset']['num_classes']
        arch_params = {
            'input_channels': input_channels,
            'output_channels': output_channels,
            'in_channels': input_channels,
            'out_channels': output_channels,
            'num_classes': output_channels,
            **model_params['params']
        }
        try:
            module, model_cls = get_module_class_from_path(model_params['name'])
            model_module = importlib.import_module(module)
            model_cls = getattr(model_module, model_cls)
            model = model_cls(arch_params)
        except (AttributeError, ValueError):
            if model_params['name'] in MODELS_DICT.keys():
                model = MODELS_DICT[model_params['name']](arch_params)
            else:
                model = models.get(model_name=model_params['name'], arch_params=arch_params)

        if 'num_classes' not in arch_params.keys():
            if self.classes is None and self.dataset_interface is None:
                raise Exception('Error', 'Number of classes not defined in arch params and dataset is not defined')
            else:
                arch_params['num_classes'] = len(self.classes)

        self.arch_params = core_utils.HpmStruct(**arch_params)
        self.net = model

        self._net_to_device()
        # SET THE FLAG FOR DIFFERENT PARAMETER GROUP OPTIMIZER UPDATE
        self.update_param_groups = hasattr(self.net.module, 'update_param_groups')

        if resume and checkpoint_path is not None:
            self.checkpoint = load_checkpoint_to_model(ckpt_local_path=checkpoint_path,
                                                       load_backbone=False,
                                                       net=self.net,
                                                       strict=StrictLoad.ON.value,
                                                       load_weights_only=self.load_weights_only)
            self.load_checkpoint = True

            if 'ema_net' in self.checkpoint.keys():
                logger.warning(
                    "[WARNING] Main network has been loaded from checkpoint but EMA network exists as well. It "
                    " will only be loaded during validation when training with ema=True. ")

            # UPDATE TRAINING PARAMS IF THEY EXIST & WE ARE NOT LOADING AN EXTERNAL MODEL's WEIGHTS
            self.best_metric = self.checkpoint['acc'] if 'acc' in self.checkpoint.keys() else -1
            self.start_epoch = self.checkpoint['epoch'] if 'epoch' in self.checkpoint.keys() else 0

    def train(self, training_params: dict = dict()):
        super().train(model=self.net,
                      training_params=training_params,
                      train_loader=self.train_loader,
                      valid_loader=self.valid_loader)
        if self.train_loader.num_workers > 0:
            self.train_loader._iterator._shutdown_workers()
            self.valid_loader._iterator._shutdown_workers()
        # Restore best parameters
        self.checkpoint = load_checkpoint_to_model(ckpt_local_path=self.checkpoints_dir_path + '/ckpt_best.pth',
                                                   load_backbone=False,
                                                   net=self.net,
                                                   strict=StrictLoad.ON.value,
                                                   load_weights_only=True)

    def test(self,  # noqa: C901
             test_loader: torch.utils.data.DataLoader = None,
             loss: torch.nn.modules.loss._Loss = None,
             silent_mode: bool = False,
             test_metrics: Mapping = None,
             loss_logging_items_names=None, metrics_progress_verbose=False, test_phase_callbacks=(),
             use_ema_net=True) -> dict:
        """
        Evaluates the model on given dataloader and metrics.

        :param test_loader: dataloader to perform test on.
        :param test_metrics: dict name: Metric for evaluation.
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False). Slows down the program.
        :param use_ema_net (bool) whether to perform test on self.ema_model.ema (when self.ema_model.ema exists,
            otherwise self.net will be tested) (default=True)
        :return: results tuple (tuple) containing the loss items and metric values.

        All of the above args will override SgModel's corresponding attribute when not equal to None. Then evaluation
         is ran on self.test_loader with self.test_metrics.
        """
        test_loader = test_loader or self.test_loader
        loss = loss or self.training_params.loss
        if loss is not None:
            loss.to(self.device)
        test_phase_callbacks = list(test_phase_callbacks) + [
            SegmentationVisualizationCallback(phase=Phase.TEST_BATCH_END,
                                              freq=5,
                                              batch_idxs=list(range(test_loader.__len__())),
                                              num_classes=self.dataset_interface.trainset.CLASS_LABELS,
                                              undo_preprocessing=self.dataset_interface.undo_preprocess)
        ]
        metrics_values = super().test(model=self.net,
                                      test_loader=test_loader,
                                      loss=loss,
                                      silent_mode=silent_mode,
                                      test_metrics_list=list(test_metrics.values()),
                                      loss_logging_items_names=loss_logging_items_names,
                                      metrics_progress_verbose=metrics_progress_verbose,
                                      test_phase_callbacks=test_phase_callbacks,
                                      use_ema_net=use_ema_net)

        metric_names = test_metrics.keys()

        if self.test_loader.num_workers > 0:
            self.test_loader._iterator._shutdown_workers()
        metrics = {'test_loss': metrics_values[0], **dict(zip(metric_names, metrics_values[1:]))}
        if 'conf_mat' in metrics.keys():
            metrics.pop('conf_mat')
            cf = test_metrics['conf_mat'].get_cf()
            logger.info(f'Confusion matrix:\n{cf}')
            self.sg_logger.add_table('confusion_matrix', cf,
                                     columns=list(self.dataset_interface.testset.CLASS_LABELS.values()),
                                     rows=list(self.dataset_interface.testset.CLASS_LABELS.values())
                                     )
        self.sg_logger.add_summary(metrics)
        if 'auc' in metrics.keys():
            logger.info('Computing ROC curve...')
            roc = test_metrics['auc'].get_roc()
            fpr_tpr = [(roc[0][i], roc[1][i]) for i in range(len(roc))]
            skip = [x[0].shape.numel() // 1000 for x in fpr_tpr]
            fpr_tpr = [(fpr[::sk], tpr[::sk]) for (fpr, tpr), sk in zip(fpr_tpr, skip)]
            fprs, tprs = zip(*fpr_tpr)
            fprs = torch.cat(fprs)
            tprs = torch.cat(tprs)
            classes = list(self.dataset_interface.testset.CLASS_LABELS.values())
            cls = [[classes[i]] * len(fpr)
                   for i, (fpr, tpr) in enumerate(fpr_tpr)]
            cls = [item for sublist in cls for item in sublist]
            df = pd.DataFrame({'class': cls, 'fpr': fprs, 'tpr': tprs})
            table = wandb.Table(columns=["class", "fpr", "tpr"], dataframe=df)
            plt = wandb.plot_table(
                "wandb/area-under-curve/v0",
                table,
                {"x": "fpr", "y": "tpr", "class": "class"},
                {
                    "title": "ROC",
                    "x-axis-title": "False positive rate",
                    "y-axis-title": "True positive rate",
                },
            )
            logger.info('ROC curve computed.')
            wandb.log({"roc": plt})
        return metrics

    def _init_monitored_items(self):
        if self.metric_to_watch == 'loss':
            if len(self.loss_logging_items_names) == 1:
                self.metric_to_watch = self.loss_logging_items_names[0]
            else:
                raise ValueError(f"Specify which loss {self.loss_logging_items_names} to watch")
        return super()._init_monitored_items()

    def _initialize_sg_logger_objects(self):
        if not self.train_initialized:
            self.train_initialized = True
            # OVERRIDE SOME PARAMETERS TO MAKE SURE THEY MATCH THE TRAINING PARAMETERS
            general_sg_logger_params = {  # 'experiment_name': self.experiment_name,
                'experiment_name': '',
                'group': self.experiment_name,
                'storage_location': self.model_checkpoints_location,
                'resumed': self.load_checkpoint,
                'training_params': self.training_params,
                'checkpoints_dir_path': self.checkpoints_dir_path,
                'run_id': self.run_id
            }
            sg_logger_params = core_utils.get_param(self.training_params, 'sg_logger_params', {})
            sg_logger = WandBSGLogger(**sg_logger_params, **general_sg_logger_params)
            self.checkpoints_dir_path = sg_logger.local_dir()
            self.training_params.override(sg_logger=sg_logger)
            sg_logger = core_utils.get_param(self.training_params, 'sg_logger')

            # OVERRIDE SOME PARAMETERS TO MAKE SURE THEY MATCH THE TRAINING PARAMETERS
            general_sg_logger_params = {'experiment_name': self.experiment_name,
                                        'storage_location': self.model_checkpoints_location,
                                        'resumed': self.load_checkpoint,
                                        'training_params': self.training_params,
                                        'checkpoints_dir_path': self.checkpoints_dir_path}

            if sg_logger is None:
                raise RuntimeError('sg_logger must be defined in training params (see default_training_params)')

            if isinstance(sg_logger, AbstractSGLogger):
                self.sg_logger = sg_logger
            elif isinstance(sg_logger, str):
                sg_logger_params = core_utils.get_param(self.training_params, 'sg_logger_params', {})
                if issubclass(SG_LOGGERS[sg_logger], BaseSGLogger):
                    sg_logger_params = {**sg_logger_params, **general_sg_logger_params}
                if sg_logger not in SG_LOGGERS:
                    raise RuntimeError('sg_logger not defined in SG_LOGGERS')

                self.sg_logger = SG_LOGGERS[sg_logger](**sg_logger_params)
            else:
                raise RuntimeError('sg_logger can be either an sg_logger name (str) or a subcalss of AbstractSGLogger')

            if not (isinstance(self.sg_logger, BaseSGLogger) or isinstance(self.sg_logger, BaseLogger)):
                logger.warning(
                    "WARNING! Using a user-defined sg_logger: files will not be automatically written to disk!\n"
                    "Please make sure the provided sg_logger writes to disk or compose your sg_logger to BaseSGLogger")

            # IN CASE SG_LOGGER UPDATED THE DIR PATH
            self.checkpoints_dir_path = self.sg_logger.local_dir()
            additional_log_items = {'num_devices': self.num_devices,
                                    'multi_gpu': str(self.multi_gpu),
                                    'device_type': torch.cuda.get_device_name(
                                        0) if torch.cuda.is_available() else 'cpu'}

            # ADD INSTALLED PACKAGE LIST + THEIR VERSIONS
            if self.training_params.log_installed_packages:
                pkg_list = list(map(lambda pkg: str(pkg), _get_installed_distributions()))
                additional_log_items['installed_packages'] = pkg_list

            self.sg_logger.add_config("additional_log_items", additional_log_items)
            self.sg_logger.flush()

    def init_loggers(self,
                     in_params: Mapping = None,
                     train_params: Mapping = None,
                     init_sg_loggers: bool = True,
                     run_id=None) -> None:

        self.run_id = run_id
        if self.training_params is None:
            self.training_params = TrainingParams()
        self.training_params.override(**train_params)
        if init_sg_loggers:
            self._initialize_sg_logger_objects()
        if self.phase_callbacks is None:
            self.phase_callbacks = []
        self.phase_callback_handler = CallbackHandler(self.phase_callbacks)
        self.sg_logger.add_config(config=in_params)

    def run(self, data_loader: torch.utils.data.DataLoader, callbacks=None, silent_mode: bool = False):
        """
        Runs the model on given dataloader.

        :param data_loader: dataloader to perform run on

        """

        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        if callbacks is None:
            callbacks = []
        progress_bar_data_loader = tqdm(data_loader, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True,
                                        disable=silent_mode)
        context = PhaseContext(criterion=self.criterion,
                               device=self.device,
                               sg_logger=self.sg_logger)

        self.phase_callbacks.extend(callbacks)

        if not silent_mode:
            # PRINT TITLES
            pbar_start_msg = f"Running model on {len(data_loader)} batches"
            progress_bar_data_loader.set_description(pbar_start_msg)

        with torch.no_grad():
            for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                batch_items = core_utils.tensor_container_to_device(batch_items, self.device, non_blocking=True)

                additional_batch_items = {}
                targets = None
                if hasattr(batch_items, '__len__'):
                    if len(batch_items) == 2:
                        inputs, targets = batch_items
                    elif len(batch_items) == 3:
                        inputs, targets, additional_batch_items = batch_items
                    else:
                        raise ValueError(f"Expected 1, 2 or 3 items in batch_items, got {len(batch_items)}")
                else:
                    inputs = batch_items

                output = self.net(inputs)

                context.update_context(batch_idx=batch_idx,
                                       inputs=inputs,
                                       preds=output,
                                       target=targets,
                                       **additional_batch_items)

                # TRIGGER PHASE CALLBACKS
                self.phase_callback_handler(Phase.POST_TRAINING, context)
