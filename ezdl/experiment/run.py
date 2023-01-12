import gc
import sys
import os

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.callbacks import Phase
from copy import deepcopy

from ezdl.callbacks import MetricsLogCallback, callback_factory
from ezdl.experiment.kd_seg_trainer import KDSegTrainer
from ezdl.experiment.parameters import parse_params
from ezdl.experiment.seg_trainer import SegmentationTrainer
from ezdl.utils.utilities import dict_to_yaml_string, values_to_number, nested_dict_update

logger = get_logger(__name__)


class Run:
    def __init__(self):
        self.kd = None
        self.params = None
        self.dataset = None
        self.train_callbacks = None
        self.val_callbacks = None
        self.test_callbacks = None
        self.dataset_params = None
        self.seg_trainer = None
        self.train_params = None
        self.test_params = None
        self.run_params = None
        self.phases = None
        if '.' not in sys.path:
            sys.path.extend('.')

    def parse_params(self, params):
        self.params = deepcopy(params)
        self.phases = params['phases']

        self.train_params, self.test_params, self.dataset_params, callbacks, self.kd = parse_params(self.params)
        self.train_callbacks, self.val_callbacks, self.test_callbacks = callbacks
        self.run_params = params.get('run_params') or {}

    def init(self, params: dict):
        self.seg_trainer = None
        try:
            self.parse_params(params)
            trainer_class = KDSegTrainer if self.kd else SegmentationTrainer
            self.seg_trainer = trainer_class(
                experiment_name=self.params['experiment']['group'],
                ckpt_root_dir=self.params['experiment']['tracking_dir'] or 'wandb',
            )
            self.dataset = self.seg_trainer.init_dataset \
                    (params['dataset_interface'], dataset_params=deepcopy(self.dataset_params))
            self.seg_trainer.init_model(params, False, None)
            self.seg_trainer.init_loggers({"in_params": params}, deepcopy(self.train_params))
            logger.info(f"Input params: \n\n {dict_to_yaml_string(params)}")
        except Exception as e:
            if (
                self.seg_trainer is not None
                and self.seg_trainer.sg_logger is not None
            ):
                self.seg_trainer.sg_logger.close(True)
            raise e

    def resume(self, wandb_run, updated_config, phases):
        try:
            try:
                self.params = values_to_number(wandb_run.config['in_params'])
            except KeyError as e:
                raise RuntimeError("No params recorded for run, just delete it!") from e
            self.params = nested_dict_update(self.params, updated_config)
            self.phases = phases
            wandb_run.config['in_params'] = self.params
            wandb_run.update()
            self.train_params, self.test_params, self.dataset_params, callbacks, kd = parse_params(self.params)
            self.train_callbacks, self.val_callbacks, self.test_callbacks = callbacks

            trainer_class = KDSegTrainer if kd else SegmentationTrainer
            self.seg_trainer = trainer_class(
                experiment_name=self.params['experiment']['group'],
                ckpt_root_dir=self.params['experiment']['tracking_dir'] or 'wandb',
            )
            self.dataset = self.seg_trainer.init_dataset \
                (wandb_run.config['in_params']['dataset_interface'], dataset_params=deepcopy(self.dataset_params))
            track_dir = wandb_run.config.get('in_params').get('experiment').get('tracking_dir') or 'wandb'
            checkpoint_path_group = os.path.join(track_dir, wandb_run.group, 'wandb')
            run_folder = list(filter(lambda x: str(wandb_run.id) in x, os.listdir(checkpoint_path_group)))
            checkpoint_path = None
            if 'epoch' in wandb_run.summary:
                ckpt = 'ckpt_latest.pth' if 'train' in phases else 'ckpt_best.pth'
                try:
                    checkpoint_path = os.path.join(checkpoint_path_group, run_folder[0], 'files', ckpt)
                except IndexError as exc:
                    logger.error(f"{wandb_run.id} not found in {checkpoint_path_group}")
                    raise ValueError(
                        f"{wandb_run.id} not found in {checkpoint_path_group}"
                    ) from exc
            self.seg_trainer.init_model(self.params, True, checkpoint_path)
            self.seg_trainer.init_loggers({"in_params": self.params}, self.train_params, run_id=wandb_run.id)
        except Exception as e:
            if self.seg_trainer is not None:
                self.seg_trainer.sg_logger.close(really=True)
            raise e

    def launch(self):
        try:
            if 'train' in self.phases:
                train(self.seg_trainer, self.train_params, self.dataset, self.train_callbacks, self.val_callbacks)

            if 'test' in self.phases:
                test_metrics = self.seg_trainer.test(**self.test_params, test_phase_callbacks=self.test_callbacks)

            if 'inference' in self.phases:
                inference(self.seg_trainer, self.run_params, self.dataset)
        finally:
            if self.seg_trainer is not None:
                self.seg_trainer.sg_logger.close(True)

    @property
    def name(self):
        return self.seg_trainer.sg_logger.name

    @property
    def url(self):
        return self.seg_trainer.sg_logger.url


def train(seg_trainer, train_params, dataset, train_callbacks, val_callbacks):
    # Callbacks

    cbcks = [callback_factory(name, params, seg_trainer=seg_trainer, dataset=dataset, loader=dataset.train_loader)
             for name, params in {**train_callbacks, **val_callbacks}.items()
             ]
    cbcks = [
        MetricsLogCallback(Phase.TRAIN_EPOCH_END, freq=1),
        MetricsLogCallback(Phase.VALIDATION_EPOCH_END, freq=1),
        *cbcks
    ]
    train_params["phase_callbacks"] = cbcks

    seg_trainer.train(train_params)
    gc.collect()


def inference(seg_trainer, run_params, dataset):
    run_loader = dataset.get_run_loader(folders=run_params['run_folders'], batch_size=run_params['batch_size'])
    cbcks = [
        # SaveSegmentationPredictionsCallback(phase=Phase.POST_TRAINING,
        #                                     path=
        #                                     run_params['prediction_folder']
        #                                     if run_params['prediction_folder'] != 'mlflow'
        #                                     else mlclient.run.info.artifact_uri + '/predictions',
        #                                     num_classes=len(seg_trainer.test_loader.dataset.classes),
        #                                     )
    ]
    run_loader.dataset.return_name = True
    seg_trainer.run(run_loader, callbacks=cbcks)
    # seg_trainer.valid_loader.dataset.return_name = True
    # seg_trainer.run(seg_trainer.valid_loader, callbacks=cbcks)
