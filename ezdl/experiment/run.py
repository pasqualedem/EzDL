import gc
import sys
import os
import traceback

from copy import deepcopy
from requests.exceptions import ConnectionError

from super_gradients.training.utils.callbacks import Phase
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from ezdl.logger.basesg_logger import AbstractRunWrapper

from ezdl.logger.text_logger import get_logger
from ezdl.callbacks import MetricsLogCallback, callback_factory
from ezdl.experiment.kd_seg_trainer import KDSegTrainer
from ezdl.experiment.ez_trainer import EzTrainer
from ezdl.experiment.kd_ez_trainer import KDEzTrainer
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
        self.carbon_tracker = None
        if '.' not in sys.path:
            sys.path.extend('.')

    def parse_params(self, params):
        self.params = deepcopy(params)
        self.phases = params['phases']

        self.train_params, self.test_params, self.dataset_params, callbacks, self.kd = parse_params(self.params)
        self.train_callbacks, self.val_callbacks, self.test_callbacks = callbacks
        self.run_params = params.get('run_params') or {}
        
    def _init_carbon_tracker(self):
        try:
            self.carbon_tracker = EmissionsTracker(output_dir=self.seg_trainer.sg_logger._local_dir, log_level="warning")
        except ConnectionError:
            logger.warning("CodeCarbon is not connected to a server, using offline tracker")
            self.carbon_tracker = OfflineEmissionsTracker(output_dir=self.seg_trainer.sg_logger._local_dir, log_level="warning", country_iso_code="ITA")
        self.carbon_tracker.start()

    def init(self, params: dict):
        self.seg_trainer = None
        try:
            self.parse_params(params)
            # trainer_class = KDSegTrainer if kd else SegmentationTrainer
            trainer_class = KDEzTrainer if self.kd else EzTrainer
            self.seg_trainer = trainer_class(
                project_name=self.params['experiment']['name'],
                group_name=self.params['experiment']['group'],
                ckpt_root_dir=self.params['experiment']['tracking_dir'] or 'experiments',
            )
            self.dataset = self.seg_trainer.init_dataset \
                    (params['dataset_interface'], dataset_params=deepcopy(self.dataset_params))
            self.seg_trainer.init_model(params, False, None)
            self.seg_trainer.init_loggers({"in_params": params}, deepcopy(self.train_params))
            logger.info(f"Input params: \n\n {dict_to_yaml_string(params)}")
            self.seg_trainer.print_model_summary()
            self._init_carbon_tracker()
        except Exception as e:
            if (
                self.seg_trainer is not None
                and self.seg_trainer.sg_logger is not None
            ):
                self.seg_trainer.sg_logger.close(really=True, failed=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

    def resume(self, logger_run: AbstractRunWrapper, updated_config, phases):
        try:
            try:
                self.params = values_to_number(logger_run.get_params())
            except KeyError as e:
                raise RuntimeError("No params recorded for run, just delete it!") from e
            self.params = nested_dict_update(self.params, updated_config)
            self.phases = phases
            logger_run.update_params(self.params)
            self.train_params, self.test_params, self.dataset_params, callbacks, kd = parse_params(self.params)
            self.train_callbacks, self.val_callbacks, self.test_callbacks = callbacks

            # trainer_class = KDSegTrainer if kd else SegmentationTrainer
            trainer_class = KDEzTrainer if kd else EzTrainer
            self.seg_trainer = trainer_class(
                project_name=self.params['experiment']['name'],
                group_name=self.params['experiment']['group'],
                ckpt_root_dir=self.params['experiment']['tracking_dir'] or 'experiments',
            )
            self.dataset = self.seg_trainer.init_dataset \
                (logger_run.get_params()['dataset_interface'], dataset_params=deepcopy(self.dataset_params))
            checkpoint_path = logger_run.get_local_checkpoint_path(phases)
            self.seg_trainer.init_model(self.params, True, checkpoint_path)
            self.seg_trainer.init_loggers({"in_params": self.params}, self.train_params, run_id=logger_run.id, logger_run=logger_run)
            self._init_carbon_tracker()
        except Exception as e:
            if self.seg_trainer is not None and self.seg_trainer.sg_logger is not None:
                self.seg_trainer.sg_logger.close(really=True, failed=True)
            traceback.print_exception(type(e), e, e.__traceback__)
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
            if self.seg_trainer is not None and self.seg_trainer.sg_logger is not None:
                self.upload_emissions()
                self.seg_trainer.sg_logger.close(True)
                
    def upload_emissions(self):
        self.carbon_tracker.stop()
        self.seg_trainer.sg_logger.add_summary({"emissions": self.carbon_tracker.final_emissions})
        emissions_data = self.carbon_tracker.final_emissions_data.values
        emissions_data_values = [[v for k, v in emissions_data.items()]]
        self.seg_trainer.sg_logger.add_table("emissions", emissions_data_values, emissions_data.keys(), [0])

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
    # cbcks = [
    #     MetricsLogCallback(Phase.TRAIN_EPOCH_END, freq=1),
    #     MetricsLogCallback(Phase.VALIDATION_EPOCH_END, freq=1),
    #     *cbcks
    # ]
    train_params["phase_callbacks"] = cbcks

    seg_trainer.train(training_params=train_params)
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
