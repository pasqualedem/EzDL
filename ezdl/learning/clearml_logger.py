import json
import os
import signal
import time
from typing import Optional, Union, Any

import pandas as pd
import psutil
import torch

from PIL import Image
from flatbuffers.builder import np
from matplotlib import pyplot as plt
from super_gradients.common import ADNNModelRepositoryDataInterfaces
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils import sg_trainer_utils
from clearml import Task, OutputModel

from ezdl.learning.basesg_logger import BaseSGLogger

logger = get_logger(__name__)

WANDB_ID_PREFIX = 'wandb_id.'
WANDB_INCLUDE_FILE_NAME = '.wandbinclude'


class ClearMLLogger(BaseSGLogger):

    def __init__(self, project_name: str, experiment_name: str, storage_location: str, resumed: bool,
                 training_params: dict, checkpoints_dir_path: str, tb_files_user_prompt: bool = False,
                 launch_tensorboard: bool = False, tensorboard_port: int = None, save_checkpoints_remote: bool = True,
                 save_tensorboard_remote: bool = True, save_logs_remote: bool = True, entity: Optional[str] = None,
                 api_server: Optional[str] = None, save_code: bool = False, tags=None, run_id=None, **kwargs):
        """

        :param experiment_name: Used for logging and loading purposes
        :param s3_path: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param checkpoint_loaded: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param max_epochs: the number of epochs planned for this training
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        :param save_code: save current code to wandb
        """
        self.s3_location_available = storage_location.startswith('s3')
        self.resumed = resumed
        resume = 'must' if resumed else None
        os.makedirs(checkpoints_dir_path, exist_ok=True)
        self.run = Task.init(project_name=project_name, task_name=experiment_name, continue_last_task=resume)
        if save_code:
            self._save_code()

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        super().__init__(project_name, experiment_name, storage_location, resumed, training_params,
                         checkpoints_dir_path, tb_files_user_prompt, launch_tensorboard, tensorboard_port,
                         self.s3_location_available, self.s3_location_available, self.s3_location_available)

    @multi_process_safe
    def add_config(self, tag: str = None, config: dict = None):
        self.run.connect(config, name=tag)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        self.run.get_logger().report_scalar(title=tag, value=scalar_value, iteration=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            self.add_scalar(name, value, global_step=global_step)

    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format='CHW', global_step: int = 0):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        self.run.get_logger().report_image(title=tag, series=tag, image=image, iteration=global_step)

    @multi_process_safe
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format='NCHW', global_step: int = 0):
        raise NotImplementedError()


    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0):
        raise NotImplementedError()

    @multi_process_safe
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: str, global_step: int = 0):
        self.run.get_logger().report_histogram(title=tag, series=tag, values=values, iteration=global_step)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        self.run.get_logger().report_text(msg=text_string)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        self.run.get_logger().report_matplotlib_figure(title=tag, figure=figure, iteration=global_step)

    @multi_process_safe
    def add_table(self, tag, data, columns, rows):
        self.run.get_logger().report_table(title=tag, series=tag, table_plot=pd.DataFrame(data, columns=columns))

    @multi_process_safe
    def add_plot(self, tag: str, values: pd.DataFrame, xtitle, ytitle, classes_marker=None):

        if classes_marker:
            for cls in classes_marker:
                scatter2d = values[values == cls].values

                self.run.get_logger().current_logger().report_scatter2d(
                    tag,
                    tag,
                    iteration=None,
                    scatter=scatter2d,
                    xaxis=xtitle,
                    yaxis=ytitle,
                    mode='lines'
                )
        else:
            self.run.get_logger().current_logger().report_scatter2d(
                tag,
                tag,
                iteration=None,
                scatter=values.values,
                xaxis=xtitle,
                yaxis=ytitle,
                mode='lines+markers'
            )

    @multi_process_safe
    def close(self, really=False):
        if really:
            super().close()
            self.run.close()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        self.run.upload_artifact(file_name, artifact_object=os.path.join(self._local_dir, file_name))

    @multi_process_safe
    def add_summary(self, metrics: dict):
        for name, value in metrics:
            self.run.get_logger().report_single_value(name=name, value=value)

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f'ckpt.pth' if tag is None else tag
        if not name.endswith('.pth'):
            name += '.pth'
        models = self.run.get_models()['output']
        model = list(filter(lambda x: x.name == name, models))
        if len(model) > 0:
            model = model[0]
        else:
            model = OutputModel(task=self.run, name=name)

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)
        model.update_weights(weights_filename=path, iteration=global_step)

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning('tensorboard file could not be located for ')
            return None

        return tb_file_path

    @staticmethod
    def _search_upwards_for_file(file_name: str):
        """
        Search in the current directory and all directories above it for a file of a particular name.
        :param file_name: file name to look for.
        :return: pathlib.Path, the location of the first file found or None, if none was found
        """

        try:
            cur_dir = os.getcwd()
            while cur_dir != '/':
                if file_name in os.listdir(cur_dir):
                    return os.path.join(cur_dir, file_name)
                else:
                    cur_dir = os.path.dirname(cur_dir)
        except RuntimeError as e:
            return None

        return None

    def __repr__(self):
        return "ClearMLLogger"
