import os
from typing import Optional, Union, Any

import pandas as pd
import numpy as np

import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from super_gradients.common.environment.ddp_utils import multi_process_safe

from ezdl.logger.basesg_logger import AbstractRunWrapper, BaseSGLogger
from ezdl.logger.text_logger import get_logger


logger = get_logger(__name__)

WANDB_ID_PREFIX = 'wandb_id.'
WANDB_INCLUDE_FILE_NAME = '.wandbinclude'


class WandBSGLogger(BaseSGLogger):

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
        self.run = wandb.init(project=project_name, name=experiment_name,
                         entity=entity, resume=resume, id=run_id, tags=tags,
                         dir=checkpoints_dir_path, group=kwargs.get('group'))
        if save_code:
            self._save_code()

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        checkpoints_dir_path = os.path.relpath(self.run.dir, os.getcwd())
        super().__init__(project_name, experiment_name, storage_location, resumed, training_params,
                         checkpoints_dir_path, tb_files_user_prompt, launch_tensorboard, tensorboard_port,
                         self.s3_location_available, self.s3_location_available, self.s3_location_available)

        self._set_wandb_id(self.run.id)
        if api_server is not None:
            if api_server != os.getenv('WANDB_BASE_URL'):
                logger.warning(f'WANDB_BASE_URL environment parameter not set to {api_server}. Setting the parameter')
                os.putenv('WANDB_BASE_URL', api_server)

    @multi_process_safe
    def _save_code(self):
        """
        Save the current code to wandb.
        If a file named .wandbinclude is avilable in the root dir of the project the settings will be taken from the file.
        Otherwise, all python file in the current working dir (recursively) will be saved.
        File structure: a single relative path or a single type in each line.
        i.e:

        src
        tests
        examples
        *.py
        *.yaml

        The paths and types in the file are the paths and types to be included in code upload to wandb
        """
        base_path, paths, types = self._get_include_paths()

        if len(types) > 0:
            def func(path):
                for p in paths:
                    if path.startswith(p):
                        for t in types:
                            if path.endswith(t):
                                return True
                return False

            include_fn = func
        else:
            include_fn = lambda path: path.endswith(".py")

        if base_path != ".":
            wandb.run.log_code(base_path, include_fn=include_fn)
        else:
            wandb.run.log_code(".", include_fn=include_fn)


    @multi_process_safe
    def add_config(self, tag: str = None, config: dict = None):
        if tag:
            config = {tag: config}
        wandb.config.update(config, allow_val_change=self.resumed)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        wandb.log(data={tag: scalar_value}, step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            if isinstance(value, dict):
                tag_scalar_dict[name] = value['value']
        wandb.log(data=tag_scalar_dict, step=global_step)

    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format='CHW', global_step: int = 0):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        wandb.log(data={tag: wandb.Image(image, caption=tag)}, step=global_step)

    @multi_process_safe
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format='NCHW', global_step: int = 0):

        wandb_images = []
        for im in images:
            if isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()

            if im.shape[0] < 5:
                im = im.transpose([1, 2, 0])
            wandb_images.append(wandb.Image(im))
        wandb.log({tag: wandb_images}, step=global_step)

    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0):

        if video.ndim > 4:
            for index, vid in enumerate(video):
                self.add_video(tag=f'{tag}_{index}', video=vid, global_step=global_step)
        else:
            if isinstance(video, torch.Tensor):
                video = video.cpu().detach().numpy()
            wandb.log({tag: wandb.Video(video, fps=4)}, step=global_step)

    @multi_process_safe
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: str, global_step: int = 0):
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

    @multi_process_safe
    def add_plot(self, tag: str, values: pd.DataFrame, xtitle, ytitle, classes_marker):

        table = wandb.Table(columns=[classes_marker, xtitle, ytitle], dataframe=values)
        plt = wandb.plot_table(
            tag,
            table,
            {"x": xtitle, "y": ytitle, "class": classes_marker},
            {
                "title": tag,
                "x-axis-title": xtitle,
                "y-axis-title": ytitle,
            },
        )
        wandb.log({tag: plt})

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        wandb.log({tag: text_string}, step=global_step)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        wandb.log({tag: figure}, step=global_step)

    @multi_process_safe
    def add_mask(self, tag: str, image, mask_dict, global_step: int = 0):
        wandb.log({tag: wandb.Image(image, masks=mask_dict)}, step=global_step)

    @multi_process_safe
    def add_table(self, tag, data, columns, rows):
        if isinstance(data, torch.Tensor):
            data = [[x.item() for x in row] for row in data]
        table = wandb.Table(data=data, rows=rows, columns=columns)
        wandb.log({tag: table})

    @multi_process_safe
    def close(self, really=False, failed=False):
        if really:
            super().close()
            wandb.finish()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        wandb.save(glob_str=os.path.join(self._local_dir, file_name), base_path=self._local_dir, policy='now')

    @multi_process_safe
    def add_summary(self, metrics: dict):
        wandb.summary.update(metrics)

    @multi_process_safe
    def upload(self):

        if self.save_tensorboard_wandb:
            wandb.save(glob_str=self._get_tensorboard_file_name(), base_path=self._local_dir, policy='now')

        if self.save_logs_wandb:
            wandb.save(glob_str=self.log_file_path, base_path=self._local_dir, policy='now')

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f'ckpt_{global_step}.pth' if tag is None else tag
        if not name.endswith('.pth'):
            name += '.pth'

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_wandb:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, name)
            wandb.save(glob_str=path, base_path=self._local_dir, policy='now')

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning('tensorboard file could not be located for ')
            return None

        return tb_file_path

    def _get_wandb_id(self):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, '')

    def _set_wandb_id(self, id):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self._local_dir, file))

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    def _get_include_paths(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        wandb_include_file_path = self._search_upwards_for_file(WANDB_INCLUDE_FILE_NAME)
        if wandb_include_file_path is not None:
            with open(wandb_include_file_path) as file:
                lines = file.readlines()

            base_path = os.path.dirname(wandb_include_file_path)
            paths = []
            types = []
            for line in lines:
                line = line.strip().strip('/n')
                if line == "" or line.startswith("#"):
                    continue

                if line.startswith('*.'):
                    types.append(line.replace('*', ''))
                else:
                    paths.append(os.path.join(base_path, line))
            return base_path, paths, types

        return ".", [], []

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

    def create_image_mask_sequence(self, name):
        self.sequences[name] = wandb.Table(["ID", "Image"])

    def add_image_mask_to_sequence(self, sequence_name, name, image, mask_dict):
        self.sequences[sequence_name].add_data(name, wandb.Image(image, masks=mask_dict))

    def add_image_mask_sequence(self, name):
        wandb.log({name: self.sequences[name]})

    @property
    def name(self):
        return self.run.name

    @property
    def url(self):
        return self.run.url

    def __repr__(self):
        return "WandbSGLogger"
    
    @classmethod
    def get_interrupted_run(cls, input_settings):
        namespace = input_settings["name"]
        group = input_settings["group"]
        last_run = wandb.Api().runs(path=namespace, filters={"group": group,}, order="-created_at")[0]
        filters = {"group": group, "name": last_run.id}
        stage = ["train", "test"]
        updated_config = None
        api = wandb.Api()
        runs = api.runs(path=namespace, filters=filters)
        if len(runs) == 0:
            raise RuntimeError("No runs found")
        if len(runs) > 1:
            raise EnvironmentError("More than 1 run???")
        return WandbRunWrapper(runs[0]), updated_config, stage


class WandbRunWrapper(AbstractRunWrapper):
    def __init__(self, wandb_run) -> None:
        super().__init__()
        self.wadb_run = wandb_run
        
    def get_params(self):
        return self.wadb_run.config.get("in_params")
    
    def update_params(self, params):
        self.wandb_run.config['in_params'] = params
        self.wandb_run.update()
        
    def get_summary(self):
        return self.wandb_run.summary
    
    def get_local_checkpoint_path(self, phases):
        track_dir = self.get_params().get('experiment').get('tracking_dir') or 'experiments'
        checkpoint_path_group = os.path.join(track_dir, self.group, 'experiments')
        run_folder = list(filter(lambda x: str(self.id) in x, os.listdir(checkpoint_path_group)))
        checkpoint_path = None
        if 'epoch' in self.get_summary():
            ckpt = 'ckpt_latest.pth' if 'train' in phases else 'ckpt_best.pth'
            try:
                checkpoint_path = os.path.join(checkpoint_path_group, run_folder[0], 'files', ckpt)
            except IndexError as exc:
                logger.error(f"{self.id} not found in {checkpoint_path_group}")
                raise ValueError(
                    f"{self.id} not found in {checkpoint_path_group}"
                ) from exc
        return checkpoint_path
    
    @property
    def id(self):
        return self.wandb_run.id
    
    @property
    def group(self):
        return self.wandb_run.group