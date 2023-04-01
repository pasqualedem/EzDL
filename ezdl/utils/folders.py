import os

def get_checkpoints_dir_path(project_name: str, group_name: str, ckpt_root_dir: str = None):
    """Creating the checkpoint directory of a given experiment.
    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Local root directory path where all experiment logging directories will
                                reside. When none is give, it is assumed that pkg_resources.resource_filename('checkpoints', "")
                                exists and will be used.
    :return:                    checkpoints_dir_path
    """
    if ckpt_root_dir:
        return os.path.join(ckpt_root_dir, project_name, group_name)