import collections
import io
import os
from io import StringIO
from typing import Any, Mapping

import collections.abc
from ruamel.yaml import YAML

#
# def setup_mlflow(exp_name: str, description: str) -> str:
#     """
#     Setup mlflow tracking server and experiment
#     """
#     tracking_uri = registry_uri = "http://localhost:5000"
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_registry_uri(registry_uri)
#     print('URI set!')
#     client = MlflowClient(tracking_uri, registry_uri)
#     exp_info = client.get_experiment_by_name(exp_name)
#     exp_id = exp_info.experiment_id if exp_info else \
#         client.create_experiment(exp_name, tags={'mlflow.note.content': description})
#     print('Experiment set')
#     return exp_id
#
#
# def mlflow_server(mlruns_folder=False):
#     """
#     Start mlflow server
#     """
#     cmd = ["mlflow", 'server']
#     if mlruns_folder:
#         cmd.extend(['--backend-store-uri', f'file:{mlruns_folder}'])
#     cmd_env = cmd_env = os.environ.copy()
#     child = subprocess.Popen(
#         cmd, env=cmd_env, universal_newlines=True, stdin=subprocess.PIPE
#     )
#     return child
#
#
# class MLRun(MlflowClient):
#     def __init__(self, exp_name: str, description: str, run_id: Optional[str] = None):
#         exp_id = setup_mlflow(exp_name, description)
#         super().__init__(mlflow.get_tracking_uri(), mlflow.get_registry_uri())
#         if run_id is None:
#             self.run = self.create_run(experiment_id=exp_id)
#         else:
#             self.run = self.get_run(run_id)
#
#     def log_params(self, params: Mapping):
#         params = mlflow_linearize(params)
#         for k, v in params.items():
#             self.log_param(k, v)
#
#     def log_metrics(self, metrics: Mapping):
#         for k, v in metrics.items():
#             self.log_metric(k, v)
#
#     def log_param(self, key: str, value: Any, run_id: str = None) -> None:
#         if run_id is None:
#             run_id = self.run.info.run_id
#         super().log_param(run_id, key, value)
#
#     def log_metric(self, key: str, value: Any, run_id: str = None,
#                    timestamp: Optional[int] = None,
#                    step: Optional[int] = None) -> None:
#
#         value = value.item() if type(value) == torch.Tensor else value
#
#         if run_id is None:
#             run_id = self.run.info.run_id
#         super().log_metric(run_id, key, value)


def mlflow_linearize(dictionary: Mapping) -> Mapping:
    """
    Linearize a nested dictionary concatenating keys in order to allow mlflow parameters recording.

    :param dictionary: nested dict
    :return: one level dict
    """
    exps = {}
    for key, value in dictionary.items():
        if isinstance(value, collections.abc.Mapping):
            exps = {**exps,
                    **{key + '.' + lin_key: lin_value for lin_key, lin_value in mlflow_linearize(value).items()}}
        else:
            exps[key] = value
    return exps


def nested_dict_update(d, u):
    if u is not None:
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = nested_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
    return d


def dict_to_yaml_string(mapping: Mapping) -> str:
    """
    Convert a nested dictionary or list to a string
    """
    string_stream = StringIO()
    yaml = YAML()
    yaml.dump(mapping, string_stream)
    output_str = string_stream.getvalue()
    string_stream.close()
    return output_str


def yaml_string_to_dict(s):
    return YAML(typ='safe', pure=True).load(s)


def load_yaml(path, return_string=False):
    if hasattr(path, "readlines"):
        d = YAML(typ='safe', pure=True).load(path)
        if return_string:
            path.seek(0)
            return d, path.read().decode('utf-8')
    with open(path, 'r') as param_stream:
        d = YAML(typ='safe', pure=True).load(param_stream)
        if return_string:
            param_stream.seek(0)
            return d, str(param_stream.read())
    return d


def dict_to_yaml(d: Mapping):
    stream = io.StringIO()
    YAML().dump(d, stream)
    stream.seek(0)
    return stream.read()


def values_to_number(collec) -> Any:
    """
    Convert all values in a dictionary or list to numbers
    """
    if isinstance(collec, collections.abc.Mapping):
        for key, value in collec.items():
            collec[key] = values_to_number(value)
    elif isinstance(collec, list):
        return [values_to_number(v) for v in collec]
    elif isinstance(collec, str) and collec.startswith('00'):
        return collec
    elif isinstance(collec, int):
        return collec
    else:
        try:
            return float(collec)
        except (ValueError, TypeError):
            pass
    return collec


def filter_none(collec) -> Any:
    """
    Filter out None values from a dictionary or list
    """
    if isinstance(collec, collections.abc.Mapping):
        for key in list(collec.keys()):
            value = filter_none(collec[key])
            if value is None:
                del collec[key]
            else:
                collec[key] = value
        return collec
    elif isinstance(collec, list):
        return [filter_none(v) for v in collec if v is not None]
    else:
        return collec


def update_collection(collec, value, key=None):
    if isinstance(collec, dict):
        if isinstance(value, dict):
            for keyv, valuev in value.items():
                collec = update_collection(collec, valuev, keyv)
        elif key is not None:
            if value is not None:
                collec[key] = value
        else:
            collec = {**collec, **value} \
                if value is not None else collec
    else:
        collec = value if value is not None else collec
    return collec


def safe_execute(default, exception, function, *args):
    try:
        return function(*args)
    except exception:
        return default


def get_module_class_from_path(path):
    path = os.path.normpath(path)
    splitted = path.split(os.sep)
    module = ".".join(splitted[:-1])
    cls = splitted[-1]
    return module, cls


def recursive_get(dictionary, *keys):
    """
    Get a value from nested dicts passing the keys
    :param dictionary: The nested dicts
    :param keys: sequence of keys
    :return: the value
    """
    if len(keys) == 1:
        return dictionary.get(keys[0])
    else:
        if dictionary.get(keys[0]):
            return recursive_get(dictionary[keys[0]], *keys[1:])

