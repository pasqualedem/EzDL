from learning.clearml_logger import ClearMLLogger
from learning.wandb_logger import WandBSGLogger

LOGGERS = {
    'wandb': WandBSGLogger,
    'clearml': ClearMLLogger,
}