from ezdl.logger.clearml_logger import ClearMLLogger
from ezdl.logger.wandb_logger import WandBSGLogger

LOGGERS = {
    'wandb': WandBSGLogger,
    'clearml': ClearMLLogger,
}