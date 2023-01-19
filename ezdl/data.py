import torch
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from super_gradients.training import utils as core_utils
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.datasets.samplers.repeated_augmentation_sampler import RepeatAugSampler
from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException
from super_gradients.training.utils import get_param

from ezdl.logger.text_logger import get_logger

default_dataset_params = {"batch_size": 64, "val_batch_size": 200, "test_batch_size": 200, "dataset_dir": "./data/",
                          "s3_link": None}
LIBRARY_DATASETS = {
    "cifar10": {'class': datasets.CIFAR10, 'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    "cifar100": {'class': datasets.CIFAR100, 'mean': (0.5071, 0.4865, 0.4409), 'std': (0.2673, 0.2564, 0.2762)},
    "SVHN": {'class': datasets.SVHN, 'mean': None, 'std': None}
}

logger = get_logger(__name__)


class DatasetInterface:
    """
    DatasetInterface - This class manages all of the "communiation" the Model has with the Data Sets
    """

    def __init__(self, dataset_params={}, train_loader=None, val_loader=None, test_loader=None, classes=None):
        """
        @param train_loader: torch.utils.data.Dataloader (optional) dataloader for training.
        @param test_loader: torch.utils.data.Dataloader (optional) dataloader for testing.
        @param classes: list of classes.

        Note: the above parameters will be discarded in case dataset_params is passed.

        @param dataset_params:

            - `batch_size` : int (default=64)

                Number of examples per batch for training. Large batch sizes are recommended.

            - `val_batch_size` : int (default=200)

                Number of examples per batch for validation. Large batch sizes are recommended.

            - `dataset_dir` : str (default="./data/")

                Directory location for the data. Data will be downloaded to this directory when getting it from a
                remote url.

            - `s3_link` : str (default=None)

                remote s3 link to download the data (optional).

            - `aug_repeat_count` : int (default=0)

                amount of repetitions (each repetition of an example is augmented differently) for each
                 example for the trainset.

        """

        self.dataset_params = core_utils.HpmStruct(**default_dataset_params)
        self.dataset_params.override(**dataset_params)

        self.trainset, self.valset, self.testset = None, None, None
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.classes = classes
        self.batch_size_factor = 1

    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                           test_batch_size=None, distributed_sampler: bool = False):
        """

        define train, val (and optionally test) loaders. The method deals separately with distributed training and standard
        (non distributed, or parallel training). In the case of distributed training we need to rely on distributed
        samplers.
        :param batch_size_factor: int - factor to multiply the batch size (usually for multi gpu)
        :param num_workers: int - number of workers (parallel processes) for dataloaders
        :param train_batch_size: int - batch size for train loader, if None will be taken from dataset_params
        :param val_batch_size: int - batch size for val loader, if None will be taken from dataset_params
        :param distributed_sampler: boolean flag for distributed training mode
        :return: train_loader, val_loader, classes: list of classes
        """
        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUTED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING

        aug_repeat_count = get_param(self.dataset_params, "aug_repeat_count", 0)
        if aug_repeat_count > 0 and not distributed_sampler:
            raise IllegalDatasetParameterException("repeated augmentation is only supported with DDP.")

        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = RepeatAugSampler(self.trainset,
                                             num_repeats=aug_repeat_count) if aug_repeat_count > 0 else DistributedSampler(
                self.trainset)
            val_sampler = DistributedSampler(self.valset)
            test_sampler = DistributedSampler(self.testset) if self.testset is not None else None
            train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params.batch_size * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params.val_batch_size * self.batch_size_factor
        if test_batch_size is None:
            test_batch_size = self.dataset_params.test_batch_size * self.batch_size_factor

        train_loader_drop_last = core_utils.get_param(self.dataset_params, 'train_loader_drop_last', default_val=False)

        cutmix = core_utils.get_param(self.dataset_params, 'cutmix', False)
        cutmix_params = core_utils.get_param(self.dataset_params, 'cutmix_params')

        # WRAPPING collate_fn
        train_collate_fn = core_utils.get_param(self.trainset, 'collate_fn')
        val_collate_fn = core_utils.get_param(self.valset, 'collate_fn')
        test_collate_fn = core_utils.get_param(self.testset, 'collate_fn')

        if cutmix and train_collate_fn is not None:
            raise IllegalDatasetParameterException("cutmix and collate function cannot be used together")

        if cutmix:
            # FIXME - cutmix should be available only in classification dataset. once we make sure all classification
            # datasets inherit from the same super class, we should move cutmix code to that class
            logger.warning("Cutmix/mixup was enabled. This feature is currently supported only "
                           "for classification datasets.")
            train_collate_fn = CollateMixup(**cutmix_params)

        # FIXME - UNDERSTAND IF THE num_replicas VARIBALE IS NEEDED
        # train_sampler = DistributedSampler(self.trainset,
        #                                    num_replicas=distributed_gpus_num) if distributed_sampler else None
        # val_sampler = DistributedSampler(self.valset,
        #                                   num_replicas=distributed_gpus_num) if distributed_sampler else None

        self.train_loader = torch.utils.data.DataLoader(self.trainset,
                                                        batch_size=train_batch_size,
                                                        shuffle=train_shuffle,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        sampler=train_sampler,
                                                        collate_fn=train_collate_fn,
                                                        drop_last=train_loader_drop_last)

        self.val_loader = torch.utils.data.DataLoader(self.valset,
                                                      batch_size=val_batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      sampler=val_sampler,
                                                      collate_fn=val_collate_fn)

        if self.testset is not None:
            self.test_loader = torch.utils.data.DataLoader(self.testset,
                                                           batch_size=test_batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           sampler=test_sampler,
                                                           collate_fn=test_collate_fn)

        self.classes = self.trainset.classes

    def get_data_loaders(self, **kwargs):
        """
        Get self.train_loader, self.val_loader, self.test_loader, self.classes.

        If the data loaders haven't been initialized yet, build them first.

        :param kwargs: kwargs are passed to build_data_loaders.

        """

        if self.train_loader is None and self.val_loader is None:
            self.build_data_loaders(**kwargs)

        return self.train_loader, self.val_loader, self.test_loader, self.classes

    def get_val_sample(self, num_samples=1):
        if num_samples > len(self.valset):
            raise Exception("Tried to load more samples than val-set size")
        if num_samples == 1:
            return self.valset[0]
        else:
            return self.valset[0:num_samples]

    def get_dataset_params(self):
        return self.dataset_params

    def print_dataset_details(self):
        logger.info("{} training samples, {} val samples, {} classes".format(len(self.trainset), len(self.valset),
                                                                             len(self.trainset.classes)))