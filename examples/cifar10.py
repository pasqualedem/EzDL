from super_gradients.training import dataloaders

from ezdl.data import DatasetInterface


class Cifar10(DatasetInterface):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        self.train_loader = dataloaders.get("cifar10_train", dataset_params=dataset_params['trainset'],
                                            dataloader_params=dataset_params['trainloader'])
        self.test_loader = dataloaders.get("cifar10_val", dataset_params=dataset_params['testset'],
                                           dataloader_params=dataset_params['testloader'])
        self.val_loader = dataloaders.get("cifar10_val", dataset_params=dataset_params['testset'],
                                           dataloader_params=dataset_params['testloader'])