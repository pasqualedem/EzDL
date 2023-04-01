import torch.utils.data as torch_data

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
        if dataset_params.get('reduce'):
            self.train_loader = torch_data.DataLoader(
                torch_data.Subset(self.train_loader.dataset, range(500)),
                **dataset_params['trainloader']
            )
            self.test_loader = torch_data.DataLoader(
                torch_data.Subset(self.test_loader.dataset, range(20)),
                **dataset_params['testloader']
            )
            self.val_loader = torch_data.DataLoader(
                torch_data.Subset(self.val_loader.dataset, range(20)),
                **dataset_params['testloader']
            )        
    @property
    def size(self):
        return (3, 32, 32)