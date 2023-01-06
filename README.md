# EzDL
## Installation
Recommended to create a Python virtual environment

```bash
pip install ezdl
```

## Usage
```bash
ezdl <ACTION>
```
**mandatory arguments**
	
	action:	Choose the action to do perform: 
			experiment, resume_run, complete, manipulate, app

**optional arguments**:

    -h, --help            show this help message and exit
    --resume              Resume the experiment
    -d DIR, --dir DIR     Set the local tracking directory
    -f FILE, --file FILE  Set the config file
    --grid GRID           Select the first grid to start from
    --run RUN             Select the run in grid to start from
    --filters FILTERS     Filters to query in the resuming mode
    -s STAGE, --stage STAGE
                          Stages to execute in the resuming mode
    -p PATH, --path PATH  Path to the tracking url in the resuming mode
    --subset SUBSET       Subset chosen for preprocessing
		  
### Parameter file
YAML file that contains all parameters necessary to the exepriment to be run.

It must have 3 top keys:
- experiment
- parameters
- other_grids

Let's see the CIFAR10 example:

```yaml
experiment:
  # It contains all the about the grids and the group of runs:
  name: Classification # name of the logger platform experiment
  group: FirstGroup # name of group of experiments for the logger platform
  continue_with_errors: False # continue with other runs even if a run fails
  start_from_grid: 0 # skip grids in the grid search
  start_from_run: 0 # skip runs from the selected grid
  logger: clearml # logger platform to use
  tracking_dir: './examplesExp' # dir where results will be saved
  entity: null # Wandb entity (username)
  excluded_files:  null # glob of files to not upload to Wandb

parameters:
  # Contains the parameters to build the grid.
  # Each value should be a dict or a list
  tags: [[mytag1, mytag2]] # list of tags to attach to the run in logger platform
  phases: [[train, test]] # list of phases
  dataset_interface: [examples/cifar10/Cifar10] # Path to the dataset interface class

  train_params:
    loss:
      name: [cross_entropy] # class loss name
      params:
    seed: [42] # random seed to set
    max_epochs: [ 1, 2 ]
    initial_lr: [ 0.0001 ]
    optimizer: [ Adam ]
    zero_weight_decay_on_bias_and_bn: [ True ]
    average_best_models: [ False ]
    greater_metric_to_watch_is_better: [ False ]
    metric_to_watch: [ loss ]
    freeze_pretrained: [ False ] # freeze the loaded pretrained weights
    # Other parameters relative to Super-Gradients (see their docs)

  early_stopping:
    patience: [ 10 ] # number of epochs before stopping
    monitor: [ loss ] # metric to monitor
    mode: [ min ] # metric to be minimized or maximized

  train_metrics:
    # list of metrics to load from PyTorch metrics
    # where the values are their parameters used for training
    f1:
      average: [macro]
      num_classes: [10]
      mdmc_average: [global]
  test_metrics:
    # list of metrics to load from PyTorch metrics
    # where the values are their parameters used for validation and test
    f1:
      num_classes: [10]
      average: [macro]
      mdmc_average: [global]
    precision:
      average: [macro]
      num_classes: [10]
      mdmc_average: [global]
    recall:
      average: [macro]
      num_classes: [10]
      mdmc_average: [global]

  model:
    name: [resnet18]  # path to model class or model name contained in EzDL or super-gradients
    params: # model parameters
      pretrained_weights: [imagenet]
      num_classes: [10]

  dataset: # parameters depending on the class you defined for the dataset
    channels: [["R", "G", "B"]]
    num_classes: [10]
    trainset:
    testset:
    trainloader:
      batch_size: [8]
      num_workers: [0]
    testloader:
      batch_size: [8]
      num_workers: [0]

other_grids:
  # List of supplementary grids (can be empty) in which the parameters defined will override the first grid.
```
