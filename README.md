# EzDL
## Installation
Recommended to create a Python virtual environment

    pip install ezdl

## Usage

    .

**mandatory arguments**
	
	action:	Choose the action to do perform: 
			experiment, resume, resume_run, complete, preprocess, manipulate, app

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


```yaml
experiment:
  # It contains all the about the grids and the group of runs:
  name: exp-name # name of the Wandb experiment
  group: exp-group # name of group of experiments for Wandb
  continue_with_errors: True # continue with other runs even if a run fails
  start_from_grid: 0 # skip grids in the grid search
  start_from_run: 0 # skip runs from the selected grid
  tracking_dir: '.' # dir where results will be saved
  entity: myUsername # Wandb entity (username)
  excluded_files: null # glob of files to not upload to Wandb

parameters:
  # Contains the parameters to build the grid.
  # Each value should be a dict or a list
  tags: [[mytag1, mytag2]] # list of tags to attach to the run in Wandb
  phases: [[train, test]] # list of phases
  dataset_interface: [package/module/InterfaceClass] # Path to the dataset interface class
  
  train_params:
    loss:
      name: [CEloss] # class loss name
      params: 
        # params to be passed to class loss
    seed: [42] # random seed to set
  freeze_pretrained: [False] # freeze the loaded pretrained weights
  # Other parameters relative to Super-Gradients (see their docs)
  
  early_stopping:
    enabled: [True] # tells if to enable the early stopping (True, False)
    params:
      patience: [5] # number of epochs before stopping
      monitor: [loss] # metric to monitor
      mode: [min] # min or max
  
  train_metrics: 
    # list of metrics to load from PyTorch metrics 
    # where the values are their parameters used for training
  test_metrics: 
    # list of metrics to load from PyTorch metrics 
    # where the values are their parameters used for validation and test
  
  model:
    name: [package/module/MyModel] # path to model class or model name contained in EzDL or super-gradients
    params: # model parameters
  
  dataset: # parameters depending on the class you defined for the dataset
  
other grids:
  # List of supplementary grids (can be empty) in which the parameters defined will override the first grid.
  # For example
  - 
    train_params:
      loss:
      name: [AnotherLoss] # class loss name
      params: 
        # params to be passed to class loss
```