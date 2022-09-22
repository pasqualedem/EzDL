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

#### experiment

It contains all the about the grids and the group of runs:

- **name**: name of the Wandb experiment
- **group**: name of group of experiments for Wandb
- **continue_with_errors**: continue with other runs even if a run fails
- **start_from_grid**: skip grids in the grid search
- **start_from_run**: skip runs from the selected grid
- **tracking_dir**: dir where results will be saved
- **entity**: Wandb entity (username)
- **excluded_files**: glob of files to not upload to Wandb

#### parameters

Contains the parameters to build the grid.
Each value should be a dict or a list

They are:
  - **tags**: list of tags to attach to the run in Wandb
  - **phases**: list of phases (train, test)
  - **dataset_interface**: Path to the dataset interface class with the class name

  - **train_params**
    - **loss**:
      - **name**: class loss name
      - **params**: params to be passed to class loss
    - **seed**: random seed to set
    - **freeze_pretrained**: freeze the loaded pretrained weights
    - Other parameters relative to Super-Gradients (see their docs)

  - **early_stopping**:
    - **enabled**: tells if to enable the early stopping (True, False)
    - **params**:
      - **patience**: number of epochs before stopping
      - **monitor**: metric to monitor
      - **mode**: min or max

  - **train_metrics**: list of metrics to load from PyTorch metrics where the values are their parameters
  

  - **model**:
    - **name**: path to model class or model name contained in EzDL or super-gradients
    - **params**: model parameters

  **dataset**: parameters depending on the class you defined for the dataset
  
#### other grids
List of supplementary grids (can be empty) in which the parameters defined will override the first grid.