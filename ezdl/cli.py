import functools
import json
import click
from easydict import EasyDict

from ezdl.utils.utilities import update_collection, load_yaml

import os
os.environ['CRASH_HANDLER'] = "FALSE"


@click.group()
def main():
    pass


def exp_options(f):
    options = [
          click.option("--resume", is_flag=True, help='Resume the experiment'),
          click.option("--file", default="", help='Set the config file'),
          click.option("--dir", default="", help='Set the local tracking directory'),
          click.option("--grid", type=int, default=0, help="Select the first grid to start from"),
          click.option("--run", type=int, default=0, help="Select the run in grid to start from"),
    ]
    return functools.reduce(lambda x, opt: opt(x), options, f)


@main.command("preview")
@exp_options
def experiment(resume, file, dir, grid, run):
    from ezdl.experiment.experiment import preview
    param_path = file or 'parameters.yaml'
    settings = load_yaml(param_path)
    settings['experiment'] = update_collection(settings['experiment'], resume, key='resume')
    settings['experiment'] = update_collection(settings['experiment'], grid, key='start_from_grid')
    settings['experiment'] = update_collection(settings['experiment'], run, key='start_from_run')
    settings['experiment'] = update_collection(settings['experiment'], dir, key='tracking_dir')
    preview(settings)


@main.command("experiment")
@exp_options
def experiment(resume, file, dir, grid, run):
    from ezdl.experiment.experiment import experiment
    param_path = file or 'parameters.yaml'
    settings = load_yaml(param_path)
    settings['experiment'] = update_collection(settings['experiment'], resume, key='resume')
    settings['experiment'] = update_collection(settings['experiment'], grid, key='start_from_grid')
    settings['experiment'] = update_collection(settings['experiment'], run, key='start_from_run')
    settings['experiment'] = update_collection(settings['experiment'], dir, key='tracking_dir')
    experiment(settings)


@main.command("resume_run")
@click.option("--file", default="", help='Set the config resume file')
@click.option("--filters", type=json.loads, help="Filters to query in the resuming mode")
@click.option('-s', "--stage", type=json.loads, help="Stages to execute")
@click.option('-p', "--path", type=str, help="Path to the tracking url")
def resume_run(file, filters, stage, path):
    from ezdl.experiment.resume import resume_set_of_runs
    param_path = file or 'resume.yaml'
    settings = load_yaml(param_path)
    settings['runs'][0]['filters'] = update_collection(settings['runs'][0]['filters'], filters)
    settings['runs'][0]['stage'] = update_collection(settings['runs'][0]['stage'], stage)
    settings = update_collection(settings, path, key="path")
    resume_set_of_runs(settings)


@main.command("complete")
@click.option("--file", default="", help='Set the config resume file')
def complete(file):
    from ezdl.experiment.resume import complete_incompleted_runs
    param_path = file or 'resume.yaml'
    settings = load_yaml(param_path)
    complete_incompleted_runs(settings)


@main.command("manipulate")
def manipulate():
    from ezdl.wandb_manip import manipulate
    manipulate()


@main.command("app")
@exp_options
def app(resume, file, dir, grid, run):
    from ezdl.app import frontend
    args = EasyDict(
        resume=resume,
        file=file,
        dir=dir,
        grid=grid,
        run=run
    )
    frontend(args)


@main.command("complexity")
@click.option("--file", default="", help='Set the file to load the models from')
def complexity(file):
    from ezdl.complexity import complexity as cpx
    cpx(file)
