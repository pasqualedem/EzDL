import datetime
import os

import numpy as np
import wandb
import pandas as pd

from ezdl.experiment.run import Run

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def resume_set_of_runs(settings, post_filters=None):
    queries = settings['runs']
    path = settings['path']
    for query in queries:
        filters = query['filters']
        stage = query['stage']
        updated_config = query['updated_config']
        api = wandb.Api()
        runs = api.runs(path=path, filters=filters)
        runs = list(filter(post_filters, runs))
        print("Runs to resume:")
        for run in runs:
            print(f"{run.group} \t - \t {run.name}")
        if len(runs) == 0:
            logger.error(f'No runs found for query {filters} with post_filters: {post_filters}')
        for run in runs:
            resume_run(run, updated_config, stage)


def complete_incompleted_runs(settings):
    print("Going on to complete runs!")
    resume_set_of_runs(settings, lambda x: 'f1' not in x.summary)


def resume_run(wandb_run, updated_config, stage):
    to_resume_run = Run()
    to_resume_run.resume(wandb_run=wandb_run, updated_config=updated_config, phases=stage)
    to_resume_run.launch()


def get_interrupted_run(input_settings):
    namespace = input_settings["name"]
    group = input_settings["group"]
    last_run = wandb.Api().runs(path=namespace, filters={"group": group,}, order="-created_at")[0]
    filters = {"group": group, "name": last_run.id}
    stage = ["train", "test"]
    updated_config = None
    api = wandb.Api()
    runs = api.runs(path=namespace, filters=filters)
    if len(runs) == 0:
        raise RuntimeError("No runs found")
    if len(runs) > 1:
        raise EnvironmentError("More than 1 run???")
    to_resume_run = Run()
    to_resume_run.resume(wandb_run=runs[0], updated_config=updated_config, phases=stage)
    return to_resume_run


def retrieve_run_to_resume(settings, grids):
    grid_list = [(i, j) for i in range(len(grids)) for j in range(len(grids[i]))]
    dir = settings['tracking_dir']
    exp_log = ExpLog(track_dir=dir, wandb_path=settings['name'], group=settings['group'])
    i, j, finished = exp_log.get_last_run()
    if i is None:
        return 0, 0, False
    index = grid_list.index((i, j))
    try:
        start_grid, start_run = grid_list[index + 1]  # Skip interrupted run
    except IndexError as e:
        if finished:
            logger.info(e)
            raise ValueError('No experiment to resume!!')
        else:
            return len(grids), None, True
    resume_last = not finished
    return start_grid, start_run, resume_last


class ExpLog:
    STARTED = 'started'
    FINISHED = 'finished'
    CRASHED = 'crashed'
    FORMAT = "%d-%m-%YT%H:%M:%S"
    FILENAME = 'exp_log.csv'

    def __init__(self, track_dir, wandb_path, group):
        """

        :param track_dir: Directory where experiments are saves
        :param wandb_path: wand path
        :param group: wandb group
        """
        self.path = os.path.join(track_dir if track_dir is not None else '', self.FILENAME)
        self.wandb_path = wandb_path
        self.group = group
        try:
            self.exp_log = pd.read_csv(self.path)
        except FileNotFoundError:
            self.exp_log = pd.DataFrame(columns=['path', 'group', 'grid', 'run', 'crashed', 'started', 'finished'])

    def get_run_condition(self, grid, run):
        return (
                (self.exp_log.path == self.wandb_path) &
                (self.exp_log.group == self.group) &
                (self.exp_log.grid == grid) &
                (self.exp_log.run == run)
        )

    def cur_time(self):
        return datetime.datetime.now().strftime(self.FORMAT)

    def _insert_run(self, grid, run, started=True, finished=False):
        run_row = pd.DataFrame({
            'path': [self.wandb_path],
            'group': [self.group],
            'grid': [grid],
            'run': [run],
            'crashed': [False],
            'started': [self.cur_time() if started else None],
            'finished': [self.cur_time() if finished else None]
        })
        self.exp_log = pd.concat([self.exp_log, run_row])
        self.exp_log = self.exp_log.reset_index(drop=True)

    def insert_run(self, grid, run):
        run_condition = self.get_run_condition(grid, run)
        run_row = self.exp_log[run_condition]
        if run_row.empty:
            self._insert_run(grid, run)
        else:
            self.exp_log.loc[self.exp_log[run_condition].index, "started"] = self.cur_time()
            self.exp_log.loc[self.exp_log[run_condition].index, "crashed"] = False
            self.exp_log.loc[self.exp_log[run_condition].index, "finished"] = None

        self.save()

    def finish_run(self, grid, run, crashed=False):
        run_condition = self.get_run_condition(grid, run)
        run_row = self.exp_log[run_condition]
        if run_row.empty:
            logger.warning("Finishing a non started run on the CSV")
            self._insert_run(grid, run, finished=True)
        else:
            self.exp_log.loc[self.exp_log[run_condition].index, "crashed"] = crashed
            self.exp_log.loc[self.exp_log[run_condition].index, "finished"] = self.cur_time()
        self.save()

    def get_last_run(self):
        cur_exp_dataframe = self.exp_log[
            (self.exp_log.path == self.wandb_path) &
            (self.exp_log.group == self.group)
        ]
        last_run = cur_exp_dataframe[(cur_exp_dataframe.grid == cur_exp_dataframe.grid.max())].reset_index(drop=True)
        last_run = last_run[(last_run.run == last_run.run.max())]
        if len(last_run) > 1:
            raise Exception("More than one last run")
        elif len(last_run) == 0:
            logger.warning("This experiment is never started, no runs to resume")
            return None, None, False
        last_run = last_run.reset_index(drop=True).loc[0]  # Get the Series
        return last_run.grid, last_run.run, not pd.isnull(last_run.finished) and not last_run.crashed

    def save(self):
        self.exp_log.to_csv(self.path, index=False)

    def __repr__(self):
        return f"Local path : {self.path} \n" \
               f"Wandb path : {self.wandb_path} \n" \
               f"Wanfb group: {self.group} \n" + \
            self.exp_log.__repr__()
