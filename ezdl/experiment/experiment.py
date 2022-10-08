from __future__ import annotations

import copy
import gc
import os
from typing import Mapping
from easydict import EasyDict

from ezdl.experiment.resume import ExpLog
from ezdl.experiment.run import Run
from ezdl.experiment.resume import get_interrupted_run, retrieve_run_to_resume
from ezdl.utils.grid import make_grid, linearize
from ezdl.utils.utilities import nested_dict_update, update_collection
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class GridSummary:
    def __init__(self,
                 total_runs,
                 total_runs_excl_grid,
                 total_runs_to_run,
                 total_runs_excl,
                 ):
        self.total_runs = total_runs
        self.total_runs_excl_grid = total_runs_excl_grid
        self.total_runs_to_run = total_runs_to_run
        self.total_runs_excl = total_runs_excl

    def update(self, d):
        self.total_runs = d.get("total_runs") or self.total_runs
        self.total_runs_excl_grid = d.get("total_runs_excl_grid") or self.total_runs_excl_grid
        self.total_runs_to_run = d.get("total_runs_to_run") or self.total_runs_to_run
        self.total_runs_excl = d.get("total_runs_excl") or self.total_runs_to_run


class ExpSettings(EasyDict):
    def __init__(self, *args, **kwargs):
        self.start_from_grid = 0
        self.start_from_run = 0
        self.resume = False
        self.resume_last = False
        self.tracking_dir = ""
        self.excluded_files = ""
        self.name = ""
        self.group = ""
        self.continue_with_errors = True
        super().__init__(*args, **kwargs)
        self.tracking_dir = self.tracking_dir or ""

    def update(self, e: ExpSettings, **f):
        if e is None:
            return
        self.start_from_grid = e.start_from_grid or self.start_from_grid
        self.start_from_run = e.start_from_run or self.start_from_run
        self.resume = e.resume or self.resume
        self.resume_last = e.resume_last or self.resume_last
        self.tracking_dir = e.tracking_dir or self.tracking_dir
        self.excluded_files = e.excluded_files or self.excluded_files
        self.group = e.group or self.group
        self.continue_with_errors = not e.continue_with_errors or self.continue_with_errors


class Status:
    STARTING = "starting"
    CRASHED = "crashed"
    FINISHED = "finished"

    def __init__(self, grid, run, params, n_grids, grid_len, wandb_run=None):
        self.grid = grid
        self.run = run
        self.params = params
        self.status = self.STARTING
        self.grid_len = grid_len
        self.n_grids = n_grids
        self.exception = None
        self.wandb_run = wandb_run

    def finish(self):
        self.params = {}
        self.status = self.FINISHED
        return self

    def crash(self, exception):
        self.status = self.CRASHED
        self.exception = exception
        return self


class StatusManager:
    def __init__(self, n_grids):
        self.n_grids = n_grids
        self.cur_status = None

    def new_run(self, grid, run, params, grid_len, wandb_run=None):
        self.cur_status = Status(grid=grid, run=run, params=params,
                                 n_grids=self.n_grids, grid_len=grid_len, wandb_run=wandb_run)
        return self.cur_status

    def update_run(self, wandb_run):
        self.cur_status.wandb_run = wandb_run
        return self.cur_status

    def finish_run(self):
        return self.cur_status.finish()

    def crash_run(self, exception):
        return self.cur_status.crash_run(exception)


class Experimenter:
    def __init__(self):
        self.gs = None
        self.exp_settings = ExpSettings()
        self.grids = None

    def calculate_runs(self, settings):
        base_grid = settings['parameters']
        other_grids = settings['other_grids']
        self.exp_settings = ExpSettings(settings['experiment'])

        complete_grids = [base_grid]
        if other_grids:
            complete_grids += \
                [nested_dict_update(copy.deepcopy(base_grid), other_run) for other_run in other_grids]
        logger.info(f'There are {len(complete_grids)} grids')

        self.grids, dot_elements = zip(*[make_grid(grid, return_cartesian_elements=True) for grid in complete_grids])
        dot_elements = list(dot_elements)
        if len(dot_elements) > 1:
            dot_elements[1:] = [list(dict(linearize(others) + dot).items()) for others, dot in
                                zip(other_grids, dot_elements[1:])]

        # Modify starting grid and run to manage the resume
        self.manage_resume()

        for i, grid in enumerate(self.grids):
            info = f'Found {len(grid)} runs from grid {i}'
            if i < self.exp_settings.start_from_grid:
                info += f', skipping grid {i} with {len(grid)} runs'
            logger.info(info)
        self.generate_grid_summary()

        if self.exp_settings.excluded_files:
            os.environ['WANDB_IGNORE_GLOBS'] = self.exp_settings.excluded_files
        return self.gs, self.grids, dot_elements

    def generate_grid_summary(self):
        total_runs = sum(len(grid) for grid in self.grids)
        total_runs_excl_grid = total_runs - sum([len(grid) for grid in self.grids[self.exp_settings.start_from_grid:]])
        total_runs_excl = total_runs_excl_grid + self.exp_settings.start_from_run
        total_runs_to_run = total_runs - total_runs_excl
        self.gs = GridSummary(
            total_runs=total_runs,
            total_runs_excl_grid=total_runs_excl_grid,
            total_runs_to_run=total_runs_to_run,
            total_runs_excl=total_runs_excl
        )

    def execute_runs_generator(self):
        track_dir = self.exp_settings['tracking_dir']
        if track_dir:
            os.makedirs(track_dir, exist_ok=True)
        exp_log = ExpLog(track_dir, self.exp_settings.name, self.exp_settings.group)
        starting_run = self.exp_settings.start_from_run
        status_manager = StatusManager(len(self.grids))
        if self.exp_settings.resume_last:
            logger.info("+ another run to finish!")
            grid_len = len(self.grids[self.exp_settings.start_from_grid])
            grid_list = [(i, j) for i in range(len(self.grids)) for j in range(len(self.grids[i]))]
            index = grid_list.index((self.exp_settings.start_from_grid, self.exp_settings.start_from_run))
            sg, sr = grid_list[index - 1]
            try:
                exp_log.insert_run(sg, sr)
                run = get_interrupted_run(self.exp_settings)
                yield status_manager.new_run(sg, sr, run.params, grid_len, run)
                logger.info(f'Running grid {sg} out of {len(self.grids) - 1}')
                logger.info(f'Running run {sr - 1} out of {grid_len} '
                            f'({sum([len(self.grids[k]) for k in range(sg)]) + sr} / {self.gs.total_runs - 1})')
                run.launch()
                exp_log.finish_run(sg, sr)
                yield status_manager.finish_run()
            except Exception as e:
                logger.error(f'Experiment {sg} failed with error {e}')
                exp_log.finish_run(sg, sr, crashed=True)
                if not self.exp_settings.continue_with_errors:
                    raise e
                yield status_manager.crash_run(e)
        for i in range(self.exp_settings.start_from_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.exp_settings.start_from_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    exp_log.insert_run(i, j)
                    yield status_manager.new_run(i, j, params, len(grid))
                    logger.info(f'Running grid {i} out of {len(self.grids) - 1}')
                    logger.info(f'Running run {j} out of {len(grid) - 1} '
                                f'({sum([len(self.grids[k]) for k in range(i)]) + j} / {self.gs.total_runs - 1})')
                    run = Run()
                    run.init({'experiment': {**self.exp_settings}, **params})
                    yield status_manager.update_run(run.seg_trainer.sg_logger.run)
                    run.launch()
                    exp_log.finish_run(i, j)
                    gc.collect()
                    yield status_manager.finish_run()
                except Exception as e:
                    logger.error(f'Experiment {i} failed with error {e}')
                    exp_log.finish_run(i, j, crashed=True)
                    if not self.exp_settings.continue_with_errors:
                        raise e
                    yield status_manager.crash_run(e)

    def execute_runs(self):
        for _ in self.execute_runs_generator():
            pass

    def manage_resume(self):
        if self.exp_settings.resume:
            self.exp_settings.start_from_grid, \
                self.exp_settings.start_from_run, \
                self.exp_settings.resume_last = retrieve_run_to_resume(self.exp_settings, self.grids)
        else:
            self.exp_settings.resume_last = False

    def update_settings(self, d):
        self.exp_settings = update_collection(self.exp_settings, d)
        if self.gs is None:
            return
        self.gs.update(self.exp_settings)
        if "resume" in d:
            self.manage_resume()
            self.generate_grid_summary()


def experiment(settings: Mapping, param_path: str = "local variable"):
    logger.info(f'Loaded parameters from {param_path}')

    experimenter = Experimenter()
    grid_summary, grids, cartesian_elements = experimenter.calculate_runs(settings)

    logger.info(f'Total runs found:              {grid_summary.total_runs}')
    logger.info(f'Total runs excluded by grids:  {grid_summary.total_runs_excl_grid}')
    logger.info(f'Total runs excluded:           {grid_summary.total_runs_excl}')
    logger.info(f'Total runs to run:             {grid_summary.total_runs_to_run}')
    experimenter.execute_runs()
