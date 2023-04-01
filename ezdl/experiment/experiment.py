from __future__ import annotations

import copy
import gc
import os
import pandas as pd

from typing import Mapping
from easydict import EasyDict

from ezdl.experiment.resume import ExpLog
from ezdl.experiment.run import Run
from ezdl.experiment.resume import get_interrupted_run, retrieve_run_to_resume
from ezdl.utils.grid import linearized_to_string, make_grid, linearize
from ezdl.utils.utilities import nested_dict_update, update_collection
from ezdl.logger.text_logger import get_logger

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
        self.logger = None
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
        self.logger = e.logger or self.logger
        self.continue_with_errors = not e.continue_with_errors or self.continue_with_errors


class Status:
    STARTING = "starting"
    CRASHED = "crashed"
    FINISHED = "finished"

    def __init__(self, grid, run, params, n_grids, grid_len, run_name=None, run_url=None):
        self.grid = grid
        self.run = run
        self.params = params
        self.status = self.STARTING
        self.grid_len = grid_len
        self.n_grids = n_grids
        self.exception = None
        self.run_name = run_name
        self.run_url = run_url

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

    def new_run(self, grid, run, params, grid_len, run_name=None, run_url=None):
        self.cur_status = Status(grid=grid, run=run, params=params,
                                 n_grids=self.n_grids, grid_len=grid_len,
                                 run_name=run_name, run_url=run_url)
        return self.cur_status

    def update_run(self, run_name, run_url):
        self.cur_status.run_name = run_name
        self.cur_status.run_url = run_url
        return self.cur_status

    def finish_run(self):
        return self.cur_status.finish()

    def crash_run(self, exception):
        return self.cur_status.crash(exception)


class Experimenter:
    EXP_FINISH_SEP = "#"*50 + " FINISHED " + "#"*50 + "\n"
    EXP_CRASHED_SEP = "|\\"*50 + "CRASHED" + "|\\"*50 + "\n"

    def __init__(self):
        self.gs = None
        self.exp_settings = ExpSettings()
        self.grids = None

    def calculate_runs(self, settings):
        base_grid = settings['parameters']
        other_grids = settings['other_grids']
        self.exp_settings = ExpSettings(settings['experiment'])

        print('\n' + '='*100)
        complete_grids = [base_grid]
        if other_grids:
            complete_grids += \
                [nested_dict_update(copy.deepcopy(base_grid), other_run) for other_run in other_grids]
        logger.info(f'There are {len(complete_grids)} grids')

        self.grids, dot_elements = zip(*[make_grid(grid, return_cartesian_elements=True) for grid in complete_grids])
        # WARNING: Grids' objects have the same IDs!
        dot_elements = list(dot_elements)
        if len(dot_elements) > 1:
            dot_elements[1:] = [list(dict(linearize(others) + dot).items()) for others, dot in
                                zip(other_grids, dot_elements[1:])]

        # Modify starting grid and run to manage the resume
        self.manage_resume()

        for i, grid in enumerate(self.grids):
            info = f'Found {len(grid)} runs from grid {i}'
            last_grid = self.exp_settings.start_from_grid if self.exp_settings.start_from_grid is not None else len(self.grids)
            if i < last_grid:
                info += f', skipping grid {i} with {len(grid)} runs'
            logger.info(info)
        self.generate_grid_summary()

        # logger.info(f'Total runs found:              {self.gs.total_runs}')
        # logger.info(f'Total runs excluded by grids:  {self.gs.total_runs_excl_grid}')
        # logger.info(f'Total runs excluded:           {self.gs.total_runs_excl}')
        # logger.info(f'Total runs to run:             {self.gs.total_runs_to_run}')

        if self.exp_settings.excluded_files:
            os.environ['WANDB_IGNORE_GLOBS'] = self.exp_settings.excluded_files
            
        print_preview(self, self.gs, self.grids, dot_elements)
        print('='*100 + '\n')
        
        return self.gs, self.grids, dot_elements

    def generate_grid_summary(self):
        total_runs = sum(len(grid) for grid in self.grids)
        if self.exp_settings.start_from_grid is None:
            total_runs_excl_grid = total_runs - len(self.grids[-1])
            total_runs_excl = total_runs
        else:
            total_runs_excl_grid = total_runs - sum(
                len(grid) for grid in self.grids[self.exp_settings.start_from_grid :]
            )
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
            grid_list = [(i, j) for i in range(len(self.grids)) for j in range(len(self.grids[i]))]
            if self.exp_settings.start_from_grid is None:
                grid_len = len(self.grids[-1])
                sg, sr = grid_list[-1]
            else:
                grid_len = len(self.grids[self.exp_settings.start_from_grid])
                index = grid_list.index((self.exp_settings.start_from_grid, self.exp_settings.start_from_run))
                sg, sr = grid_list[index - 1]
            try:
                exp_log.insert_run(sg, sr)
                run = get_interrupted_run(self.exp_settings)
                yield status_manager.new_run(sg, sr, run.params, grid_len, run.seg_trainer.sg_logger.name, run.seg_trainer.sg_logger.url)
                logger.info(f'Running grid {sg} out of {len(self.grids) - 1}')
                logger.info(
                    f'Running run {sr - 1} out of {grid_len} ({sum(len(self.grids[k]) for k in range(sg)) + sr} / {self.gs.total_runs - 1})'
                )
                run.launch()
                print(self.EXP_FINISH_SEP)
                exp_log.finish_run(sg, sr)
                yield status_manager.finish_run()
            except Exception as ex:
                logger.error(f'Experiment {sg} failed with error {ex}')
                print(self.EXP_CRASHED_SEP)
                exp_log.finish_run(sg, sr, crashed=True)
                if not self.exp_settings.continue_with_errors:
                    raise ex
                yield status_manager.crash_run(ex)
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
                    logger.info(
                        f'Running run {j} out of {len(grid) - 1} ({sum(len(self.grids[k]) for k in range(i)) + j} / {self.gs.total_runs - 1})'
                    )
                    run = Run()
                    run.init({'experiment': {**self.exp_settings}, **params})
                    yield status_manager.update_run(run.seg_trainer.sg_logger.name, run.seg_trainer.sg_logger.url)
                    run.launch()
                    print(self.EXP_FINISH_SEP)
                    exp_log.finish_run(i, j)
                    gc.collect()
                    yield status_manager.finish_run()
                except Exception as ex:
                    logger.error(f'Experiment {i} failed with error {ex}')
                    print(self.EXP_CRASHED_SEP)
                    exp_log.finish_run(i, j, crashed=True)
                    if not self.exp_settings.continue_with_errors:
                        raise ex
                    yield status_manager.crash_run(ex)

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

    experimenter.execute_runs()


def preview(settings: Mapping, param_path: str = "local variable"):
    print(f'Loaded parameters from {param_path}')

    experimenter = Experimenter()
    _, _, _ = experimenter.calculate_runs(settings)
    
    
def print_preview(experimenter, grid_summary, grids, cartesian_elements):
    summary_series = pd.concat([pd.Series(grid_summary), pd.Series(experimenter.exp_settings.__dict__)])
    summary_string = f"\n{summary_series.to_string()}\n"
    
    dfs = [pd.DataFrame(linearized_to_string(dot_element), columns=[f"Grid {i}", f"N. runs: {len(grid)}"])
           for i, (dot_element, grid) in enumerate(zip(cartesian_elements, grids))]
    mark_grids = "\n\n".join(df.to_string(index=False) for df in dfs)
    mark_grids = "Most important parameters for each grid \n" + mark_grids
    logger.info(f"\n{summary_string}\n{mark_grids}")
