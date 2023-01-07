import streamlit as st
import wandb
import os

from ruamel.yaml.scanner import ScannerError
from streamlit_ace import st_ace
from streamlit.web import bootstrap
from streamlit import config as _config

from ezdl.app.utils import ManipulationInputs
from ezdl.app.markdown import exp_summary_builder, grid_summary_builder, MkFailures, logger_run_link, format_to_delete
from ezdl.experiment.experiment import ExpSettings, Experimenter, Status

from ezdl.utils.utilities import load_yaml, dict_to_yaml, yaml_string_to_dict
from ezdl.wandb_manip import update_metadata, fix_string_param, remove_key, delete_files, delete_artifacts, update_config
from ezdl.logger import LOGGERS

STREAMLIT_AGGRID_URL = "https://github.com/PablocFonseca/streamlit-aggrid"


def DEFAULTS(key=None):
    if key is None:
        return "", "", "", "", "", 0, 0, []
    if isinstance(key, dict):
        return {v: DEFAULTS(k) for k, v in key.items()}
    if isinstance(key, list):
        return {k: DEFAULTS(k) for k in key}
    if key in ['start_from_grid', 'start_from_run']:
        return 0
    elif key == "flags":
        return []
    else:
        return ""


def loggers():
    return list(LOGGERS.keys())


def logger_value(logger):
    return loggers().index(logger) if logger is not None else 0


def update_from_gui(group, logger, track_dir, start_grid, start_run, resume, continue_errors):
    if st.session_state.experimenter is None:
        st.session_state.experimenter = Experimenter()
    st.session_state.experimenter.update_settings({
        "resume": resume,
        "continue_with_errors": continue_errors,
        "start_from_grid": start_grid,
        "start_from_run": start_run,
        "tracking_dir": track_dir,
        "logger": logger,
        "group": group,
    })
    st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)


def set_file():
    st.session_state.experimenter = Experimenter()
    st.session_state.experimenter.exp_settings = ExpSettings()
    st.session_state.mk_summary = ""
    st.session_state.mk_params = ""
    file = st.session_state.parameter_file
    if file is None:
        return
    settings, st_string = load_yaml(file, return_string=True)
    st.session_state.settings_string = st_string
    set_settings(settings)


def edit_file():
    st.session_state.experimenter = Experimenter()
    st.session_state.experimenter.exp_settings = ExpSettings()
    st.session_state.mk_summary = ""
    st.session_state.mk_params = ""
    settings = yaml_string_to_dict(st.session_state.settings_string)
    set_settings(settings)


def set_settings(settings):
    sm, grids, dots = st.session_state.experimenter.calculate_runs(settings)
    st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)
    st.session_state.mk_params = grid_summary_builder(grids, dots)


class Interface:
    def __init__(self, parameter_file, exp_settings: ExpSettings = None, share: bool = True):
        self.mk_run_link = None
        self.mk_cur_params = None
        self.failures = None
        self.mk_failures = None
        self.current_params = None
        if "experimenter" not in st.session_state:
            self.experimenter = Experimenter()
            st.session_state.experimenter = self.experimenter
            self.experimenter.exp_settings.update(exp_settings)
            st.session_state.mk_summary = exp_summary_builder(self.experimenter)
        else:
            self.experimenter = st.session_state.experimenter

        st.set_page_config(
            layout="wide", page_icon="🖱️", page_title="EzDL"
        )
        st.title("EzDL")
        tab1, tab2, = st.tabs(["Training", "Manipulation"])
        with tab1:
            self.experiment_interface()
        with tab2:
            self.manipulation_interface()

    def experiment_interface(self):
        st.write(
            """Train your model!"""
        )
        es = self.experimenter.exp_settings
        with st.sidebar:
            with st.form("exp_form"):
                self._side_form(es)
            self.parameter_file = st.file_uploader("Parameter file", on_change=set_file, key="parameter_file")
        if "parameter_file" in st.session_state and st.session_state.parameter_file:
            self._experiment_board(es)

    def _side_form(self, es):
        self.form_path = st.text_input("Experiment name (path)", value=es.name)
        self.form_group = st.text_input("Experiment group", value=es.group)
        self.form_logger = st.selectbox("Logger", options=loggers(), index=logger_value(es.logger))
        self.form_track_dir = st.text_input("Tracking directory", value=es.tracking_dir)
        self.form_grid = st.number_input("Start from grid", min_value=0, value=es.start_from_grid)
        self.form_run = st.number_input("Start from run", min_value=0, value=es.start_from_run)
        self.form_resume = st.checkbox("Resume", value=es.resume)
        self.form_continue = st.checkbox("Continue with errors", value=es.continue_with_errors)
        if st.form_submit_button("Submit"):
            update_from_gui(group=self.form_group,
                            logger=self.form_logger,
                            track_dir=self.form_track_dir,
                            start_grid=self.form_grid,
                            start_run=self.form_run,
                            resume=self.form_resume,
                            continue_errors=self.form_continue)

    def _experiment_board(self, es):
        st.write("## ", f"{es.name} - {es.group}")
        self.yaml_error = st.empty()
        with st.expander("Grid file"):
            st.session_state.edit_mode = st.button("Edit") ^ \
                                             ("edit_mode" in st.session_state and st.session_state.edit_mode)
            if st.session_state.edit_mode:
                st.session_state.settings_string = st_ace(
                    value=st.session_state.settings_string, language="yaml", theme="twilight")
                try:
                    edit_file()
                except ScannerError as e:
                    self.yaml_error.exception(e)
            else:
                st.code(st.session_state.settings_string, language="yaml")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(st.session_state.mk_summary, unsafe_allow_html=True)
        with col2:
            st.write(st.session_state.mk_params)
        if st.button("Launch!"):
            self.experiment()


    def manipulation_interface(self):
        path = st.text_input("wandb experiment path")
        col1, col2 = st.columns(2)
        with col1:
            st.text("Filters")
            filters = st_ace(language="yaml", theme="twilight", key="filters")
        with col2:
            st.text("Updated config")
            updated_config = st_ace(language="yaml", theme="twilight", key="updated_config")
        col1, col2 = st.columns(2)
        with col1:
            st.text("Updated meta config")
            updated_meta = st_ace(language="yaml", theme="twilight", key="updated_meta")
        with col2:
            st.text("What to delete")
            to_delete_place = "keys:\n" \
                              "files:\n" \
                              "artifacts:\n"
            to_delete = st_ace(value=to_delete_place, language="yaml", theme="twilight", key="to_delete")

        fix_string_params = st.checkbox("Fix string parameters turning them into numbers")

        col1, col2 = st.columns(2)
        with col1:
            launch = st.button("Go")
        with col2:
            preview = st.button("Retrieve runs and preview")

        manip = ManipulationInputs()
        manip.update(
            path, filters, updated_config, updated_meta, to_delete, fix_string_params
        )

        if "manip" not in st.session_state:
            st.session_state.manip = manip

        if launch:
            self.manipulate()

        if preview:
            self.preview_manipulation()

    def manipulate(self):
        mp = st.session_state.manip
        api = wandb.Api()
        runs = api.runs(path=mp.path, filters=mp.filters)
        if len(runs) != 0:
            for run in runs:
                print(f"Name: {run.name} Id: {run.id}")
                if mp.fix_string_params:
                    fix_string_param(run)
                if mp.updated_config is None:
                    print('No config to update')
                else:
                    update_config(run, mp.updated_config)
                if mp.updated_metadata is None:
                    print('No metadata to update')
                else:
                    update_metadata(run, mp.updated_metadata)
                if mp.keys_to_delete is None:
                    print('No keys to delete')
                else:
                    remove_key(run, mp.keys_to_delete)
                if mp.files_to_delete is None:
                    print("No files to delete")
                else:
                    delete_files(run, mp.files_to_delete)
                if mp.artifacts_to_delete is None:
                    print("No artifacts to delete")
                else:
                    delete_artifacts(run, mp.artifacts_to_delete)

    def preview_manipulation(self):
        mp = st.session_state.manip
        with open("log.txt") as log:
            log.write(f"Filters: {mp.filters} path: {mp.path}")
        api = wandb.Api()
        runs = api.runs(path=mp.path, filters=mp.filters)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.table(
                {"Name": list(map(lambda x: x.name, runs)), "Group": list(map(lambda x: x.group, runs)), "Id": list(map(lambda x: x.id, runs))}
            )
        with col2:
            st.markdown(format_to_delete(mp.keys_to_delete, mp.files_to_delete, mp.artifacts_to_delete))

    def update_bars(self, bars, status: Status):
        for i in range(status.n_grids):
            if i < status.grid:
                bars[i].progress(1.0)
            elif i == status.grid:
                if status in [Status.FINISHED, Status.CRASHED]:
                    status.run += 1
                bars[i].progress(status.run / status.grid_len)
            else:
                bars[i].progress(0)
        self.current_params.code(dict_to_yaml(status.params), language="yaml")
        self.mk_progress.text(f"Grid {status.grid} / {status.n_grids - 1} \n"
                              f"Run  {status.run} / {status.grid_len - 1}")
        if status == "crashed":
            self.failures.update(status.grid, status.run, status.exception)
            self.mk_failures.markdown(self.failures.get_text())
        if status == Status.STARTING:
            print("Just started")
        if status.run_name is not None:
            self.mk_run_link.markdown(logger_run_link(status.run_name, status.run_url))

    def experiment(self):
        self.mk_progress = st.text("Starting... wait!")
        bars = [st.progress(0) for i in range(len(st.session_state.experimenter.grids))]
        col1, col2 = st.columns(2)
        with col1:
            self.mk_cur_params = st.markdown("### Current run")
            self.mk_run_link = st.markdown("Waiting to create run on the logger platform ")
            self.current_params = st.empty()
        with col2:
            self.failures = MkFailures()
            self.mk_failures = st.markdown("### Failures")
        for status in st.session_state.experimenter.execute_runs_generator():
            self.update_bars(bars, status)
        st.balloons()


def launch_streamlit(args):
    cli_args = ['--grid', str(args.grid),
                '--run', str(args.run),
                ]
    if args.dir:
        cli_args += ['--dir', args.dir]
    if args.file:
        cli_args += ['--file', args.file]
    if args.resume:
        cli_args += ['--resume']

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'guicli.py')

    _config.set_option("server.headless", True)
    print(cli_args)
    bootstrap.run(filename, '', cli_args, flag_options={})


def streamlit_entry(parameter_file, args, share):
    settings = ExpSettings(**args)
    Interface(parameter_file, settings, share)


def frontend(args):
    launch_streamlit(args)
