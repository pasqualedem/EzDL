import streamlit as st
import wandb
import os

import requests
import threading

from ruamel.yaml.scanner import ScannerError
from streamlit_ace import st_ace
from streamlit.web import bootstrap
from streamlit import config as _config

from ezdl.app.utils import ManipulationInputs
from ezdl.app.markdown import exp_summary_builder, grid_summary_builder, MkFailures, logger_run_link, format_to_delete
from ezdl.experiment.experiment import ExpSettings, Experimenter, Status, FINISHED, STARTING, CRASHED

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


class Dashboard:
    endpoint = "http://localhost:8502"
    def __init__(self, ):
        self.mk_run_link = None
        self.mk_cur_params = None
        self.failures = None
        self.mk_failures = None
        self.current_params = None

        st.set_page_config(
            layout="wide", page_icon="🖱️", page_title="EzDL Dashboard"
        )
        st.title("EzDL - Dashboard")

        experiments = self.get_experiments()
        tabs = st.tabs([f"{uid} - {exp.exp_settings.group}" for uid, (exp, status) in experiments.items()])
        for tab, (uid, (es, status)) in zip(tabs, experiments.items()):
            with tab:
                self._experiment_board(es, status)
        
    def get_experiments(self):
        response = requests.get("http://localhost:8502/experiments")
        response = response.json()['experiments']
        return {uid: (Experimenter(**exp), Status(**status)) for uid, (exp, status) in response.items()}

    def _experiment_board(self, experimenter: Experimenter, status: Status):
        st.write("## ", f"{experimenter.exp_settings.name} - {experimenter.exp_settings.group}")
        self.yaml_error = st.empty()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(exp_summary_builder(experimenter), unsafe_allow_html=True)
        with col2:
            st.write(grid_summary_builder(experimenter.grids, experimenter.dot_elements))
        st.write("---")
        bars = [
            st.progress(0) for _ in range(len(experimenter.grids))
        ]
        col1, col2 = st.columns(2)
        with col1:
            self.mk_cur_params = st.markdown("### Current run")
            self.mk_run_link = st.markdown("Waiting to create run on the logger platform ")
            self.current_params = st.empty()
        with col2:
            self.failures = MkFailures()
            self.mk_failures = st.markdown("### Failures")
        self.update_bars(bars, status)

    def update_bars(self, bars, status: Status):
        for i in range(status.n_grids):
            if i < status.grid:
                bars[i].progress(1.0)
            elif i == status.grid:
                if status in [FINISHED, CRASHED]:
                    status.run += 1
                bars[i].progress(status.run / status.grid_len)
            else:
                bars[i].progress(0)
        self.current_params.code(dict_to_yaml(status.params), language="yaml")
        st.text(f"Grid {status.grid} / {status.n_grids - 1} \n"
                              f"Run  {status.run} / {status.grid_len - 1}")
        if status == "crashed":
            self.failures.update(status.grid, status.run, status.exception)
            self.mk_failures.markdown(self.failures.get_text())
        if status == STARTING:
            print("Just started")
        if status.run_name is not None:
            self.mk_run_link.markdown(logger_run_link(status.run_name, status.run_url))
                    
    def start_polling(self, bars):
        if st.session_state.get('polling', False):
            return
        thread = threading.Thread(target=self.poll_endpoint, args=[bars])
        print("Starting thread")
        thread.start()
        print("Started thread")
        st.session_state['polling'] = True
        
    def poll_endpoint(self, bars):
        while True:
            status = requests.get("http://localhost:8502/status").json()
            print(status)
            self.update_bars(bars, status)

def launch_streamlit():

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'dashcli.py')

    _config.set_option("server.headless", True)
    bootstrap.run(filename, '', [], flag_options={})


def streamlit_entry():
    Dashboard()


def dashboard():
    launch_streamlit()
