import sys

import streamlit as st
from ruamel.yaml.scanner import ScannerError
from streamlit.web import cli as stcli
from streamlit_ace import st_ace

from ezdl.app.markdown import exp_summary_builder, grid_summary_builder, MkFailures, wandb_run_link
from ezdl.experiment.experiment import ExpSettings, Experimenter, Status

from ezdl.utils.utilities import load_yaml, dict_to_yaml, yaml_string_to_dict

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


def update_from_gui(group, track_dir, start_grid, start_run, resume, continue_errors):
    if st.session_state.experimenter is None:
        st.session_state.experimenter = Experimenter()
    st.session_state.experimenter.update_settings({
        "resume": resume,
        "continue_with_errors": continue_errors,
        "start_from_grid": start_grid,
        "start_from_run": start_run,
        "tracking_dir": track_dir,
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
        self.mk_wandb_link = None
        self.mk_cur_params = None
        self.failures = None
        self.mk_failures = None
        self.current_params = None
        if "experimenter" not in st.session_state:
            experimenter = Experimenter()
            st.session_state.experimenter = experimenter
            experimenter.exp_settings.update(exp_settings)
            st.session_state.mk_summary = exp_summary_builder(experimenter)
        else:
            experimenter = st.session_state.experimenter

        st.set_page_config(
            layout="wide", page_icon="🖱️", page_title="EzDL"
        )
        st.title("EzDL")
        st.write(
            """Train your model!"""
        )
        es = experimenter.exp_settings
        with st.sidebar:
            with st.form("exp_form"):
                self.form_path = st.text_input("Experiment name (path)", value=es.name)
                self.form_group = st.text_input("Experiment group", value=es.group)
                self.form_track_dir = st.text_input("Tracking directory", value=es.tracking_dir)
                self.form_grid = st.number_input("Start from grid", min_value=0, value=es.start_from_grid)
                self.form_run = st.number_input("Start from run", min_value=0, value=es.start_from_run)
                self.form_resume = st.checkbox("Resume", value=es.resume)
                self.form_continue = st.checkbox("Continue with errors", value=es.continue_with_errors)
                submitted = st.form_submit_button("Submit")
                if submitted:
                    update_from_gui(group=self.form_group,
                                    track_dir=self.form_track_dir,
                                    start_grid=self.form_grid,
                                    start_run=self.form_run,
                                    resume=self.form_resume,
                                    continue_errors=self.form_continue)

            self.parameter_file = st.file_uploader("Parameter file", on_change=set_file, key="parameter_file")
        if "parameter_file" in st.session_state and st.session_state.parameter_file:
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
            launch = st.button("Launch!")
            if launch:
                self.experiment()

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
        if status.wandb_run is not None:
            self.mk_wandb_link.markdown(wandb_run_link(status.wandb_run))

    def experiment(self):
        self.mk_progress = st.text("Starting... wait!")
        bars = [st.progress(0) for i in range(len(st.session_state.experimenter.grids))]
        col1, col2 = st.columns(2)
        with col1:
            self.mk_cur_params = st.markdown("### Current run")
            self.mk_wandb_link = st.markdown("Waiting to create run on Wandb")
            self.current_params = st.empty()
        with col2:
            self.failures = MkFailures()
            self.mk_failures = st.markdown("### Failures")
        for status in st.session_state.experimenter.execute_runs_generator():
            self.update_bars(bars, status)
        st.balloons()


def launch_streamlit(args):
    sys.argv = ["streamlit", "run", "./ezdl/app/guicli.py"]
    # cli_args = f'-- ' \
    #            f'--f {args.file} ' \
    #            f'--grid {args.grid} ' \
    #            f'--run {args.run} ' \
    #            f'--resume {args.resume} ' \
    #            f'--dir {args.dir} ' \
    #            f'--share {args.share}'
    cli_args = ['--',
                '--grid', str(args.grid),
                '--run', str(args.run),
                ]
    if args.dir:
        cli_args += ['--dir', args.dir]
    if args.file:
        cli_args += ['--file', args.file]
    if args.resume:
        cli_args += ['--resume']
    if args.share:
        cli_args += ['--share']

    sys.argv += cli_args
    sys.exit(stcli.main())


def streamlit_entry(parameter_file, args, share):
    settings = ExpSettings(**args)
    Interface(parameter_file, settings, share)


def frontend(args):
    launch_streamlit(args)
