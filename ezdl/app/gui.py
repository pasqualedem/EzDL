import time

import streamlit as st

from ezdl.app.markdown import title_builder, exp_summary_builder, grid_summary_builder, MkFailures
from ezdl.experiment.experiment import ExpSettings, Experimenter

from ezdl.utils.utilities import load_yaml

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
    file = st.session_state.parameter_file
    if file is None:
        return
    settings = load_yaml(file)
    sm, grids, dots = st.session_state.experimenter.calculate_runs(settings)
    st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)
    st.session_state.mk_params = grid_summary_builder(grids, dots)


class Interface:
    def __init__(self, parameter_file, exp_settings: ExpSettings = None, share: bool = True):
        self.mk_cur_params = None
        self.failures = None
        self.mk_failures = None
        self.json = None
        if "experimenter" not in st.session_state:
            st.session_state.experimenter = Experimenter()
            st.session_state.experimenter.exp_settings.update(exp_settings)
            st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)

        st.set_page_config(
            layout="centered", page_icon="🖱️", page_title="EzDL"
        )
        st.title("EzDL")
        st.write(
            """Train your model!"""
        )
        es = st.session_state.experimenter.exp_settings
        with st.sidebar:
            with st.form("exp_form"):
                self.form_group = st.text_input("Experiment group", value=es.group)
                self.form_track_dir = st.text_input("Tracking directory", value=es.tracking_dir)
                self.form_grid = st.number_input("Start from grid", min_value=0, value=es.start_from_grid)
                self.form_run = st.number_input("Start from run", min_value=0, value=es.start_from_run)
                self.form_resume = st.checkbox("Resume", value=es.resume)
                self.form_continue = st.checkbox("Continue with errors", value=es.continue_with_errors)
                submitted = st.form_submit_button("Submit", on_click=update_from_gui,
                                                  kwargs=dict(
                                                      group=self.form_group,
                                                      track_dir=self.form_track_dir,
                                                      start_grid=self.form_grid,
                                                      start_run=self.form_run,
                                                      resume=self.form_resume,
                                                      continue_errors=self.form_continue
                                                  ))

            self.parameter_file = st.file_uploader("Parameter file", on_change=set_file, key="parameter_file")
        if submitted or st.session_state.parameter_file:
            st.write("## ", st.session_state.experimenter.exp_settings.group)
            col1, col2 = st.columns(2)
            with col1:
                st.write(st.session_state.mk_summary)
            with col2:
                st.write(st.session_state.mk_params)
            launch = st.button("Launch!")
            if launch:
                self.experiment()

    def update_bars(self, bars, n_grids, n_runs, status, cur_run, cur_grid, exception):
        for i in range(n_grids):
            if i < cur_grid:
                bars[i].progress(1.0)
            elif i == cur_grid:
                if status in ["finished", "crashed"]:
                    cur_run += 1
                bars[i].progress(cur_run / n_runs)
            else:
                bars[i].progress(0)
        self.json.json({1:2, 3:4})
        self.mk_progress.text(f"Grid {cur_grid} / {n_grids - 1} \n"
                              f"Run  {cur_run - 1} / {n_runs - 1}")
        if status == "crashed":
            self.failures.update(cur_grid, cur_run, exception)
            self.mk_failures.markdown(self.failures.get_text())

    def experiment(self):
        self.mk_progress = st.text("Starting... wait!")
        bars = [st.progress(0) for i in range(len(st.session_state.experimenter.grids))]
        # bars = [st.progress(0), st.progress(0), st.progress(0)]
        col1, col2 = st.columns(2)
        with col1:
            self.mk_cur_params = st.markdown("### Current parameters")
            self.json = st.empty()
        with col2:
            self.failures = MkFailures()
            self.mk_failures = st.markdown("### Failures")
        for out in st.session_state.experimenter.execute_runs_generator():
        # for out in placeholder():
            cur_grid, cur_run, n_grids, n_runs, status, run_params, exception = out
            self.update_bars(bars, n_grids, n_runs, status, cur_run, cur_grid, exception)
        st.balloons()


def placeholder():
    grids = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
    n_runs = [3, 2, 2]
    for grid, run in grids:
        time.sleep(1)
        yield grid, run, 3, n_runs[grid], "finished", {}, None


def frontend(parameter_file, args, share):
    settings = ExpSettings(**args)
    Interface(parameter_file, settings, share)
    # import datetime
    # st.title('Counter Example')
    # if 'count' not in st.session_state:
    #     st.session_state.count = 0
    #     st.session_state.last_updated = datetime.time(0, 0)
    #
    # def update_counter():
    #     st.session_state.count += st.session_state.increment_value
    #     st.session_state.last_updated = st.session_state.update_time
    #
    # with st.form(key='my_form'):
    #     st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    #     st.number_input('Enter a value', value=0, step=1, key='increment_value')
    #     submit = st.form_submit_button(label='Update', on_click=update_counter)
    #
    # st.write('Current Count = ', st.session_state.count)
    # st.write('Last Updated = ', st.session_state.last_updated)
