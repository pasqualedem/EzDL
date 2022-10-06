import time

import streamlit as st

from ezdl.app.markdown import title_builder, exp_summary_builder, grid_summary_builder
from ezdl.experiment.experiment import ExpSettings, Experimenter

from utils.utilities import load_yaml

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
    print(st.session_state.experimenter)
    st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)
    print(st.session_state.mk_summary)


def set_file(file):
    st.session_state.experimenter = Experimenter()
    st.session_state.experimenter.exp_settings = ExpSettings()
    st.session_state.mk_summary = ""
    if file is None:
        return

    settings = load_yaml(file.name)
    sm, grids, dots = st.session_state.experimenter.calculate_runs(settings)
    st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)
    st.session_state.mk_params = grid_summary_builder(grids, dots)


class Interface:
    def __init__(self, parameter_file, exp_settings: ExpSettings = None, share: bool = True):
        if "experimenter" not in st.session_state:
            st.session_state.experimenter = Experimenter()
            set_file(parameter_file)
            st.session_state.experimenter.exp_settings.update(exp_settings)
            st.session_state.mk_summary = exp_summary_builder(st.session_state.experimenter)

        st.set_page_config(
            layout="centered", page_icon="🖱️", page_title="EzDL"
        )
        st.title("EzDL")
        st.write(
            """Train your model!"""
        )
        with st.sidebar:
            with st.form("exp_form"):
                self.form_group = st.text_input("Experiment group")
                self.form_track_dir = st.text_input("Tracking directory")
                self.form_grid = st.number_input("Start from grid", value=0, min_value=0)
                self.form_run = st.number_input("Start from run", value=0, min_value=0)
                self.form_resume = st.checkbox("Resume")
                self.form_continue = st.checkbox("Continue with errors")
                submitted = st.form_submit_button("Submit", on_click=update_from_gui,
                                                  kwargs=dict(
                                                      group=self.form_group,
                                                      track_dir=self.form_track_dir,
                                                      start_grid=self.form_grid,
                                                      start_run=self.form_run,
                                                      resume=self.form_resume,
                                                      continue_errors=self.form_continue
                                                  ))

            self.parameter_file = st.file_uploader("Parameter file")
        st.write("## ", st.session_state.experimenter.exp_settings.group)
        print(st.session_state.mk_summary)
        st.write(st.session_state.mk_summary)


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