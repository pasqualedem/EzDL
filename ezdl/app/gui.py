import time

import gradio as gr
import numpy as np
import pandas as pd
import torch
import wandb

from ezdl.app.markdown import grid_summary_builder, exp_summary_builder, title_builder, MkFailures
from ezdl.utils.utilities import load_yaml, update_collection
from ezdl.utils.segmentation import tensor_to_segmentation_image
from ezdl.app.inference import WandbInferencer, compose_images

from ezdl.experiment.experiment import Experimenter, ExpSettings

DEFAULT_CMAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0)
}


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


TITLE = "# EzDL"
TEXT = "Load config and model to select image channels"
DIV = "---"
COLS = ['Grid', 'N. runs']


def segment(pred: torch.Tensor):
    pred = pred.squeeze(0).argmax(dim=0).cpu()
    return tensor_to_segmentation_image(pred, DEFAULT_CMAP)


def flag_maker(exp_setings: ExpSettings):
    flags = []
    if exp_setings.resume:
        flags.append('resume')
    if exp_setings.continue_with_errors:
        flags.append('continue_with_errors')
    return flags


class Interface:
    def __init__(self, parameter_file: str = None, exp_settings: ExpSettings = None, share: bool = False):
        with gr.Blocks() as demo:
            gr.Markdown(TITLE)
            with gr.Tab("Training"):
                TrainingInterface(parameter_file, exp_settings)
            with gr.Tab("Inference"):
                InferenceInterface()

        demo.queue()
        demo.launch(share=share)


class InferenceInterface:
    def __init__(self):
        use_gpu = gr.components.Checkbox(label="Use GPU")
        with gr.Row():
            model = gr.components.File(type='file', label="Model checkpoint in Wandb folder")
            config = gr.components.File(type='file', label="Wandb config input")
        text = gr.Textbox(TEXT, show_label=False)
        divider = gr.Markdown(DIV)
        with gr.Row() as self.input_images_row:
            self.input_channels = {
                'R': gr.Image(type='pil', image_mode='L', label='R Channel', visible=False),
                'G': gr.Image(type='pil', image_mode='L', label='G Channel', visible=False),
                'B': gr.Image(type='pil', image_mode='L', label='B Channel', visible=False),
                "NDVI": gr.Image(type='pil', image_mode='L', label='NDVI Channel', visible=False),
                "NIR": gr.Image(type='pil', image_mode='L', label='NIR Channel', visible=False),
            }
        config.change(self.set_model,
                      inputs=[model, config, use_gpu],
                      outputs=[*self.input_channels.values(), text],
                      )
        model.change(self.set_model,
                     inputs=[model, config, use_gpu],
                     outputs=[*self.input_channels.values(), text],
                     )
        submit = gr.Button(variant="primary")
        segmentation = gr.Image(label="Segmentation")

        submit.click(self.predict, inputs=[use_gpu, *self.input_channels.values()], outputs=[segmentation])

    def set_model(self, model_path_wrapper, config_wrapper, gpu):
        if model_path_wrapper is None or config_wrapper is None:
            return [gr.update(visible=False) for _ in self.input_channels] + [TEXT]
        self.inferencer = WandbInferencer(model_path_wrapper, config_wrapper, gpu)
        input_channels = []
        for channel, input_form in self.input_channels.items():
            if channel in self.inferencer.channels:
                input_channels.append(gr.update(visible=True))
            else:
                input_channels.append(gr.update(visible=False))
        return input_channels + ["".join([f"{channel} " for channel in self.inferencer.channels]) + "needed"]

    def predict(self, use_gpu, *inputs):
        if use_gpu:
            self.inferencer.cuda()
        else:
            self.inferencer.cpu()
        inputs = filter(lambda x: x is not None, inputs)
        image = compose_images(list(inputs))
        pred = self.inferencer(image)
        segmentation = segment(pred)
        return segmentation


class TrainingInterface:

    def __init__(self, parameter_file: str = None, exp_settings: ExpSettings = None):
        self.experimenter = Experimenter()
        default_updates = self.set_file(parameter_file)
        self.experimenter.exp_settings.update(exp_settings)
        exp_settings = self.experimenter.exp_settings
        gr.Markdown("Train your model!")
        with gr.Row():
            self.group_txt = gr.Textbox(label="Experiment group", value=exp_settings.group)
            self.track_dir = gr.Textbox(label="Tracking directory", value=exp_settings.tracking_dir)
            self.start_from_grid_txt = gr.Number(label="Start from grid",
                                                 precision=0,
                                                 value=exp_settings.start_from_grid)
            self.start_from_run_txt = gr.Number(label="Start from run",
                                                precision=0,
                                                value=exp_settings.start_from_run)

            self.flags = gr.CheckboxGroup(["resume", "continue_with_errors"], label="", value=flag_maker(exp_settings))

        self.parameters = gr.File(label="Parameter file", value=parameter_file)
        self.group_mk = gr.Markdown(exp_settings.group)
        with gr.Row():
            self.exp_summary = gr.Markdown(value=default_updates["exp_summary"])
            self.grids_dataframe = gr.Markdown(value=default_updates["grids_dataframe"])

        parameters_comps = [self.group_txt,
                            self.track_dir, self.start_from_grid_txt,
                            self.start_from_run_txt, self.flags]
        for com in parameters_comps:
            com.change(self.update_from_gui, inputs=parameters_comps, outputs=[self.group_mk, self.exp_summary])
        self.parameters.change(self.update_from_file, inputs=[self.parameters],
                               outputs=[self.group_mk, self.exp_summary, self.grids_dataframe] + parameters_comps)

        self.experiment_btn = gr.Button("Experiment!", variant="primary")
        self.progress = gr.Label(value={}, label="Progress")
        with gr.Row():
            self.json = gr.JSON(label="Current run params")
            self.failures = MkFailures()
            self.failures_mk = gr.Markdown(self.failures.get_text())
        self.experiment_btn.click(self.experiment, inputs=[], outputs=[self.progress, self.json, self.failures_mk])

    def experiment(self):
        yield self.update_progress(self.experimenter.exp_settings.start_from_grid,
                                   self.experimenter.exp_settings.start_from_run - 1,
                                   len(self.experimenter.grids),
                                   len(self.experimenter.grids[self.experimenter.exp_settings.start_from_grid]),
                                   "starting",
                                   {})
        for out in self.experimenter.execute_runs_generator(callback=self.update_progress):
            yield out

    def update_progress(self, cur_grid, cur_run, n_grids, n_runs, status, run_params, exception=None):
        d = {}
        for i in range(n_grids):
            if i < cur_grid:
                d[f"Grid {i}"] = 1
            elif i == cur_grid:
                if status in ["finished", "crashed"]:
                    cur_run += 1
                d[f"Grid {i} ({cur_run} / {n_runs})"] = cur_run / n_runs
            else:
                d[f"Grid {i}"] = 0
        if status == "crashed":
            self.failures.update(cur_grid, cur_run, exception)
        return d, run_params, self.failures.get_text()

    def set_file(self, file):
        self.experimenter = Experimenter()
        self.experimenter.exp_settings = ExpSettings()
        if file is None:
            return DEFAULTS([
                "group",
                "track_dir",
                "start_from_grid",
                "start_from_run",
                "flags",
                "group_mk",
                "grids_dataframe",
                "exp_summary",
            ])

        settings = load_yaml(file.name)
        sm, grids, dots = self.experimenter.calculate_runs(settings)
        sum_mk = exp_summary_builder(self.experimenter)
        params_mk = grid_summary_builder(grids, dots)
        flags = flag_maker(self.experimenter.exp_settings)
        return {
                "group": self.experimenter.exp_settings['group'],
                "track_dir": self.experimenter.exp_settings['tracking_dir'] or DEFAULTS('tracking_dir'),
                "start_from_grid": self.experimenter.exp_settings['start_from_grid'],
                "start_from_run": self.experimenter.exp_settings['start_from_run'],
                "flags": flags,
                "group_mk": title_builder(self.experimenter.exp_settings['group']),
                "exp_summary": sum_mk,
                "grids_dataframe": params_mk
            }

    def update_from_file(self, file):
        updates = self.set_file(file)
        return {
                self.group_txt: self.experimenter.exp_settings['group'],
                self.track_dir: self.experimenter.exp_settings['tracking_dir'] or DEFAULTS('tracking_dir'),
                self.start_from_grid_txt: self.experimenter.exp_settings['start_from_grid'],
                self.start_from_run_txt: self.experimenter.exp_settings['start_from_run'],
                self.flags: updates["flags"],
                self.group_mk: title_builder(self.experimenter.exp_settings['group']),
                self.exp_summary: updates["exp_summary"],
                self.grids_dataframe: updates["grids_dataframe"]
            }

    def update_from_gui(self, name, track_dir, start_grid, start_run, flags):
        if self.experimenter is None:
            self.experimenter = Experimenter()
        self.experimenter.update_settings({
            "resume": ("resume" in flags),
            "continue_with_errors": ("continue_with_errors" in flags),
            "start_from_grid": start_grid,
            "start_from_run": start_run,
            "tracking_dir": track_dir,
            "group": name,
        })
        return title_builder(self.experimenter.exp_settings['group']) or "", exp_summary_builder(self.experimenter)


def frontend(parameter_file, args, share):
    settings = ExpSettings(**args)
    Interface(parameter_file, settings, share)
