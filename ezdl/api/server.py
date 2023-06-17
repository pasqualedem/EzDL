
import uvicorn

from typing import Union

from fastapi import FastAPI, BackgroundTasks

from ezdl.experiment.experiment import Experimenter, Status

app = FastAPI()
EXPERIMENTS = {}


@app.post("/add_experiment")
def add_experiment(id: str, experimenter: Experimenter):
    """
    Add an experiment
    """
    EXPERIMENTS[id] = (experimenter, None)
    

@app.post("/update_experiment")
def update_experiment(id: str, status: Status):
    """
    Update an experiment
    """
    EXPERIMENTS[id] = (EXPERIMENTS[id][0], status)


@app.get("/experiments")
def status():
    return {"experiments": EXPERIMENTS}


def server():
    uvicorn.run(app, host="0.0.0.0", port=8502, log_level="debug")