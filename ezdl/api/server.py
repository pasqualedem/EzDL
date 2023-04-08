
import uvicorn

from typing import Union

from fastapi import FastAPI, BackgroundTasks

from ezdl.experiment.experiment import Experimenter, Status

app = FastAPI()
EXPERIMENTER = None
STATUS = None


def background_experiment():
    global STATUS
    for status in EXPERIMENTER.execute_runs_generator():
        STATUS = status

@app.post("/experiment")
async def experiment(experimenter: Experimenter, background_tasks: BackgroundTasks):
    """
    Run an experiment
    """
    global EXPERIMENTER
    global STATUS
    EXPERIMENTER = experimenter
    background_tasks.add_task(background_experiment)
    return {"running": True}


@app.get("/status")
def status() -> Status:
    print(STATUS)
    return {"status": STATUS}


@app.get("/ok")
def status() -> Status:
    return {"ok": 0}


def server():
    uvicorn.run(app, host="0.0.0.0", port=8502)