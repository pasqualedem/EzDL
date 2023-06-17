import requests
import uuid
import time

from typing import Mapping

from ezdl.experiment.experiment import Experimenter, Status
from ezdl.logger.text_logger import get_logger

logger = get_logger(__name__)


def logged_experiment(settings: Mapping, param_path: str = "local variable"):
    URL = "http://localhost:8502"
    ADD_API = "add_experiment"
    UPDATE_API = "update_experiment"
    logger.info(f'Loaded parameters from {param_path}')
    headers = {
    "Content-Type": "application/json"
    }
    params = {
        "id": str(uuid.uuid4()[:8]),
    }

    experimenter = Experimenter()
    grid_summary, grids, cartesian_elements = experimenter.calculate_runs(settings)

    response = requests.post(f"{URL}/{ADD_API}", json=experimenter.dict(), headers=headers, params=params)
    print(response.json())
    for i in range(len(cartesian_elements)):
        print(f"Run {i}")
        status = Status(
            grid=0,
            run=i,
            params={"bello": "ciao"},
            n_grids=1,
            grid_len=32,
            run_name=f"popo{str(i)}",
            run_url="preppoen",
        )
        time.sleep(5)
        response = requests.post(f"{URL}/{UPDATE_API}", json=status.dict(), headers=headers, params=params)
        print(response.json())