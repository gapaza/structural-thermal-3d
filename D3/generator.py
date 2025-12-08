"""This module generates a dataset for the thermoelastic3d problem."""
import os
from pathlib import Path
import pickle
import random

from datasets import Dataset
from datasets import DatasetDict
import numpy as np

from engibench.problems.thermoelastic3d.v0 import ThermoElastic3D

SEED=42
random.seed(SEED)

def get_rnd_heatsink(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Randomly places a heatsink on a node in the domain.

    Args:
        nelx: number of x elements
        nely: number of y elements
        nelz: number of z elements

    Returns:
        elements: a 3D numpy mask encoding heatsink nodes
    """
    elements = np.zeros((nelx+1, nely+1, nelz+1), dtype=int)
    rand_x = random.randint(0, nelx)
    rand_y = random.randint(0, nely)
    elements[rand_x, rand_y, 0] = 1
    return elements

def get_rnd_fixed(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Randomly places fixed supports on a node in the domain.

    Args:
        nelx: number of x elements
        nely: number of y elements
        nelz: number of z elements

    Returns:
        elements: a 3D numpy mask encoding fixed support nodes
    """
    elements = np.zeros((nelx+1, nely+1, nelz+1), dtype=int)
    rand_y1 = random.randint(0, nely)
    rand_y2 = random.randint(0, nely)
    rand_y3 = random.randint(0, nely)
    rand_z1 = random.randint(0, nelz)
    rand_z2 = random.randint(0, nelz)
    rand_z3 = random.randint(0, nelz)
    elements[0, rand_y1, rand_z1] = 1
    elements[0, rand_y2, rand_z2] = 1
    elements[0, rand_y3, rand_z3] = 1
    return elements

def get_rnd_force(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Randomly places forces on a node in the domain.

    Args:
        nelx: number of x elements
        nely: number of y elements
        nelz: number of z elements

    Returns:
        elements: a 3D numpy mask encoding forces nodes
    """
    elements = np.zeros((nelx+1, nely+1, nelz+1), dtype=int)
    rand_y = random.randint(0, nely)
    rand_z = random.randint(0, nelz)
    elements[-1, rand_y, rand_z] = 1
    elements_x = elements_y = elements_z = elements
    return elements_x, elements_y, elements_z

def get_rnd_volfrac() -> float:
    """Randomly samples a volume fraction.

    Returns:
        options: a volume fraction
    """
    options = [0.25, 0.30, 0.35, 0.40, 0.45]
    return random.choice(options)


def gen_dataset(save_path: str) -> None:
    """Generate a dataset of optimized designs for random boundary conditions.

    Args:
        save_path: The path to save the resultant pickle files

    Returns:
        None
    """
    nelx = nely = nelz = 16
    n_datapoints = 100

    conditions = []
    for _ in range(n_datapoints):
        fixed_elements = get_rnd_fixed(nelx, nely, nelz)
        force_elements_x, force_elements_y, force_elements_z = get_rnd_force(nelx, nely, nelz)
        heatsink_elements = get_rnd_heatsink(nelx, nely, nelz)
        volfrac = get_rnd_volfrac()
        condition = {
            "fixed_elements": fixed_elements,
            "force_elements_x": force_elements_x,
            "force_elements_y": force_elements_y,
            "force_elements_z": force_elements_z,
            "heatsink_elements": heatsink_elements,
            "volfrac": volfrac,
            "rmin": 1.5,
            "weight": 0.5,
            "nelx": nelx,
            "nely": nely,
            "nelz": nelz,
        }
        conditions.append(condition)

    for idx, cond in enumerate(conditions):
        starting_point = cond["volfrac"] * np.ones((nelx, nely, nelz))
        problem = ThermoElastic3D(seed=0)
        design, objectives = problem.optimize(starting_point, cond)
        final_objectives = objectives[-1].obj_values
        structural_compliance = final_objectives[0]
        thermal_compliance = final_objectives[1]
        volume_fraction_error = final_objectives[2]

        cond["optimal_design"] = design
        cond["volume_fraction"] = volume_fraction_error
        cond["structural_compliance"] = structural_compliance
        cond["thermal_compliance"] = thermal_compliance

        f_path = os.path.join(save_path, f"{idx}.pkl")
        with open(f_path, "wb") as f:
            pickle.dump(cond, f)


def _convert_entry_to_hf_friendly(entry: dict) -> dict:
    """Converts a datapoint to the appropriate format for huggingface saving.

    Args:
        entry: a dictionary holding a single datapoint

    Returns:
        out: a formatted datapoint
    """
    out = {}
    for k, v in entry.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

def push_dataset(save_path: str) -> None:
    """Pushes the generated dataset to huggingface.

    Args:
        save_path: The path to load the pickle files containing the datapoints.

    Returns:
        None
    """
    directory = Path(save_path)
    files = sorted(
        [f for f in directory.iterdir() if f.suffix in {".pkl", ".pickle"}],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    datapoints = []
    for file in files:
        with open(file, "rb") as f:
            dp = pickle.load(f)
            dp_volfrac = np.mean(dp["optimal_design"])
            dp["volume_fraction"] = dp_volfrac

            datapoints.append(dp)
    datapoints = [_convert_entry_to_hf_friendly(dp) for dp in datapoints]

    random.seed(0)
    random.shuffle(datapoints)

    n = len(datapoints)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_data = datapoints[:n_train]
    val_data = datapoints[n_train:n_train + n_val]
    test_data = datapoints[n_train + n_val:]

    ds_train = Dataset.from_list(train_data)
    ds_val = Dataset.from_list(val_data)
    ds_test = Dataset.from_list(test_data)

    ds_dict = DatasetDict(
        {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
        }
    )

    hf_dataset_id = "IDEALLab/thermoelastic_3d_v0"

    # This uses your existing HF auth (e.g. `huggingface-cli login`)
    ds_dict.push_to_hub(hf_dataset_id)










if __name__ == "__main__":
    save_path = "/Users/gapaza/repos/ideal/EngiBench/engibench/problems/thermoelastic3d/designs"
    gen_dataset(save_path)
    push_dataset(save_path)
