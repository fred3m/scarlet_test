import os
import errno
from typing import List
import shutil

import numpy as np
import matplotlib.pyplot as plt

from . import deblend
from . import settings


# Paths to directories for different file types
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
__DATA_PATH__ = os.path.join(__ROOT__, "data")
__BLEND_PATH__ = os.path.join(__DATA_PATH__, "blends")
__DOCS_PATH__ = os.path.join(__ROOT__, "docs")
__PLOT_PATH__ = os.path.join(__DOCS_PATH__, "plots")
__SCENE_PATH__ = os.path.join(__PLOT_PATH__)


def get_blend_ids(set_id: str) -> List[str]:
    """
    Get all of the blend IDs contained in the set
    :param set_id: The set ID to search
    :return: List of blend IDs
    """
    id_data = np.load(os.path.join(__BLEND_PATH__, "test_ids.npz"))
    return id_data[set_id]


def get_prs(path: str):
    # Make sure that the set id is valid
    return [f.split(".")[0] for f in os.listdir(path)]


def check_data_existence(path: str, set_id: str, pr: str, overwrite: bool) -> bool:
    """
    Check if the pr has already been processed and if it's ok to overwrite
    :param path: The path to check for the data.
    :param set_id: ID of the set to analyze
    :param pr: The scarlet PR number
    :param overwrite: Whether or not it is ok to rewrite the existing PR
    :return: `True` if the pr can be written
    """
    prs = get_prs(path)
    if pr in prs and not overwrite:
        msg = "PR {} has already been analyzed for set {}, to overwrite set the `overwrite` flag"
        raise ValueError(msg.format(pr, set_id))
    return True


def create_path(path: str) -> None:
    """
    Create a path if it doesn't already exist
    :param path: The name of the path to check/create
    """
    try:
        os.makedirs(path)
        print("created path '{}'".format(path))
    except OSError as e:
        print("results will be written to {}".format(path))
        if e.errno != errno.EEXIST:
            raise e


def get_filename(pr: str) -> str:
    """
    Consistently set the filename for each PR, for each set

    :param pr: The scarlet PR number
    :return: The filename to store the measurements for each source
    """
    return "{}.npz".format(pr)


def deblend_and_measure(set_id: str, pr: str, overwrite: bool = False, save_path: str = None) -> None:
    """
    Deblend an entire test set and store the measurements

    :param set_id: ID of the set to analyze
    :param pr: The scarlet PR number
    :param overwrite: Whether or not it is ok to rewrite the existing PR
    :param save_path: Path to save the data to. This is usually `__DATA_PATH__`.
    :return: The measurement `records` for each blend.
    """
    if save_path is not None:
        create_path(save_path)
        check_data_existence(save_path, set_id, pr, overwrite)

    # Load the blend data
    blend_ids = get_blend_ids(set_id)

    # Deblend the scene
    results = []
    for blend_id in blend_ids:
        print(blend_id)
        result, _, _, _ = deblend.deblend(
            blend_id,
            settings.max_iter,
            settings.e_rel,
            __BLEND_PATH__,
            settings.filters,
        )
        results += result

    # Combine all of the records together and save
    records = [tuple(m.values()) for m in results]
    keys = tuple(results[0].keys())
    records = np.rec.fromrecords(records, names=keys)
    # Save the data if a path was provided
    if save_path is not None:
        np.savez(os.path.join(save_path, get_filename(pr)), records=records)
    return records


def deblend_residuals(set_id: str, pr: str, overwrite: bool = False, save_path: str = None) -> None:
    """
    Deblend an entire test set and store the residual plot

    :param set_id: ID of the set to analyze
    :param pr: The scarlet PR number
    :param overwrite: Whether or not it is ok to rewrite the existing PR
    :param save_path: Path to save the data to. This is usually `__SCENE_PATH__`.
    """
    from scarlet import display

    if save_path is not None:
        create_path(save_path)
        check_data_existence(save_path, set_id, pr, overwrite)

    # Load the blend data
    blend_ids = get_blend_ids(set_id)
    for blend_id in blend_ids:
        print(blend_id)
        results = deblend.deblend(
            blend_id,
            settings.max_iter,
            settings.e_rel,
            __BLEND_PATH__,
            settings.filters,
        )
        measurements, images, observation, sources = results

        filename = os.path.join(save_path, "{}.scene.png".format(blend_id))
        if os.path.exists(filename):
            # Copy the current version as the old filename
            new_filename = os.path.join(save_path, "old_{}.scene.png".format(blend_id))
            shutil.move(filename, new_filename)

        norm = display.AsinhMapping(minimum=np.min(images), stretch=np.max(images) * 0.055, Q=10)
        display.show_scene(sources, observation, show_model=False, show_observed=True, show_rendered=True,
                           show_residual=True, norm=norm, figsize=(15, 5))
        plt.suptitle(pr, y=1.05)
        if save_path is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
