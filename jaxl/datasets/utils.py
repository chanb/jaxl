from typing import Any, Tuple

import _pickle as pickle
import os


def maybe_save_dataset(data: Any, save_dir: str, dataset_name: str):
    if save_dir is not None:
        full_path = os.path.join(save_dir, dataset_name)
        if not os.path.isfile(full_path):
            print("Saving to {}".format(full_path))
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                data,
                open(full_path, "wb"),
            )


def maybe_load_dataset(save_dir: str, dataset_name: str) -> Tuple[bool, Any]:
    if save_dir is not None:
        full_path = os.path.join(save_dir, dataset_name)
        if os.path.isfile(full_path):
            print("Loading from {}".format(full_path))
            return True, pickle.load(
                open(full_path, "rb")
            )
        print("{} not found".format(full_path))
    return False, None
