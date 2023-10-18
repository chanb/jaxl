import os

CONST_CPU = "cpu"
CONST_GPU = "gpu"


def get_device(device):
    (device_name, *device_ids) = device.split(":")
    if device_name == CONST_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_name == CONST_GPU:
        assert (
            len(device_ids) > 0
        ), f"at least one device_id is needed, got {device_ids}"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
    else:
        raise ValueError(f"{device_name} is not a supported device.")
