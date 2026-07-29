import torch


def get_sm_version() -> int:
    props = torch.cuda.get_device_properties(0)
    return props.major * 10 + props.minor
