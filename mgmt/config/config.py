from typing import List

from fvcore.common.config import CfgNode


def get_cfg(config_files: None | List[str] | str = None) -> CfgNode:
    """
    Get a copy of the config. Load the default configs and overwrites the
    values with the provided config_files

    Args:
        config_files: filepaths to YAML config files. The order of the config
        files matters. Subsequent configs overwrite the previous.
    """
    if isinstance(config_files, str):
        config_files = [config_files]
    if config_files is None:
        file_cfg = None
    else:
        file_cfg = CfgNode(new_allowed=True)
        for c in config_files:
            print(c)
            file_cfg.merge_from_file(c, allow_unsafe=True)
            file_cfg.set_new_allowed(True)

    from .defaults import _C

    C = CfgNode(new_allowed=True)
    C.merge_from_other_cfg(_C)
    C.set_new_allowed(True)

    if file_cfg is not None:
        C.merge_from_other_cfg(file_cfg)
    C = cfg_set_none(C)
    return C


def cfg_set_none(cfg: CfgNode) -> CfgNode:
    for key, value in cfg.items():
        if value == "None":
            cfg[key] = None
        elif isinstance(value, CfgNode):
            cfg[key] = cfg_set_none(value)
    return cfg
