import os
import pathlib
import logging as log
from typing import Any, Dict, Union


def pipeline_operator(func):
    """
    Use to decorate a function to be used as a pipeline operator.
    """
    base_path = "/tmp/outputs"

    def _dump_dict(r: Dict[str, Any]) -> None:
        for k, v in r.items():
            var_path = os.path.join(base_path, k)
            pathlib.Path(var_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(var_path, "data"), "w") as f:
                f.write(str(v))

    def _dump_result(r: Any):
        if type(r) in set([int, str, float]):
            _dump_dict({"Output": r})
        elif type(r) == dict:
            _dump_dict(r)
        elif hasattr(r, "__dict__"):
            _dump_dict(r.__dict__)
        else:
            raise Exception("unsupported return type")

    def wrapper(*argv, **kwargs):
        r = func(*argv, **kwargs)
        log.info(f"saving function output to {base_path}")
        _dump_result(r)
        return r

    return wrapper
