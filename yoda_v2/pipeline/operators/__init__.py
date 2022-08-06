import copy
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
)

import kfp.dsl as dsl
from kfp.components import create_component_from_func, func_to_container_op
from kfp.dsl._container_op import ArgumentOrArguments, StringOrStringList
from kubernetes.client.models import V1EnvVar


def _get_wrapped_function(f: Callable) -> Callable:
    if hasattr(f, "__closure__") and f.__closure__ is not None:
        r = list(
            filter(
                lambda x: hasattr(x, "__name__") and x.__name__ != "_dump_result",
                [c.cell_contents for c in f.__closure__],
            )
        )
        if len(r) != 1:
            raise Exception(
                "Ambiguous function. Most likely function has multiple decorators."
            )
        return r[0]
    else:
        return f


def _is_simple_type(t: type) -> bool:
    return t in set([int, float, str, bool])


def _create_args(f: Callable) -> List[str]:
    annotations = copy.deepcopy(_get_wrapped_function(f).__annotations__)
    if "return" in annotations:
        annotations.pop("return")
    args = []
    for k, v in annotations.items():
        if _is_simple_type(v):
            args.append(f"--{k} " + "{" + f"{k}" + "} ")
        else:
            raise Exception(f"unsupported input type: {k}: [{v}]")
    return args


def _create_outputs(f: Callable) -> Dict[str, str]:
    annotations = copy.deepcopy(_get_wrapped_function(f).__annotations__)
    if "return" not in annotations:
        return []
    return_type = annotations.pop("return")
    if _is_simple_type(return_type):
        return {"Output": "/tmp/outputs/Output/data"}
    if (
        hasattr(return_type, "__dict__")
        and "__dataclass_fields__" in return_type.__dict__
    ):
        fields = return_type.__dict__["__dataclass_fields__"]
        result = {}
        for k, v in fields.items():
            if _is_simple_type(v.type):
                result[k] = f"/tmp/outputs/{k}/data"
            else:
                raise Exception(f"unsupported output type: {k}: [{v.type}]")
        return result


class PythonJobOp(dsl.ContainerOp):
    def __init__(
        self,
        f,
        cmd: StringOrStringList,
        args: ArgumentOrArguments,
        image: str,
        name: str = "python-job-operator",
    ):
        # TODO: check signature
        # f_args = ''.join(create_args(f))
        final_arguments = "python " + " ".join(cmd) if isinstance(cmd, list) else cmd
        final_arguments += " " + " ".join(args) if isinstance(args, list) else args
        super().__init__(
            name=name,
            image=image,
            command=["sh", "-c"],
            arguments=final_arguments,
            file_outputs=_create_outputs(f),
            container_kwargs={
                "env": [
                    V1EnvVar(
                        "MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow:8081"
                    ),
                    V1EnvVar(
                        "MLFLOW_S3_ENDPOINT_URL", "http://minio-service.minio:8081"
                    ),
                    V1EnvVar("AWS_ACCESS_KEY_ID", "minio123"),
                    V1EnvVar("AWS_SECRET_ACCESS_KEY", "minio123"),
                ]
            },
        )


def _publish_model(
    artifact_uri: str, model_name: str, dimension: int = None
) -> NamedTuple("ModelVersion", [("name", str), ("version", int)]):
    """ returns model (name,version) """
    import os
    from collections import namedtuple

    import mlflow
    from mlflow.entities.model_registry import ModelVersion

    # TODO: use .set_env() for operator
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-service.mlflow:8081"
    # final model name: model_name.dimension
    model_name = model_name if dimension is None else f"{model_name}.{dimension}"
    mv: ModelVersion = mlflow.register_model(artifact_uri, model_name)
    print(f"published successfully as {mv.name}:{mv.version}")
    name_version = namedtuple("ModelVersion", ["name", "version"])
    return name_version(mv.name, mv.version)


publish_model_op = create_component_from_func(
    func=_publish_model,
    base_image="dmitryb/wine-quality-predictor:latest",
    packages_to_install=[],
)


def deploy_model_op(model_name: str, model_version: int, dimension: int):
    return dsl.ContainerOp(
        name="model-deployment",
        image="library/bash:4.4.23",
        command=["sh", "-c"],
        arguments=[
            f"echo updating serving for model={model_name}:{model_version} on dimention={dimension}"
        ],
    )


def send_notification_op(msg: str):
    return dsl.ContainerOp(
        name="validation-failed",
        image="library/bash:4.4.23",
        command=["sh", "-c"],
        arguments=["echo", f"{msg}"],
    )
