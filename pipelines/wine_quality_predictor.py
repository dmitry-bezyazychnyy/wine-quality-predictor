from typing import List

import kfp.dsl as dsl
from kfp import compiler
from kfp.components import create_component_from_func
import warnings

from jobs.train_elasticnet.train import train_model
from yoda_v2.pipeline.operators import (
    PythonJobOp,
    publish_model_op,
    deploy_model_op,
    send_notification_op,
)


def train_op(alpha: float, l1_ratio: float, dimension: int):
    """ Trains a model with provided params for dimention """
    return PythonJobOp(
        f=train_model,
        cmd=["jobs/train_elasticnet/train.py"],
        args=[f"--alpha {alpha} --l1_ratio {l1_ratio} --dimension {dimension}"],
        image="dmitryb/wine-quality-predictor:latest",
        name="train-elasticnet-model",
    )


def validate_model(dimension: int, rmse: float) -> int:
    """ 
        Validates a model using custom validation strategu
        returns 1 if validation was successful, otherwise 0
    """
    print(f"validating model for dimetion: {dimension}")
    if rmse > 0.8:
        return 1
    else:
        return 0


# convert function to container operator
validate_op = create_component_from_func(
    func=validate_model,
    base_image="dmitryb/wine-quality-predictor:latest",
    packages_to_install=[],
)


@dsl.pipeline(
    name="wine-quality-predictor",
    description="Train model for wine-quality-predictor project",
)
def sequential_pipeline(
    target_dimensions: List[int] = [1, 2, 3], alpha: float = 0.7, l1_ratio: float = 0.5
):
    # set num of concurrently running operators
    dsl.get_pipeline_conf().set_parallelism(3)

    with dsl.ParallelFor(target_dimensions) as dimension:

        train_task = train_op(alpha=alpha, l1_ratio=l1_ratio, dimension=dimension)
        train_task.set_cpu_limit("100m").set_memory_limit("256Mi").set_retry(
            1
        )  # TODO: use predefined resources: xs,s,m,l,xl

        validate_task = validate_op(
            dimension=dimension, rmse=train_task.outputs["rmse"]
        )
        validate_task.set_cpu_limit("100m").set_memory_limit("256Mi")

        with dsl.Condition(validate_task.output == 1):
            publish_task = publish_model_op(
                artifact_uri=train_task.outputs["model_url"],
                model_name="wine-quality.elasticnet.v1",
                dimension=dimension,
            )
            _ = deploy_model_op(
                model_name=publish_task.outputs["name"],
                model_version=publish_task.outputs["version"],
                dimension=dimension,
            )
        # or (exclusive)
        with dsl.Condition(validate_task.output == 0):
            _ = send_notification_op(
                msg=f"Validation model for dimetion {dimension} failed. Check logs for details"
            )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    compiler.Compiler().compile(
        pipeline_func=sequential_pipeline, package_path=__file__.replace(".py", ".yaml")
    )

