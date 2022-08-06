import logging as log

import argparse
import warnings
import mlflow
import numpy as np
import pandas as pd
from dataclasses import dataclass

from typing import Dict, Any
from mlflow.models.signature import infer_signature
from yoda_v2.dataset import YodaDatasets
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from yoda_v2.pipeline.decorators import pipeline_operator


@dataclass
class Result:
    model_url: str
    rmse: float
    r2: float
    mae: float


def eval_metrics(actual, pred):
    """ """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


@pipeline_operator
def train_model(alpha: float, l1_ratio: float, dimension: int, **kwargs) -> Result:
    """
    train elasticnet model
    """
    log.info(f"train model for dimention: {dimension}")
    # get function input args;
    args: Dict[str, Any] = dict(map(lambda kv: (f"_{kv[0]}", kv[1]), locals().items()))

    # load Yoda dataset
    yoda_ds = YodaDatasets.get(
        name="wine-quality-ds", use_last_version=True
    )  # load the last available version
    log.info(f"dataset schema:\n {yoda_ds.schema()}")
    log.info(f"dataset partitions: {yoda_ds.num_blocks()}")

    # convert to pandas
    data: pd.DataFrame = yoda_ds.to_pandas()
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    with mlflow.start_run() as run:

        mlflow.log_params(args)
        if "tags" in kwargs: mlflow.set_tags(kwargs['tags'])

        # train model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # eval model
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        metrics = {"rmse": rmse, "r2": r2, "mae": mae}

        # model signature
        model_signature = infer_signature(train_x, lr.predict(train_x))

        mlflow.log_metrics(metrics)
        log.info(f"metrics: {metrics}")

        # log model as artifacts
        mlflow.sklearn.log_model(lr, "model", signature=model_signature)

        # return params can be used by downstream operators (pipeline)
        model_uri = "runs:/{}/model".format(run.info.run_id)
        return Result(model_url=model_uri, **metrics)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(
        description="train wine-quality-predictor elsticnet model"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="l1_ratio")
    parser.add_argument("--dimension", type=int, default=1, help="dimension")
    parser.add_argument("--seed", type=int, default=40, help="seed")

    args: Dict[str, Any] = vars(parser.parse_args())
    log.info(f"args: {args}")
    train_model(**args)
