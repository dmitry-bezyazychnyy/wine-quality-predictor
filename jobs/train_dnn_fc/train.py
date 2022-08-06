import logging as log
import warnings
import argparse
from typing import Any, Dict
from dataclasses import dataclass

import mlflow
from mlflow.models.signature import infer_signature
import tensorflow as tf
from sklearn.model_selection import train_test_split
from yoda_v2.dataset import YodaDatasets

@dataclass
class Result:
    model_url: str
    rmse: float
    mae: float


def train_model(
    n_layers: int = 3,
    dropout_rate: float = 0.1,
    n_neurons: int = 512,
    activation_f: str = "relu",
    lr: float = 0.0001,
    batch_size: int = 128,
    n_epochs: int = 10,
    **kwargs
) -> None:

    args: Dict[str, Any] = dict(map(lambda kv: (f"_{kv[0]}", kv[1]), locals().items()))

    yoda_ds = YodaDatasets.get(name="")
    data = yoda_ds.to_pandas()
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(train_x, dtype=tf.float32),
            tf.convert_to_tensor(train_y, dtype=tf.float32),
        )
    ).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(test_x, dtype=tf.float32),
            tf.convert_to_tensor(test_y, dtype=tf.float32),
        )
    ).batch(batch_size)

    with mlflow.start_run() as run:
        
        mlflow.log_params(args)
        if "tags" in kwargs: mlflow.set_tags(kwargs['tags'])

        # create model
        inputs = tf.keras.Input(shape=(11,), name="input-vector", dtype=tf.float32)
        l = inputs
        for idx in range(n_layers):
            l = tf.keras.layers.Dense(
                units=n_neurons, activation=activation_f, name=f"fc-{idx+1}"
            )(l)
            if dropout_rate:
                l = tf.keras.layers.Dropout(rate=dropout_rate)(l)
        outputs = tf.keras.layers.Dense(1, activation="relu")(l)
        m = tf.keras.Model(inputs=inputs, outputs=outputs)

        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError(),
            ],
        )
        m.summary()

        # train model
        for epoch in range(n_epochs):
            history = m.fit(
                train_ds,
                epochs=1,
                verbose=1,
                validation_data=test_ds,
                validation_steps=1,
            )
            for metric_name in history.history:
                metric_value = history.history[metric_name][0]
                mlflow.log_metric(
                    key=metric_name, value=float(metric_value), step=int(epoch)
                )
                # log.info(f" {metric_value} = {metric_value}")

        # log model as artifacts
        x,y = next(train_ds.as_numpy_iterator())
        model_signature = infer_signature(x, y)
        log.info(f"model signature: {model_signature}")
        mlflow.sklearn.log_model(lr, "model", signature=model_signature)

        # return params can be used by downstream operators (pipeline)
        model_uri = "runs:/{}/model".format(run.info.run_id)
        # return Result(model_url=model_uri, **metrics)
        return None


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(
        description="train wine-quality-predictor DNN"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=1, help="n_epochs")
    
    args: Dict[str, Any] = vars(parser.parse_args())
    log.info(f"args: {args}")
    train_model(**args)
