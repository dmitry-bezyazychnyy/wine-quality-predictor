import click
import mlflow
import warnings
import time
import logging as log
import argparse
from typing import Dict, Any
from ray import tune

from jobs.train_elasticnet.train import train_model, Result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(
        description="hps for wine-quality-predictor elsticnet model"
    )
    parser.add_argument("--num_samples", type=int, default=1, help="num_samples")
    parser.add_argument(
        "--max_concurrent_trials", type=int, default=3, help="max_concurrent_trials"
    )
    parser.add_argument("--dimension", type=int, default=1, help="dimension")
    parser.add_argument(
        "--exp",
        type=str,
        default="wine-quality-predictor",
        help="max_concurrent_trials",
    )

    args = parser.parse_args()
    log.info(f"args: {args}")

    search_space = {
        "alpha": tune.grid_search([0.5, 0.7, 1.0]),
        "l1_ratio": tune.choice([0.5, 0.7, 1.0]),
    }
    tags = {"batch_id": str(int(time.time()))}

    def target_fn(h_params) -> Dict[str, float]:
        mlflow.set_experiment(experiment_name=args.exp)
        r: Result = train_model(**h_params, dimension=args.dimension, tags=tags)
        tune.report(score=r.rmse)

    analysis = tune.run(
        target_fn,
        config=search_space,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent_trials,
    )
    best_config = analysis.get_best_config(metric="score", mode="min")
    click.echo(f"best config: {best_config}")

