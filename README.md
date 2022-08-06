# Wine quality predictor

Test project to demonstrate the capabilities of mlflow/argo pipelines

## Requirements

Infra:
* Local k8s cluster
* Minio
* Mlflow
* Argo

## Environment variables

Set the following envs :

```bash
export PYTHONPATH=.
export MLFLOW_TRACKING_URI=http://mlflow-service.mlflow:8081
export MLFLOW_S3_ENDPOINT_URL=http://minio-service.minio:8081
export AWS_ACCESS_KEY_ID=minio123
export AWS_SECRET_ACCESS_KEY=minio123
```

## Run job locally

Create and activate conda env using `conda.yaml`

```bash
mlflow run . \
    --env-manager local \
    --entry-point train_elasticnet \
    --experiment-name wine-quality-predictor \
    --run-name run-1 \
    -P alpha=1.0 \
    -P l1_ratio=1.0

mlflow run . \
    --env-manager local \
    --entry-point train_dnn \
    --experiment-name wine-quality-predictor \
    --run-name run-2
```

## Run on kubernetes

Update MLproject file to set

```yaml
# use conda env
conda_env: conda.yaml
# or
# use docker
# docker_env: 
#   image: dmitryb/wine-quality-predictor:base
```

Build base docker image (once)

```bash
docker build -t dmitryb/wine-quality-predictor:base -f ./Dockerfiles/Dockerfile.project .
docker push dmitryb/wine-quality-predictor:base
```

Start training job (default namespace)

```bash
mlflow run . \
    --backend kubernetes --backend-config k8s/k8s_cfg.json \
    --entry-point train_elasticnet \
    --experiment-name wine-quality-predictor \
    --run-name run-1 \
    -P alpha=1.0 \
    -P l1_ratio=1.0
```

## Generate and run pipeline

To generate pipeline code (argo yaml): `python pipelines/wine_quality_predictor.py`
Check `pipelines/wine_quality_predictor.yaml `

To run pipeline: `argo -n argo submit pipelines/wine_quality_predictor.yaml`
Check pipeline: `argo -n argo list | head -n 2`