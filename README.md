# mlops-har: human activity recognition (UCI HAR) with MLflow, W&B and DVC

**English** · [Italiano](README.it.md)

A personal project to practise MLOps on public data. The model is deliberately simple
(scikit-learn on CPU): the focus is the model **lifecycle**, that is experiment tracking,
model registration and promotion, data versioning, testing and serving.

| Piece | Tool | Where |
|---|---|---|
| Data and per-subject split | UCI HAR, parquet | `scripts/scarica_dati.py` |
| Data versioning (A) | DVC + local remote | `data/*.dvc`, `.dvc/config.local` |
| Data versioning (B) | sha256 manifest in every run's config | `src/har/manifest.py` |
| Experiment tracking | MLflow (local SQLite) | `src/har/train.py` |
| Model Registry, aliases, gate | MLflow | `src/har/registry.py`, `MODEL_CARD.md` |
| Second dashboard | Weights & Biases (cloud) | `train.py --wandb` |
| Tests and CI | pytest, smoke test, GitHub Actions | `tests/`, `scripts/smoke_test.py`, `.github/workflows/ci.yml` |
| Serving | FastAPI + Docker, @produzione model | `serving/`, `scripts/esporta_modello.py` |

The code uses Italian names. The ones you will type are listed in
[Italian names](#italian-names) at the end.

## Reproduce everything from scratch

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock      # exact versions (or requirements.txt)
```

### 1. Data

Two options:

```bash
.venv/bin/python scripts/scarica_dati.py         # download from UCI and convert
# or, with a DVC remote (here a local folder, outside git):
.venv/bin/dvc remote add --local -d locale ~/dvc-remote/mlops-personale
.venv/bin/dvc pull
```

The remote lives in `.dvc/config.local`, which is not committed: everyone uses their own
(a folder, S3, Azure Blob, GCS). The repository only holds the `data/*.dvc` pointers.

Then check that the data are exactly the expected ones:

```bash
.venv/bin/dvc status                             # "up to date" = same bytes as the .dvc in git
.venv/bin/python -m src.har.manifest data/processed   # compare dataset_sha256 with the run
```

The split is the official **per-subject** one: 21 people in train (7352 windows) and 9 in
test (2947). Cross-validation also uses `GroupKFold` by subject. If one person were both in
train and in validation, the metric would reward a model that recognises *who* is walking
rather than *how*: that is leakage.

### 2. Training with MLflow tracking

```bash
.venv/bin/python -m src.har.train --modello logreg --C 1.0
.venv/bin/python -m src.har.train --modello rf --n-estimators 200
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

Every run logs:
- **parameters**: hyperparameters, seed, `dataset_sha256` and `git_commit` (suffixed with
  `-dirty` when there are uncommitted changes);
- **metrics**: test macro-F1 and accuracy, cross-validated macro-F1 (mean and std);
- **artifacts**: the model with its input/output signature, the confusion matrix and the
  dataset manifest.

In the UI, select runs and use **Compare** for side-by-side parameters and metric plots.

### 3. Model Registry

```bash
.venv/bin/python -m src.har.registry candidata   # register the best run, alias @candidata
.venv/bin/python -m src.har.registry promuovi    # @candidata -> @produzione if it passes the gate
.venv/bin/python -m src.har.registry stato
.venv/bin/python -m scripts.prova_produzione     # load models:/har-classificatore@produzione
```

- **Version**: each registration creates an immutable, numbered version linked to its run,
  and so to its parameters and data hash. Version tags repeat `dataset_sha256` and
  `git_commit`.
- **Aliases**: movable labels. Whoever serves the model loads `@produzione` and never needs
  the version number. Promoting means moving the alias; rolling back means moving it to the
  previous version. Aliases replace the deprecated *stages* (Staging/Production).
- **Gate** (in `registry.py`): test macro-F1 ≥ 0.90 and not worse than production.
- **Model card**: `MODEL_CARD.md` is the registered model's description (task, data,
  limitations).

### 4. W&B on the same runs

```bash
.venv/bin/wandb login                            # once, with YOUR API key
.venv/bin/python -m src.har.train --modello logreg --C 1.0 --wandb
```

With `--wandb` the same run goes to **both** MLflow **and** the W&B project `mlops-har`,
with the same config (data hash included) and the MLflow run id in the W&B summary, so the
two dashboards can be cross-referenced.

### 5. Tests and CI

```bash
.venv/bin/python -m pytest -q               # unit tests (manifest)
.venv/bin/python -m scripts.smoke_test      # data hash + model quality threshold
```

CI (`.github/workflows/ci.yml`, GitHub Actions) runs on every push to `main` and on every
pull request, installing the exact versions from `requirements.lock`. It has three chained
jobs:
- **test**: pytest, no data needed.
- **smoke**: re-downloads UCI HAR (the archive is cached) and checks two things. First, the
  converted data have the **same hash** as `data/manifest.json`, so the data pipeline is
  reproducible on another machine. Second, a short training reaches **macro-F1 ≥ 0.90**, so
  a change that breaks the model turns CI red before merging. This training uses a temporary
  MLflow registry.
- **serving**: replays the whole lifecycle on a fresh MLflow registry. It trains the model,
  registers it as `@candidata`, promotes it to `@produzione` only if it passes the gate,
  exports it, builds the Docker image, starts the container and tests it with
  `scripts/prova_container.py`.

### 6. Serving the model in a container

```bash
.venv/bin/python -m scripts.esporta_modello          # @produzione -> serving/model/ + model_info.json
docker build -t har-classificatore:v1 serving/
docker run --rm -d --name har -p 127.0.0.1:8080:8080 har-classificatore:v1
.venv/bin/python -m scripts.prova_container          # health, info, predict, bad input
docker stop har
```

- The image holds **one** model version, exported from the registry. The container does not
  talk to MLflow, and `/info` reports which version it serves, with data hash and commit.
  To update it: move the alias, re-export, rebuild the image.
- The server is FastAPI and loads the model as an MLflow *pyfunc*, so the **registered
  schema** (561 named, typed features) is enforced on every request. Bad input gets a 422
  instead of a wrong prediction.
- `mlflow-skinny` instead of `mlflow`: the container only needs to load the model, not the
  tracking server. It runs as a non-root user and has a `HEALTHCHECK` on `/health`.

```bash
curl -s localhost:8080/info
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' \
     -d '{"instances": [[...561 numbers...]]}'
```

## Results (2026-11-11)

UCI HAR, official per-subject split, `dataset_sha256 = a52a6144a991…`. One run per
configuration, CPU, seed 0. F1 = macro-F1 over the 6 classes.

| Model | Test F1 (9 subjects) | Test accuracy | Per-subject CV F1 (5 folds, mean ± std) |
|---|---|---|---|
| StandardScaler + LogisticRegression C=1 | 0.955 | 0.955 | 0.933 ± 0.042 |
| RandomForest, 200 trees | 0.926 | 0.928 | 0.916 ± 0.045 |

Logistic regression is in production (v1). The random forest was registered as a candidate
(v2) and **rejected by the gate** because it does not improve on production. The CV std of
about 0.04 comes from differences between subjects: with a single run, the 0.03 test gap
between the two models is of the same order and should not be read as final.

Same data (same hash), same code, same package versions: logistic regression scores test F1
0.9550 on the workstation and 0.9544 on the GitHub Actions runner. The difference comes from
floating-point arithmetic on different CPUs and numerical libraries. That is why the gate
uses a threshold rather than an exact value.

## DVC or a sha256 manifest? A comparison

| | DVC | sha256 manifest |
|---|---|---|
| What goes into git | a `.dvc` pointer (folder md5) | nothing required: the hash goes into each run's parameters |
| Restores the data of an old version | **yes**: `git checkout <commit> && dvc checkout` | **no**: proves which data were used, but cannot rebuild them |
| Where the data live | `.dvc/cache` + a remote (folder, S3, Azure, GCS…) | wherever they already are |
| Cost | one more tool; the cache takes space (about 320 MB extra here) | about 40 lines of Python, no dependencies |
| Pipelines | `dvc.yaml` reruns only the stages whose inputs changed | none |
| Link to the run | indirect: run → git commit → `.dvc` file | direct: the hash is a run parameter, searchable in MLflow |

This project uses **both**: DVC to get the data back, the hash in each run to see at a
glance, in MLflow and W&B, which runs saw the same data.

A pitfall I hit: a parquet file records the pyarrow version that wrote it. Converting again
with another pyarrow gives identical numbers but a different file hash. To compare
*content* you would hash the loaded data (for example with
`pandas.util.hash_pandas_object`). To compare *files* this approach is fine: with the
versions pinned in the lock file, the hash also matches on the GitHub Actions runner.

On Databricks the same role is played by **Delta tables**: *time travel*
(`VERSION AS OF`) reads a past version, and MLflow is built in, with Unity Catalog as the
Model Registry.

## Italian names

| Name | Meaning |
|---|---|
| `--modello`, `--seme` | model, seed |
| `candidata`, `produzione` | aliases: candidate, production |
| `promuovi`, `stato` | promote, status |
| `scarica_dati`, `esporta_modello`, `prova_container` | download data, export model, test container |
