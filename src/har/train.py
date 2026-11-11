"""Training su UCI HAR con experiment tracking MLflow (e, a richiesta, W&B).

Ogni run registra:
- parametri: modello, iperparametri, seme, hash del dataset (manifest) e commit git;
- metriche: accuratezza e F1 macro sul test (9 soggetti mai visti) e in validazione
  incrociata per soggetto sul train (GroupKFold: nessun soggetto in due fold);
- artefatti: modello sklearn con firma, matrice di confusione, manifest del dataset.

Esempi:
    python -m src.har.train --modello logreg --C 1.0
    python -m src.har.train --modello rf --n-estimators 300 --max-depth 20
    python -m src.har.train --modello rf --wandb
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.har.manifest import crea_manifest

RADICE = Path(__file__).resolve().parents[2]
DATI = RADICE / "data" / "processed"
ESPERIMENTO = "har-attivita"
# Il Model Registry richiede un backend a database: un file SQLite locale basta.
# Si cambia con la variabile MLFLOW_TRACKING_URI (per esempio un server o Databricks).
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{RADICE / 'mlflow.db'}")


def carica() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    tr = pd.read_parquet(DATI / "train.parquet")
    te = pd.read_parquet(DATI / "test.parquet")
    feat = [c for c in tr.columns if c not in ("activity", "subject")]
    return tr[feat], tr["activity"], tr["subject"], te[feat], te["activity"]


def costruisci(a: argparse.Namespace):
    if a.modello == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
    return RandomForestClassifier(
        n_estimators=a.n_estimators, max_depth=a.max_depth, n_jobs=-1, random_state=a.seme
    )


def commit_git() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RADICE, text=True).strip()
        sporco = subprocess.call(["git", "diff", "--quiet"], cwd=RADICE) != 0
        return sha + ("-dirty" if sporco else "")
    except Exception:
        return "nessun-commit"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modello", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--seme", type=int, default=0)
    p.add_argument("--cv", type=int, default=5, help="fold di GroupKFold per soggetto")
    p.add_argument("--wandb", action="store_true", help="registra la run anche su W&B")
    a = p.parse_args()

    Xtr, ytr, gruppi, Xte, yte = carica()
    manifest = crea_manifest(DATI)
    parametri = {
        "modello": a.modello, "seme": a.seme, "cv_fold": a.cv,
        **({"C": a.C} if a.modello == "logreg" else
           {"n_estimators": a.n_estimators, "max_depth": a.max_depth}),
        "dataset_sha256": manifest["dataset_sha256"],
        "git_commit": commit_git(),
        "n_train": len(Xtr), "n_test": len(Xte),
    }

    run_wandb = None
    if a.wandb:
        import wandb
        run_wandb = wandb.init(project="mlops-har", config=parametri, job_type="train")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(ESPERIMENTO)
    with mlflow.start_run(run_name=f"{a.modello}-seme{a.seme}") as run:
        mlflow.log_params(parametri)
        mlflow.set_tags({"dataset": "UCI HAR", "split": "per soggetto (ufficiale)"})

        modello = costruisci(a)
        cv = cross_val_score(modello, Xtr, ytr, groups=gruppi, cv=GroupKFold(a.cv),
                             scoring="f1_macro", n_jobs=1)
        modello.fit(Xtr, ytr)
        pred = modello.predict(Xte)
        metriche = {
            "cv_f1_macro_media": float(cv.mean()),
            "cv_f1_macro_std": float(cv.std()),
            "test_accuracy": float(accuracy_score(yte, pred)),
            "test_f1_macro": float(f1_score(yte, pred, average="macro")),
        }
        mlflow.log_metrics(metriche)

        classi = sorted(yte.unique())
        cm = pd.DataFrame(confusion_matrix(yte, pred, labels=classi), index=classi, columns=classi)
        mlflow.log_text(cm.to_csv(), "confusion_matrix.csv")
        mlflow.log_dict(manifest, "dataset_manifest.json")
        mlflow.sklearn.log_model(
            modello, name="modello",
            signature=infer_signature(Xte.head(5), pred[:5]),
            input_example=Xte.head(2),
            # MLflow 3 serializza con skops, che rifiuta i tipi non verificati. Il modello
            # l'abbiamo creato noi: dichiaro fidato SOLO il tipo segnalato (nodi degli alberi).
            **({"skops_trusted_types": ["sklearn.tree._tree.Tree"]} if a.modello == "rf" else {}),
        )

        if run_wandb is not None:
            import wandb
            run_wandb.log(metriche)
            run_wandb.log({"confusion_matrix": wandb.Table(dataframe=cm.reset_index())})
            run_wandb.summary["mlflow_run_id"] = run.info.run_id
            run_wandb.finish()

        print(json.dumps({"run_id": run.info.run_id, **metriche}, indent=2))


if __name__ == "__main__":
    main()
