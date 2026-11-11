"""Esporta dal Model Registry la versione con un certo alias, per costruire l'immagine Docker.

Il container NON parla col registry: riceve una cartella con il modello e un
model_info.json (nome, versione, alias, hash dei dati, commit). Così l'immagine è
autosufficiente e dice da sola quale modello contiene.

    python -m scripts.esporta_modello                 # @produzione -> serving/model/
    python -m scripts.esporta_modello --alias candidata
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))

from src.har.registry import NOME  # noqa: E402
from src.har.train import TRACKING_URI  # noqa: E402

USCITA = RADICE / "serving" / "model"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alias", default="produzione")
    a = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    v = MlflowClient().get_model_version_by_alias(NOME, a.alias)

    if USCITA.exists():
        shutil.rmtree(USCITA)
    USCITA.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(f"models:/{NOME}@{a.alias}", dst_path=str(USCITA))

    info = {"name": NOME, "version": str(v.version), "alias": a.alias,
            "run_id": v.run_id, **dict(v.tags)}
    (USCITA / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"esportata v{v.version} (@{a.alias}) in {USCITA.relative_to(RADICE)}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
