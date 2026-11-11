"""Controllo automatico del modello, pensato per la CI (ma gira anche in locale).

Due verifiche:
1. DATI: i dati riscaricati da UCI e riconvertiti hanno lo stesso hash di data/manifest.json
   (cioè la pipeline dei dati è riproducibile, con le versioni del lock).
2. MODELLO: un training breve (regressione logistica, CV a 2 fold) supera la soglia di
   F1 macro sul test. Se una modifica al codice rompe il modello, la CI diventa rossa.

Il training usa un registro MLflow temporaneo: la CI non tocca mlflow.db del progetto.

    python -m scripts.smoke_test            # entrambe le verifiche
    python -m scripts.smoke_test --no-dati  # solo il modello
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))

from src.har.manifest import crea_manifest  # noqa: E402

SOGLIA_F1 = 0.90


def controlla_dati() -> bool:
    atteso = json.loads((RADICE / "data" / "manifest.json").read_text())["dataset_sha256"]
    trovato = crea_manifest(RADICE / "data" / "processed")["dataset_sha256"]
    ok = atteso == trovato
    print(f"[dati]    {'OK ' if ok else 'KO '} hash atteso {atteso[:12]}, trovato {trovato[:12]}")
    return ok


def controlla_modello() -> bool:
    with tempfile.TemporaryDirectory() as d:
        esito = Path(d) / "esito.json"
        env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{d}/mlflow.db",
               "PYTHONPATH": str(RADICE)}
        subprocess.run(
            [sys.executable, "-m", "src.har.train", "--modello", "logreg", "--cv", "2",
             "--esito", str(esito)],
            cwd=d, env=env, check=True, stdout=subprocess.DEVNULL,
        )
        m = json.loads(esito.read_text())
    ok = m["test_f1_macro"] >= SOGLIA_F1
    print(f"[modello] {'OK ' if ok else 'KO '} test_f1_macro {m['test_f1_macro']:.4f} "
          f"(soglia {SOGLIA_F1})")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-dati", action="store_true", help="salta il controllo dell'hash")
    a = p.parse_args()
    esiti = ([] if a.no_dati else [controlla_dati()]) + [controlla_modello()]
    sys.exit(0 if all(esiti) else 1)
