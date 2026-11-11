"""Prova il container in esecuzione: /health, /info, /predict su 200 righe del test, e un
input sbagliato che deve dare 422. Usa solo la libreria standard per le richieste HTTP.

    python -m scripts.prova_container                  # http://127.0.0.1:8080
    python -m scripts.prova_container --url http://...
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATI = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.parquet"


def chiama(url: str, corpo: dict | None = None) -> tuple[int, dict]:
    dati = None if corpo is None else json.dumps(corpo).encode()
    req = urllib.request.Request(url, data=dati, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    u = p.parse_args().url

    print("health:", chiama(f"{u}/health"))
    s, info = chiama(f"{u}/info")
    print("info:", s, {k: info[k] for k in ("name", "version", "alias", "n_features")},
          "dataset", info.get("dataset_sha256", "")[:12])

    te = pd.read_parquet(DATI).sample(200, random_state=0)
    X = te.drop(columns=["activity", "subject"])
    s, r = chiama(f"{u}/predict", {"instances": X.values.tolist()})
    acc = (pd.Series(r["predictions"]) == te["activity"].values).mean()
    print(f"predict: {s}, v{r['model_version']}, accuratezza su 200 righe del test = {acc:.3f}")

    s, r = chiama(f"{u}/predict", {"instances": [[0.0] * 10]})
    print(f"input sbagliato (10 valori invece di 561): {s} -> {r['detail']}")


if __name__ == "__main__":
    main()
