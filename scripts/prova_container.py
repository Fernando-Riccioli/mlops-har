"""Test del container in esecuzione (usato anche dalla CI): esce con codice 1 se qualcosa non va.

Controlla: /health risponde (aspetta fino a --attendi secondi che il server parta), /info
espone versione e 561 feature, /predict su 200 righe del test supera --soglia di
accuratezza, e un input sbagliato riceve 422 invece di una predizione.

    python -m scripts.prova_container                  # http://127.0.0.1:8080
    python -m scripts.prova_container --url http://... --attendi 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def aspetta(url: str, secondi: float) -> bool:
    fine = time.monotonic() + secondi
    while time.monotonic() < fine:
        try:
            if chiama(f"{url}/health")[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--attendi", type=float, default=30, help="secondi di attesa per /health")
    p.add_argument("--soglia", type=float, default=0.90, help="accuratezza minima su 200 righe")
    a = p.parse_args()
    u, esiti = a.url, []

    def verifica(nome: str, ok: bool, dettaglio: str) -> None:
        esiti.append(ok)
        print(f"{'OK ' if ok else 'KO '} {nome}: {dettaglio}")

    verifica("health", aspetta(u, a.attendi), f"{u}/health entro {a.attendi:.0f} s")
    if not esiti[-1]:
        return 1

    s, info = chiama(f"{u}/info")
    verifica("info", s == 200 and info.get("n_features") == 561,
             f"{s}, v{info.get('version')} @{info.get('alias')}, "
             f"dataset {info.get('dataset_sha256', '')[:12]}")

    te = pd.read_parquet(DATI).sample(200, random_state=0)
    X = te.drop(columns=["activity", "subject"])
    s, r = chiama(f"{u}/predict", {"instances": X.values.tolist()})
    acc = (pd.Series(r.get("predictions", [])) == te["activity"].values).mean() if s == 200 else 0.0
    verifica("predict", s == 200 and acc >= a.soglia,
             f"{s}, accuratezza su 200 righe del test {acc:.3f} (soglia {a.soglia})")

    s, r = chiama(f"{u}/predict", {"instances": [[0.0] * 10]})
    verifica("input sbagliato", s == 422, f"{s} -> {r.get('detail')}")

    return 0 if all(esiti) else 1


if __name__ == "__main__":
    sys.exit(main())
