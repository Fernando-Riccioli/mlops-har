"""API che serve il modello esportato dal registry (cartella MODEL_DIR).

Endpoint:
    GET  /health   -> {"status": "ok"}           (per Docker HEALTHCHECK / orchestratori)
    GET  /info     -> versione, alias, hash dei dati, commit, metrica
    POST /predict  -> {"instances": [[561 numeri], ...]} oppure
                      {"records": [{"tBodyAcc-mean()-X": ..., ...}, ...]}
                      risponde {"predictions": [...], "model_version": "..."}

Il modello è caricato come pyfunc MLflow: lo schema registrato (nomi e tipi delle 561
feature) viene controllato a ogni richiesta, e un input sbagliato dà 422, non una
predizione silenziosamente sbagliata.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))

modello = mlflow.pyfunc.load_model(str(MODEL_DIR))
INFO = json.loads((MODEL_DIR / "model_info.json").read_text())
COLONNE = modello.metadata.get_input_schema().input_names()

app = FastAPI(title="har-classificatore", version=INFO["version"])


class Richiesta(BaseModel):
    instances: list[list[float]] | None = None
    records: list[dict[str, float]] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    return {**INFO, "n_features": len(COLONNE)}


@app.post("/predict")
def predict(r: Richiesta) -> dict:
    if (r.instances is None) == (r.records is None):
        raise HTTPException(422, "passa esattamente uno fra 'instances' e 'records'")
    if r.instances is not None:
        sbagliate = [i for i, riga in enumerate(r.instances) if len(riga) != len(COLONNE)]
        if sbagliate:
            raise HTTPException(422, f"le righe {sbagliate} non hanno {len(COLONNE)} valori")
        df = pd.DataFrame(r.instances, columns=COLONNE)
    else:
        df = pd.DataFrame(r.records)
    try:
        pred = modello.predict(df)
    except Exception as e:  # schema MLflow non rispettato (colonne mancanti, tipi)
        raise HTTPException(422, f"input non valido per lo schema del modello: {e}") from e
    return {"predictions": [str(p) for p in pred], "model_version": INFO["version"]}
