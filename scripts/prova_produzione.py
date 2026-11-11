"""Carica il modello per ALIAS (non per numero di versione) e lo prova su 5 righe del test.

Così lavora chi serve il modello: chiede "@produzione" e il registry decide quale versione è.
"""
import mlflow
import pandas as pd

from src.har.train import DATI, TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)
modello = mlflow.pyfunc.load_model("models:/har-classificatore@produzione")
te = pd.read_parquet(DATI / "test.parquet").sample(5, random_state=0)
pred = modello.predict(te.drop(columns=["activity", "subject"]))
for vero, p in zip(te["activity"], pred):
    print(f"{'OK ' if vero == p else 'NO '} vero={vero:<20} predetto={p}")
