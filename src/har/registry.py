"""Model Registry MLflow: registra, mette alias, promuove con un controllo (gate).

Concetti:
- Registered model "har-classificatore": il "nome" del modello, con versioni 1, 2, 3...
- Versione: punta a un modello loggato da una run (quindi a parametri, metriche, hash dei dati).
- Alias: un'etichetta mobile su UNA versione ("candidata", "produzione"). Chi serve il modello
  carica "models:/har-classificatore@produzione" e non deve sapere il numero di versione.
  Promuovere = spostare l'alias; tornare indietro = rimetterlo sulla versione di prima.

Comandi:
    python -m src.har.registry candidata          # registra la run migliore (test_f1_macro)
    python -m src.har.registry candidata --run-id ID   # oppure una run precisa
    python -m src.har.registry promuovi           # candidata -> produzione, se passa il gate
    python -m src.har.registry stato              # elenca versioni e alias
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.har.train import ESPERIMENTO, TRACKING_URI

NOME = "har-classificatore"
METRICA = "test_f1_macro"
SOGLIA_MINIMA = 0.90  # gate assoluto: sotto questa F1 macro non si va in produzione
MARGINE = 0.0         # gate relativo: la candidata deve fare almeno quanto la produzione + margine
CARD = Path(__file__).resolve().parents[2] / "MODEL_CARD.md"


def client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def run_migliore(c: MlflowClient):
    exp = c.get_experiment_by_name(ESPERIMENTO)
    runs = c.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'",
                         order_by=[f"metrics.{METRICA} DESC"], max_results=1)
    if not runs:
        raise SystemExit("nessuna run: lancia prima src.har.train")
    return runs[0]


def candidata(c: MlflowClient, run_id: str | None = None) -> None:
    run = c.get_run(run_id) if run_id else run_migliore(c)
    uri = f"runs:/{run.info.run_id}/modello"
    # Se il modello loggato è salvato come LoggedModel (MLflow 3), recupero il suo uri.
    modelli = c.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
    )
    if modelli:
        uri = modelli[0].model_uri
    v = mlflow.register_model(uri, NOME)
    p = run.data.params
    c.update_model_version(
        NOME, v.version,
        description=(f"{p.get('modello')} | {METRICA}={run.data.metrics[METRICA]:.4f} | "
                     f"dataset_sha256={p.get('dataset_sha256', '')[:12]} | "
                     f"git={p.get('git_commit', '')[:12]}"),
    )
    for k in ("modello", "dataset_sha256", "git_commit"):
        c.set_model_version_tag(NOME, v.version, k, p.get(k, ""))
    c.set_model_version_tag(NOME, v.version, METRICA, f"{run.data.metrics[METRICA]:.4f}")
    c.set_registered_model_alias(NOME, "candidata", v.version)
    if CARD.exists():
        c.update_registered_model(NOME, description=CARD.read_text())
    print(f"versione {v.version} = candidata (run {run.info.run_id}, "
          f"{METRICA}={run.data.metrics[METRICA]:.4f})")


def _metrica(c: MlflowClient, alias: str) -> float | None:
    try:
        v = c.get_model_version_by_alias(NOME, alias)
    except mlflow.exceptions.MlflowException:
        return None
    return float(v.tags[METRICA])


def promuovi(c: MlflowClient) -> None:
    cand = c.get_model_version_by_alias(NOME, "candidata")
    f1_cand = _metrica(c, "candidata")
    f1_prod = _metrica(c, "produzione")
    print(f"candidata v{cand.version}: {METRICA}={f1_cand:.4f}; produzione: {f1_prod}")
    if f1_cand < SOGLIA_MINIMA:
        raise SystemExit(f"BOCCIATA: {f1_cand:.4f} < soglia {SOGLIA_MINIMA}")
    if f1_prod is not None and f1_cand < f1_prod + MARGINE:
        raise SystemExit(f"BOCCIATA: non migliora la produzione ({f1_prod:.4f})")
    c.set_registered_model_alias(NOME, "produzione", cand.version)
    print(f"PROMOSSA: v{cand.version} = produzione")


def stato(c: MlflowClient) -> None:
    rm = c.get_registered_model(NOME)
    alias_per_v: dict[str, list[str]] = {}
    for a, v in rm.aliases.items():
        alias_per_v.setdefault(str(v), []).append(a)
    for v in sorted(c.search_model_versions(f"name='{NOME}'"), key=lambda v: int(v.version)):
        print(f"v{v.version:>3}  {', '.join(alias_per_v.get(str(v.version), [])):<22} {v.description}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("azione", choices=["candidata", "promuovi", "stato"])
    p.add_argument("--run-id", help="per 'candidata': registra questa run invece della migliore")
    a = p.parse_args()
    c = client()
    if a.azione == "candidata":
        candidata(c, a.run_id)
    else:
        {"promuovi": promuovi, "stato": stato}[a.azione](c)
