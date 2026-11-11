# mlops-personale: riconoscimento attività (UCI HAR) con MLflow, W&B e DVC

Progetto personale per imparare MLOps su dati pubblici. Il modello è volutamente
semplice (scikit-learn su CPU): il centro è il **ciclo di vita** del modello, cioè
tracciare le run, registrare e promuovere il modello e versionare i dati.

| Pezzo | Strumento | Dove |
|---|---|---|
| Dati e split per soggetto | UCI HAR, parquet | `scripts/scarica_dati.py` |
| Versionamento dati (A) | DVC + remote locale | `data/*.dvc`, `.dvc/config` |
| Versionamento dati (B) | manifest sha256 nella config di ogni run | `src/har/manifest.py` |
| Experiment tracking | MLflow (SQLite locale) | `src/har/train.py` |
| Model Registry, alias, gate | MLflow | `src/har/registry.py`, `MODEL_CARD.md` |
| Seconda dashboard | Weights & Biases (cloud) | `train.py --wandb` |
| Test e CI | pytest, smoke test, GitHub Actions | `tests/`, `scripts/smoke_test.py`, `.github/workflows/ci.yml` |

## Rifare tutto da zero

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock      # versioni esatte (o requirements.txt)
```

### 1. Dati

Due strade, a scelta:

```bash
.venv/bin/python scripts/scarica_dati.py         # riscarica da UCI e riconverte
# oppure, con un remote DVC (qui è una cartella locale, fuori da git):
.venv/bin/dvc remote add --local -d locale ~/dvc-remote/mlops-personale
.venv/bin/dvc pull
```

Il remote sta in `.dvc/config.local`, che non va in git: ognuno usa il proprio (una
cartella, S3, Azure Blob, GCS). Nel repo restano solo i puntatori `data/*.dvc`.

Poi si controlla che i dati siano esattamente quelli attesi:

```bash
.venv/bin/dvc status                             # "up to date" = stessi byte del .dvc in git
.venv/bin/python -m src.har.manifest data/processed   # confronta dataset_sha256 con la run
```

Lo split è quello ufficiale **per soggetto**: 21 persone in train (7352 finestre) e 9 in test
(2947). Anche la validazione incrociata usa `GroupKFold` per soggetto. Se una persona
finisse sia in train sia in validazione, la metrica premierebbe un modello che riconosce *chi*
cammina e non *come*: è leakage.

### 2. Training con tracking MLflow

```bash
.venv/bin/python -m src.har.train --modello logreg --C 1.0
.venv/bin/python -m src.har.train --modello rf --n-estimators 200
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

Ogni run registra:
- **parametri**: gli iperparametri, il seme, `dataset_sha256` e `git_commit`, che finisce in
  `-dirty` se ci sono modifiche non committate;
- **metriche**: F1 macro e accuratezza sul test, F1 macro in validazione incrociata (media e std);
- **artefatti**: il modello con la firma di input e output, la matrice di confusione e il manifest.

Nella UI si selezionano le run e si usa **Compare**: parametri affiancati e grafici fra metriche.

### 3. Model Registry

```bash
.venv/bin/python -m src.har.registry candidata   # registra la run migliore, alias @candidata
.venv/bin/python -m src.har.registry promuovi    # @candidata -> @produzione se passa il gate
.venv/bin/python -m src.har.registry stato
.venv/bin/python -m scripts.prova_produzione     # carica models:/har-classificatore@produzione
```

- **Versione**: ogni registrazione crea una versione numerata e immutabile, collegata alla run,
  quindi anche ai parametri e all'hash dei dati. I tag della versione ripetono
  `dataset_sha256` e `git_commit`.
- **Alias**: etichette mobili. Chi serve il modello carica `@produzione` e non sa il numero
  di versione. Promuovere significa spostare l'alias; per il rollback basta rimetterlo sulla
  versione precedente. Gli alias sostituiscono i vecchi *stage* (Staging/Production), deprecati.
- **Gate** (in `registry.py`): F1 macro test ≥ 0,90 e non peggiore della produzione.
- **Model card**: `MODEL_CARD.md` è la descrizione del registered model (compito, dati, limiti).

### 4. W&B sulle stesse run

```bash
.venv/bin/wandb login                            # una volta, con la TUA chiave API
.venv/bin/python -m src.har.train --modello logreg --C 1.0 --wandb
```

Con `--wandb` la stessa run finisce **sia** in MLflow **sia** nel progetto W&B `mlops-har`,
con la stessa config (hash dei dati incluso) e l'id della run MLflow nel summary, così le due
dashboard si incrociano.

### 5. Test e CI

```bash
.venv/bin/python -m pytest -q               # unit test (manifest)
.venv/bin/python -m scripts.smoke_test      # hash dei dati + soglia del modello
```

La CI (`.github/workflows/ci.yml`, GitHub Actions) gira a ogni push su `main` e a ogni PR,
installando le versioni esatte di `requirements.lock`. Ha due job:
- **test**: pytest, senza dati.
- **smoke**: parte solo se `test` passa. Riscarica UCI HAR (l'archivio è in cache) e
  controlla due cose. La prima: i dati riconvertiti hanno lo **stesso hash** di
  `data/manifest.json`, cioè la pipeline dei dati è riproducibile su un'altra macchina. La
  seconda: un training breve supera **F1 macro ≥ 0,90**, così una modifica che rompe il
  modello rende la CI rossa prima del merge.

Il training dello smoke test usa un registro MLflow temporaneo e non tocca quello del progetto.

## Risultati (11-11-2026)

Dati UCI HAR, split ufficiale per soggetto, `dataset_sha256 = a52a6144a991…`. Una run per
configurazione, CPU, seme 0. F1 = F1 macro sulle 6 classi.

| Modello | F1 test (9 soggetti) | Accuratezza test | F1 CV per soggetto (5 fold, media ± std) |
|---|---|---|---|
| StandardScaler + LogisticRegression C=1 | 0,955 | 0,955 | 0,933 ± 0,042 |
| RandomForest 200 alberi | 0,926 | 0,928 | 0,916 ± 0,045 |

La regressione logistica è in produzione (v1). La random forest è stata registrata come
candidata (v2) ed è stata **bocciata dal gate** perché non migliora la produzione. La std
della CV, circa 0,04, viene dalla differenza fra soggetti: con una sola run la differenza fra i
due modelli sul test (0,03) è dello stesso ordine e non va letta come definitiva.

## DVC o manifest sha256? Il confronto

| | DVC | Manifest sha256 |
|---|---|---|
| Cosa salva in git | un puntatore `.dvc` (md5 della cartella) | niente di obbligatorio: l'hash va nei parametri di ogni run |
| Recupera i dati di una versione vecchia | **sì**: `git checkout <commit> && dvc checkout` | **no**: dimostra quali dati erano, ma non li ricostruisce |
| Dove stanno i dati | cache `.dvc/cache` + remote (cartella, S3, Azure, GCS…) | dove già sono |
| Costo | uno strumento in più, la cache occupa spazio (qui circa 320 MB in più) | circa 40 righe di Python, nessuna dipendenza |
| Pipeline | `dvc.yaml` rilancia solo i passi i cui input sono cambiati | nessuna |
| Legame con la run | indiretto: run → commit git → file `.dvc` | diretto: l'hash è un parametro della run, filtrabile in MLflow |

In questo progetto si usano **entrambi**: DVC per poter riavere i dati, l'hash nella run per
vedere a colpo d'occhio, in MLflow e W&B, quali run hanno visto gli stessi dati.

Trappola osservata: il parquet contiene la versione di pyarrow che l'ha scritto. Se si
riconverte con un altro pyarrow, i numeri sono identici ma l'hash dei file cambia. Per
confrontare *il contenuto* servirebbe un hash dei dati letti (per esempio
`pandas.util.hash_pandas_object`). Per confrontare *i file* va bene così.

Su Databricks lo stesso ruolo lo svolgono le **tabelle Delta**: il *time travel*
(`VERSION AS OF`) permette di rileggere una versione passata, e MLflow è già integrato, con
Unity Catalog come Model Registry.

## Prossimi passi

- Un container che serve `@produzione` (`mlflow models serve` o FastAPI).
