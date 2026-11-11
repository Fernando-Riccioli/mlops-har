# mlops-har: riconoscimento attività (UCI HAR) con MLflow, W&B e DVC

[English](README.md) · **Italiano**

Progetto personale per imparare MLOps su dati pubblici. Il modello è volutamente
semplice (scikit-learn su CPU): il centro è il **ciclo di vita** del modello, cioè
tracciare le run, registrare e promuovere il modello, versionare i dati, testare e servire.

| Pezzo | Strumento | Dove |
|---|---|---|
| Dati e split per soggetto | UCI HAR, parquet | `scripts/scarica_dati.py` |
| Versionamento dati (A) | DVC + remote locale | `data/*.dvc`, `.dvc/config.local` |
| Versionamento dati (B) | manifest sha256 nella config di ogni run | `src/har/manifest.py` |
| Experiment tracking | MLflow (SQLite locale) | `src/har/train.py` |
| Model Registry, alias, gate | MLflow | `src/har/registry.py`, `MODEL_CARD.md` |
| Seconda dashboard | Weights & Biases (cloud) | `train.py --wandb` |
| Test e CI | pytest, smoke test, GitHub Actions | `tests/`, `scripts/smoke_test.py`, `.github/workflows/ci.yml` |
| Serving | FastAPI + Docker, modello @produzione | `serving/`, `scripts/esporta_modello.py` |

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
- **Model card**: `MODEL_CARD.md` è la descrizione del registered model (compito, dati, limiti);
  la versione italiana è in `MODEL_CARD.it.md`.

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
installando le versioni esatte di `requirements.lock`. Ha tre job in catena:
- **test**: pytest, senza dati.
- **smoke**: riscarica UCI HAR (l'archivio è in cache) e controlla due cose. La prima: i dati
  riconvertiti hanno lo **stesso hash** di `data/manifest.json`, cioè la pipeline dei dati è
  riproducibile su un'altra macchina. La seconda: un training breve supera **F1 macro ≥ 0,90**,
  così una modifica che rompe il modello rende la CI rossa prima del merge. Questo training usa
  un registro MLflow temporaneo.
- **serving**: rifà tutto il ciclo di vita su un registro MLflow pulito. Allena il modello, lo
  registra come `@candidata`, lo promuove a `@produzione` solo se passa il gate, lo esporta,
  costruisce l'immagine Docker, avvia il container e lo prova con `scripts/prova_container.py`.

### 6. Servire il modello in un container

```bash
.venv/bin/python -m scripts.esporta_modello          # @produzione -> serving/model/ + model_info.json
docker build -t har-classificatore:v1 serving/
docker run --rm -d --name har -p 127.0.0.1:8080:8080 har-classificatore:v1
.venv/bin/python -m scripts.prova_container          # health, info, predict, input sbagliato
docker stop har
```

- L'immagine contiene **una** versione del modello, esportata dal registry. Il container non
  parla con MLflow e `/info` dice quale versione serve, con hash dei dati e commit.
  Per aggiornarlo si sposta l'alias, si riesporta e si ricostruisce l'immagine.
- Il server è FastAPI e carica il modello come *pyfunc* MLflow, quindi lo **schema registrato**
  (561 feature con nome e tipo) viene controllato a ogni richiesta. Un input sbagliato
  restituisce 422 invece di una predizione sbagliata.
- `mlflow-skinny` al posto di `mlflow`: serve solo caricare il modello, non il server di
  tracking. Il container gira senza root e ha un `HEALTHCHECK` su `/health`.

```bash
curl -s localhost:8080/info
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' \
     -d '{"instances": [[...561 numeri...]]}'
```

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

Stessi dati (stesso hash), stesso codice e stesse versioni dei pacchetti: la regressione
logistica dà F1 test 0,9550 sul PC e 0,9544 sul runner di GitHub Actions. La differenza viene dal
calcolo in virgola mobile su CPU e librerie numeriche diverse. Per questo il gate usa una soglia
e non un valore esatto.

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
`pandas.util.hash_pandas_object`). Per confrontare *i file* va bene così: con le versioni del
lock, l'hash coincide anche sul runner di GitHub Actions.

Su Databricks lo stesso ruolo lo svolgono le **tabelle Delta**: il *time travel*
(`VERSION AS OF`) permette di rileggere una versione passata, e MLflow è già integrato, con
Unity Catalog come Model Registry.

## Nomi in italiano

Il codice usa nomi in italiano. Le opzioni e gli alias principali sono:

| Nome | Significato |
|---|---|
| `--modello`, `--seme` | model, seed |
| `candidata`, `produzione` | alias candidate, production |
| `promuovi`, `stato` | promote, status |
| `scarica_dati`, `esporta_modello`, `prova_container` | download data, export model, test container |
