# Model card: har-classificatore

[English](MODEL_CARD.md) · **Italiano**

**Compito.** Riconoscere 6 attività umane (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING, LAYING) da finestre di 2,56 s di accelerometro e giroscopio di uno
smartphone in vita (50 Hz), descritte dalle 561 feature calcolate dagli autori del dataset.

**Dati.** UCI HAR, *Human Activity Recognition Using Smartphones* (Anguita et al., 2013),
licenza CC BY 4.0. 30 volontari di 19-48 anni, un solo modello di telefono (Samsung Galaxy S II).
Split ufficiale per soggetto: 21 in train (7352 finestre), 9 in test (2947 finestre).
La versione esatta dei dati è nell'hash `dataset_sha256`, salvato come tag su ogni versione.

**Valutazione.**
- Metrica di promozione: F1 macro sul test (i 9 soggetti mai visti). Macro perché le classi
  pesano uguale, anche se la frequenza è diversa.
- Controllo di stabilità: F1 macro in validazione incrociata *per soggetto* (GroupKFold a 5
  fold sul train), così un soggetto non sta mai sia in train sia in validazione.
- Gate di produzione: F1 macro test ≥ 0,90 e non peggiore della versione in produzione.

**Limiti noti.**
- Un solo telefono e una sola posizione (vita): su altri dispositivi o in tasca/mano la
  qualità non è misurata.
- 30 soggetti, giovani adulti: la generalizzazione a anziani o a persone con disturbi del
  movimento non è misurata.
- L'errore più frequente è fra le posture statiche. Per la v1 (regressione logistica), sulle
  2947 finestre del test, 58 SITTING sono letti come STANDING e 14 al contrario: 72 dei 132
  errori. Il secondo gruppo sono le scale: 23 WALKING_UPSTAIRS letti come WALKING e 18
  WALKING_DOWNSTAIRS letti come WALKING_UPSTAIRS.
- Le feature sono quelle precalcolate del dataset: in produzione andrebbe riprodotta
  esattamente la stessa catena di estrazione.

**Uso previsto.** Esercizio didattico di MLOps (tracking, registry, versionamento dei dati).
Non per uso clinico né per decisioni sulle persone.
