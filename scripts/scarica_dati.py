"""Scarica UCI HAR (Human Activity Recognition Using Smartphones) e lo converte in parquet.

Fonte: https://archive.ics.uci.edu/dataset/240 (licenza CC BY 4.0).
Uscita: data/raw/UCI HAR Dataset/ (originale) e data/processed/{train,test}.parquet.

Lo split train/test è quello ufficiale, PER SOGGETTO (21 persone in train, 9 in test):
nessuna persona compare in entrambi, quindi niente leakage fra soggetti.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
RADICE = Path(__file__).resolve().parents[1]
RAW = RADICE / "data" / "raw"
PROC = RADICE / "data" / "processed"


def scarica() -> Path:
    cartella = RAW / "UCI HAR Dataset"
    if cartella.exists():
        print(f"già presente: {cartella}")
        return cartella
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"scarico {URL}")
    esterno = urllib.request.urlopen(URL, timeout=120).read()
    # Lo zip UCI contiene a sua volta "UCI HAR Dataset.zip".
    with zipfile.ZipFile(io.BytesIO(esterno)) as z:
        interno = next(n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip"))
        with zipfile.ZipFile(io.BytesIO(z.read(interno))) as z2:
            z2.extractall(RAW)
    return cartella


def _nomi_feature(cartella: Path) -> list[str]:
    righe = (cartella / "features.txt").read_text().splitlines()
    nomi = [r.split(" ", 1)[1] for r in righe]
    # Alcuni nomi sono duplicati nel file originale: li rendo unici col loro indice.
    visti: dict[str, int] = {}
    unici = []
    for i, n in enumerate(nomi):
        unici.append(f"{n}__{i}" if n in visti else n)
        visti[n] = i
    return unici


def converti(cartella: Path) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    nomi = _nomi_feature(cartella)
    etichette = dict(
        r.split(" ", 1) for r in (cartella / "activity_labels.txt").read_text().splitlines()
    )
    for parte in ("train", "test"):
        d = cartella / parte
        X = pd.read_csv(d / f"X_{parte}.txt", sep=r"\s+", header=None, names=nomi)
        y = pd.read_csv(d / f"y_{parte}.txt", header=None)[0].astype(str).map(etichette)
        s = pd.read_csv(d / f"subject_{parte}.txt", header=None)[0]
        X["activity"] = y.values
        X["subject"] = s.values
        out = PROC / f"{parte}.parquet"
        X.to_parquet(out, index=False)
        print(f"{out.relative_to(RADICE)}: {len(X)} righe, {s.nunique()} soggetti")


if __name__ == "__main__":
    converti(scarica())
