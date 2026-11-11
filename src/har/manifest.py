"""Manifest sha256 del dataset: l'alternativa leggera a DVC.

Idea: invece di versionare i file, si versiona la loro IMPRONTA. Ogni run salva nella
sua config l'hash complessivo del dataset; se due run hanno lo stesso hash, hanno visto
esattamente gli stessi byte. Non ricostruisce i dati (DVC sì), ma dimostra quali erano.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(percorso: Path, blocco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with percorso.open("rb") as f:
        while chunk := f.read(blocco):
            h.update(chunk)
    return h.hexdigest()


def crea_manifest(cartella: Path) -> dict:
    """Hash di ogni file (ordinati per percorso) e un hash complessivo del dataset."""
    file = sorted(p for p in cartella.rglob("*") if p.is_file())
    voci = {str(p.relative_to(cartella)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in file}
    complessivo = hashlib.sha256(
        json.dumps(voci, sort_keys=True).encode()
    ).hexdigest()
    return {"cartella": str(cartella), "dataset_sha256": complessivo, "file": voci}


if __name__ == "__main__":
    import sys

    cartella = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
    m = crea_manifest(cartella)
    out = Path("data/manifest.json")
    out.write_text(json.dumps(m, indent=2) + "\n")
    print(f"{out}: dataset_sha256 = {m['dataset_sha256']}")
