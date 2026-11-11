from pathlib import Path

from src.har.manifest import crea_manifest


def test_hash_stabile_e_sensibile(tmp_path: Path):
    (tmp_path / "a.txt").write_text("uno")
    (tmp_path / "b.txt").write_text("due")
    h1 = crea_manifest(tmp_path)["dataset_sha256"]
    assert crea_manifest(tmp_path)["dataset_sha256"] == h1  # deterministico

    (tmp_path / "b.txt").write_text("due!")  # basta un byte diverso
    assert crea_manifest(tmp_path)["dataset_sha256"] != h1


def test_rinomina_cambia_hash(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    h1 = crea_manifest(tmp_path)["dataset_sha256"]
    (tmp_path / "a.txt").rename(tmp_path / "c.txt")
    assert crea_manifest(tmp_path)["dataset_sha256"] != h1
