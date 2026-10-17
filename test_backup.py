"""test_backup.py -- offline checks for the full-bundle backup / restore.

Stdlib only (unittest); no Ollama / LanceDB / customtkinter needed. Every
function under test takes explicit paths, so the tests build a throwaway
"installation" in a temp dir instead of touching the real one.

Run:  python test_backup.py
"""

import json
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import backup as B
import store


def _make_install(root: Path) -> dict:
    """Create a fake data layout and return its paths."""
    index = root / "index"
    staging = root / "staging"
    backups = root / "backups"
    db = root / "app.db"

    store.init_db(db)
    store.create_project(db, name="Demo", table_name="corpus",
                         embed_model="qwen3-embedding:0.6b")
    store.set_setting(db, "anthropic_key", "sk-secret")
    store.set_setting(db, "top_k", "12")

    (index / "corpus.lance").mkdir(parents=True)
    (index / "corpus.lance" / "data.bin").write_bytes(b"\x00\x01vectors")
    (index / "weights.json").write_text(
        json.dumps({"corpus": {"docs": 1.5}}), encoding="utf-8")

    (staging / "p1").mkdir(parents=True)
    (staging / "p1" / "note.md").write_text("# note\nbody\n", encoding="utf-8")

    return {"index": index, "staging": staging, "backups": backups, "db": db}


def _sqlite_rows(db: Path, sql: str):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


class TestExport(unittest.TestCase):
    def test_bundle_contents_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "bundle.zip"
            m = B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                                staging_dir=p["staging"])
            self.assertTrue(zpath.is_file())
            self.assertEqual(m["kind"], B.BUNDLE_KIND)
            self.assertEqual(m["counts"]["projects"], 1)
            self.assertEqual(m["counts"]["tables"], 1)
            self.assertEqual(m["counts"]["staging_files"], 1)
            with zipfile.ZipFile(zpath) as z:
                names = set(z.namelist())
            self.assertIn("app.db", names)
            self.assertIn("index/corpus.lance/data.bin", names)
            self.assertIn("index/weights.json", names)
            self.assertIn("staging/p1/note.md", names)
            self.assertIn(B.MANIFEST_NAME, names)

    def test_keys_stripped_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "b.zip"
            B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                            staging_dir=p["staging"])
            with tempfile.TemporaryDirectory() as td2:
                with zipfile.ZipFile(zpath) as z:
                    out = Path(td2) / "app.db"
                    out.write_bytes(z.read("app.db"))
                rows = _sqlite_rows(out, "SELECT value FROM settings WHERE key='anthropic_key'")
                self.assertEqual(rows[0][0], "")
                # non-secret settings survive
                rows = _sqlite_rows(out, "SELECT value FROM settings WHERE key='top_k'")
                self.assertEqual(rows[0][0], "12")

    def test_keys_kept_when_opted_in(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "b.zip"
            B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                            staging_dir=p["staging"], include_secrets=True)
            with tempfile.TemporaryDirectory() as td2:
                with zipfile.ZipFile(zpath) as z:
                    out = Path(td2) / "app.db"
                    out.write_bytes(z.read("app.db"))
                rows = _sqlite_rows(out, "SELECT value FROM settings WHERE key='anthropic_key'")
                self.assertEqual(rows[0][0], "sk-secret")

    def test_without_staging(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "b.zip"
            m = B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                                staging_dir=p["staging"], include_staging=False)
            self.assertEqual(m["counts"]["staging_files"], 0)
            with zipfile.ZipFile(zpath) as z:
                self.assertFalse(any(n.startswith("staging/")
                                     for n in z.namelist()))


class TestVerify(unittest.TestCase):
    def test_verify_ok(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "b.zip"
            B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                            staging_dir=p["staging"])
            rep = B.verify_bundle(zpath)
            self.assertTrue(rep["ok"])
            self.assertGreaterEqual(rep["checked"], 4)

    def test_verify_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "b.zip"
            B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                            staging_dir=p["staging"])
            # rebuild the zip with one tampered payload
            bad = root / "bad.zip"
            with zipfile.ZipFile(zpath) as z:
                items = {n: z.read(n) for n in z.namelist()}
            items["staging/p1/note.md"] = b"tampered"
            with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
                for n, data in items.items():
                    z.writestr(n, data)
            rep = B.verify_bundle(bad)
            self.assertFalse(rep["ok"])
            self.assertTrue(any("note.md" in b for b in rep["bad"]))

    def test_not_a_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "x.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("hello.txt", "hi")
            with self.assertRaises(B.BackupError):
                B.read_manifest(z)


class TestImport(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            src.mkdir()
            sp = _make_install(src)
            zpath = Path(td) / "b.zip"
            B.export_bundle(zpath, index_dir=sp["index"], db_path=sp["db"],
                            staging_dir=sp["staging"])

            dst = Path(td) / "dst"
            dp = _make_install(dst)
            # overwrite dst with the bundle
            summary = B.import_bundle(
                zpath, index_dir=dp["index"], db_path=dp["db"],
                staging_dir=dp["staging"], backups_dir=dp["backups"],
                snapshot=True)
            self.assertEqual(summary["kind"], "restore")
            self.assertEqual(summary["projects"], 1)
            self.assertTrue(Path(summary["snapshot"]).is_dir())

            self.assertTrue((dp["index"] / "corpus.lance" / "data.bin").is_file())
            self.assertEqual(
                (dp["index"] / "weights.json").read_text(encoding="utf-8"),
                json.dumps({"corpus": {"docs": 1.5}}))
            self.assertTrue((dp["staging"] / "p1" / "note.md").is_file())
            rows = _sqlite_rows(dp["db"], "SELECT name FROM projects")
            self.assertEqual(rows[0][0], "Demo")

    def test_snapshot_of_current_data_is_taken(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            B.snapshot_current(index_dir=p["index"], db_path=p["db"],
                               staging_dir=p["staging"], backups_dir=p["backups"])
            autos = list(p["backups"].glob("auto-*"))
            self.assertEqual(len(autos), 1)
            self.assertTrue((autos[0] / "app.db").is_file())
            self.assertTrue((autos[0] / "index" / "weights.json").is_file())
            self.assertTrue((autos[0] / "staging" / "p1" / "note.md").is_file())

    def test_import_rejects_corrupt_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            zpath = root / "b.zip"
            B.export_bundle(zpath, index_dir=p["index"], db_path=p["db"],
                            staging_dir=p["staging"])
            bad = root / "bad.zip"
            with zipfile.ZipFile(zpath) as z:
                items = {n: z.read(n) for n in z.namelist()}
            items["index/weights.json"] = b"{}"
            with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
                for n, data in items.items():
                    z.writestr(n, data)
            with self.assertRaises(B.BackupError):
                B.import_bundle(bad, index_dir=p["index"], db_path=p["db"],
                                staging_dir=p["staging"], backups_dir=p["backups"],
                                snapshot=False)

    def test_prune_keeps_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            for i in range(7):
                d = p["backups"] / f"auto-2020010{i}-000000"
                d.mkdir(parents=True)
            B._prune_snapshots(p["backups"], keep=3)
            self.assertEqual(len(list(p["backups"].glob("auto-*"))), 3)


class TestPruneInvalidKeep(unittest.TestCase):
    def test_zero_keep_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _make_install(root)
            d = p["backups"] / "auto-20200101-000000"
            d.mkdir(parents=True)
            B._prune_snapshots(p["backups"], keep=0)
            self.assertTrue(d.is_dir())


class TestGuiWiring(unittest.TestCase):
    """tk_app needs customtkinter + a display; check the wiring by AST."""

    @classmethod
    def setUpClass(cls):
        import ast
        src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        cls.src = src
        cls.tree = ast.parse(src)
        cls.fns = {n.name for n in ast.walk(cls.tree)
                   if isinstance(n, ast.FunctionDef)}

    def test_methods_exist(self):
        for name in ("do_backup_zip", "do_restore_zip", "do_open_data_folder",
                     "_after_restore", "_include_keys"):
            self.assertIn(name, self.fns)

    def test_calls_backup_module(self):
        self.assertIn("import backup", self.src)
        for call in ("backup.export_bundle", "backup.import_bundle",
                     "backup.read_manifest", "backup.BUNDLE_KIND"):
            self.assertIn(call, self.src)

    def test_busy_locks_backup_buttons(self):
        self.assertIn("btn_backup_zip", self.src)
        self.assertIn("btn_restore_zip", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
