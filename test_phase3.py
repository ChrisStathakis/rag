"""test_phase3.py -- offline checks for the Phase-3 operational features.

Covers:
  * cooperative cancellation (`request_stop` / `clear_stop` / `check_stop`)
  * `delete_sources` (exact file and folder-prefix removal)
  * `tk_app._list_sources` (distinct sources + counts)
  * GUI wiring for Sync / Sources… / Stop (checked by AST)

The functional tests use a throwaway index dir built in a temp folder; they
need LanceDB (installed via requirements.txt) but no Ollama and no display.
The AST tests always run.

Run:  python test_phase3.py
"""

import ast
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import backup as B
import pipeline as P
import store

try:
    import lancedb  # noqa: F401
    HAVE_LANCE = True
except Exception:                                        # noqa: BLE001
    HAVE_LANCE = False


def _write_sources(root: Path) -> None:
    # label "docs" is added on top of the path relative to `root`, so these
    # two files become `docs/a.md` and `docs/sub/b.md` in the index.
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "a.md").write_text(
        "# Alpha\n\nalpha body text about apples and oranges.\n",
        encoding="utf-8")
    (root / "sub" / "b.md").write_text(
        "# Beta\n\nbeta body text about bananas.\n", encoding="utf-8")


@unittest.skipUnless(HAVE_LANCE, "lancedb not installed")
class TestStop(unittest.TestCase):
    def setUp(self):
        self._orig = P.INDEX_DIR
        self._td = tempfile.TemporaryDirectory()
        P.INDEX_DIR = Path(self._td.name) / "index"
        P.clear_stop()

    def tearDown(self):
        P.clear_stop()
        P.INDEX_DIR = self._orig
        self._td.cleanup()

    def test_check_stop_raises_only_when_requested(self):
        P.check_stop()                                  # no-op by default
        P.request_stop()
        with self.assertRaises(P.PipelineError) as ctx:
            P.check_stop()
        self.assertEqual(str(ctx.exception), P.STOP_MESSAGE)

    def test_clear_stop_resets_the_flag(self):
        P.request_stop()
        P.clear_stop()
        P.check_stop()                                  # must not raise

    def test_ingest_aborts_when_stopped(self):
        src = Path(self._td.name) / "src"
        _write_sources(src)
        spec = P.SourceSpec(path=str(src), label="docs")
        P.request_stop()
        with self.assertRaises(P.PipelineError) as ctx:
            P.ingest_sources([spec], table="corpus", model="m", dims=8)
        self.assertEqual(str(ctx.exception), P.STOP_MESSAGE)

    def test_ingest_runs_when_not_stopped(self):
        src = Path(self._td.name) / "src"
        _write_sources(src)
        spec = P.SourceSpec(path=str(src), label="docs")
        summary = P.ingest_sources([spec], table="corpus", model="m", dims=8)
        self.assertGreater(summary["chunks"], 0)


@unittest.skipUnless(HAVE_LANCE, "lancedb not installed")
class TestDeleteSources(unittest.TestCase):
    def setUp(self):
        self._orig = P.INDEX_DIR
        self._td = tempfile.TemporaryDirectory()
        P.INDEX_DIR = Path(self._td.name) / "index"
        src = Path(self._td.name) / "src"
        _write_sources(src)
        P.clear_stop()
        P.ingest_sources([P.SourceSpec(path=str(src), label="docs")],
                         table="corpus", model="m", dims=8)

    def tearDown(self):
        P.clear_stop()
        P.INDEX_DIR = self._orig
        self._td.cleanup()

    def _sources(self) -> set:
        rows = P._open_table("corpus").search().select(["source"]).to_list()
        return {str(r.get("source")) for r in rows}

    def test_delete_one_file(self):
        self.assertIn("docs/a.md", self._sources())
        res = P.delete_sources("corpus", ["docs/a.md"])
        self.assertGreater(res["deleted"], 0)
        self.assertNotIn("docs/a.md", self._sources())
        self.assertIn("docs/sub/b.md", self._sources())

    def test_delete_folder_prefix(self):
        res = P.delete_sources("corpus", ["docs"])
        self.assertGreater(res["deleted"], 0)
        self.assertEqual(self._sources(), set())
        self.assertEqual(res["after"], 0)

    def test_delete_unknown_table_raises(self):
        with self.assertRaises(P.PipelineError):
            P.delete_sources("nope", ["docs/a.md"])


class TestListSources(unittest.TestCase):
    """`_list_sources` lives in tk_app; import it lazily (needs customtkinter)."""

    @unittest.skipUnless(HAVE_LANCE, "lancedb not installed")
    def test_counts_per_source(self):
        try:
            import tk_app
        except Exception as e:                           # noqa: BLE001
            self.skipTest(f"customtkinter unavailable: {e}")
        orig = P.INDEX_DIR
        with tempfile.TemporaryDirectory() as td:
            P.INDEX_DIR = Path(td) / "index"
            try:
                src = Path(td) / "src"
                _write_sources(src)
                P.ingest_sources([P.SourceSpec(path=str(src), label="docs")],
                                 table="corpus", model="m", dims=8)
                got = tk_app._list_sources("corpus")
                names = [s for s, _ in got]
                self.assertEqual(names, ["docs/a.md", "docs/sub/b.md"])
                self.assertTrue(all(c > 0 for _, c in got))
            finally:
                P.INDEX_DIR = orig

    def test_missing_table_raises(self):
        try:
            import tk_app
        except Exception as e:                           # noqa: BLE001
            self.skipTest(f"customtkinter unavailable: {e}")
        if not HAVE_LANCE:
            self.skipTest("lancedb not installed")
        orig = P.INDEX_DIR
        with tempfile.TemporaryDirectory() as td:
            P.INDEX_DIR = Path(td) / "index"
            try:
                with self.assertRaises(P.PipelineError):
                    tk_app._list_sources("missing")
            finally:
                P.INDEX_DIR = orig


class TestProjectPackage(unittest.TestCase):
    """Per-project export/import: append semantics, no clobbering."""

    def _make_install(self, root: Path, *, name: str, table: str,
                      staging: bool = True) -> dict:
        index, staging_dir, db = root / "index", root / "staging", root / "app.db"
        store.init_db(db)
        pid = store.create_project(
            db, name=name, table_name=table, embed_model="test-embed",
            chat_model="test-chat", chunk_size=900, overlap=120,
            folders=[{"path": str(root / "docs"), "only": ".md", "weight": 2.0}])
        (index / f"{table}.lance").mkdir(parents=True, exist_ok=True)
        (index / f"{table}.lance" / "data.bin").write_bytes(
            b"\x00vectors-" + table.encode())
        wf = index / "weights.json"
        cur = json.loads(wf.read_text(encoding="utf-8")) if wf.is_file() else {}
        cur[table] = {"docs": 1.5}
        wf.write_text(json.dumps(cur), encoding="utf-8")
        if staging:
            (staging_dir / f"p{pid}").mkdir(parents=True, exist_ok=True)
            (staging_dir / f"p{pid}" / "note.md").write_text(
                "# note\nbody\n", encoding="utf-8")
        return {"index": index, "staging": staging_dir, "db": db, "pid": pid}

    def _export(self, root: Path, src: dict, fname: str = "pkg.zip"):
        z = root / fname
        m = B.export_project(z, pid=src["pid"], index_dir=src["index"],
                             db_path=src["db"], staging_dir=src["staging"])
        return z, m

    def test_export_manifest_and_verify(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = self._make_install(root / "src", name="Demo", table="corpus")
            z, m = self._export(root, src)
            self.assertEqual(m["kind"], B.PROJECT_KIND)
            self.assertEqual(m["project"]["name"], "Demo")
            self.assertEqual(m["project"]["table"], "corpus")
            self.assertTrue(m["has_table"])
            self.assertEqual(m["counts"]["staging_files"], 1)
            rep = B.verify_project(z)
            self.assertTrue(rep["ok"], rep["bad"])

    def test_verify_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = self._make_install(root / "src", name="Demo", table="corpus")
            z, _ = self._export(root, src)
            # rewrite the zip with a tampered payload
            with zipfile.ZipFile(z) as zz:
                items = {n: zz.read(n) for n in zz.namelist()}
            items["index/corpus.lance/data.bin"] = b"tampered"
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
                for n, data in items.items():
                    zz.writestr(n, data)
            self.assertFalse(B.verify_project(z)["ok"])

    def test_wrong_kinds_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # a project package is not a full bundle, and vice versa
            src = self._make_install(root / "src", name="Demo", table="corpus")
            z, _ = self._export(root, src)
            with self.assertRaises(B.BackupError):
                B.verify_bundle(z)
            bundle = root / "full.zip"
            B.export_bundle(bundle, index_dir=src["index"], db_path=src["db"],
                            staging_dir=src["staging"])
            with self.assertRaises(B.BackupError):
                B.verify_project(bundle)

    def test_import_appends_and_keeps_other_tables(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = self._make_install(root / "src", name="Demo", table="corpus")
            z, _ = self._export(root, src)

            dst = self._make_install(root / "dst", name="Other", table="other")
            summary = B.import_project(z, index_dir=dst["index"],
                                       db_path=dst["db"],
                                       staging_dir=dst["staging"],
                                       backups_dir=root / "backups")
            new_pid = summary["pid"]
            self.assertEqual(summary["name"], "Demo")
            self.assertEqual(summary["table"], "corpus")
            self.assertFalse(summary["table_renamed"])
            self.assertTrue(summary["snapshot"])

            # both projects present, other table untouched
            names = {p["name"] for p in store.list_projects(dst["db"])}
            self.assertEqual(names, {"Other", "Demo"})
            self.assertTrue((dst["index"] / "other.lance").is_dir())
            self.assertTrue((dst["index"] / "corpus.lance").is_dir())

            # weights merged, not replaced
            w = json.loads((dst["index"] / "weights.json").read_text("utf-8"))
            self.assertIn("other", w)
            self.assertEqual(w["corpus"], {"docs": 1.5})

            # staging copied under the new pid
            self.assertTrue(
                (dst["staging"] / f"p{new_pid}" / "note.md").is_file())

            # second import: both name and table clash -> suffixed
            s2 = B.import_project(z, index_dir=dst["index"], db_path=dst["db"],
                                  staging_dir=dst["staging"],
                                  backups_dir=root / "backups")
            self.assertEqual(s2["name"], "Demo (imported)")
            self.assertTrue(s2["table_renamed"])
            self.assertTrue((dst["index"] / f"{s2['table']}.lance").is_dir())

    def test_snapshot_taken_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = self._make_install(root / "src", name="Demo", table="corpus")
            z, _ = self._export(root, src)
            dst = self._make_install(root / "dst", name="Other", table="other")
            backups = root / "backups"
            B.import_project(z, index_dir=dst["index"], db_path=dst["db"],
                             staging_dir=dst["staging"], backups_dir=backups)
            autos = [d for d in backups.iterdir() if d.name.startswith("auto-")]
            self.assertEqual(len(autos), 1)
            self.assertTrue((autos[0] / "app.db").is_file())


class TestSavedQueryStore(unittest.TestCase):
    def test_dedupe_list_and_delete(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "app.db"
            store.init_db(db)
            pid = store.create_project(db, name="P", table_name="corpus")
            a = store.add_saved_query(db, pid, "  hello world  ")
            b = store.add_saved_query(db, pid, "hello world")   # dedupe
            self.assertEqual(a, b)
            self.assertEqual(store.list_saved_queries(db, pid)[0]["query"],
                             "hello world")
            with self.assertRaises(ValueError):
                store.add_saved_query(db, pid, "   ")
            store.delete_saved_query(db, a)
            self.assertEqual(store.list_saved_queries(db, pid), [])

    def test_queries_are_per_project(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "app.db"
            store.init_db(db)
            p1 = store.create_project(db, name="One", table_name="t1")
            p2 = store.create_project(db, name="Two", table_name="t2")
            store.add_saved_query(db, p1, "only one")
            self.assertEqual(len(store.list_saved_queries(db, p1)), 1)
            self.assertEqual(store.list_saved_queries(db, p2), [])


class TestGuiWiring(unittest.TestCase):
    """tk_app needs a display to run; verify the wiring from the source."""

    @classmethod
    def setUpClass(cls):
        cls.src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)
        cls.fns = {n.name for n in ast.walk(cls.tree)
                   if isinstance(n, ast.FunctionDef)}

    def test_gui_methods_exist(self):
        for name in ("do_sync", "do_stop", "do_manage_sources",
                     "_sync_job", "_delete_sources_job", "_list_sources",
                     "do_save_query", "do_delete_saved_query",
                     "_refresh_saved_queries", "_on_saved_picked",
                     "do_export_project", "do_import_project",
                     "_after_project_import", "_export_project_job",
                     "_import_project_job"):
            self.assertIn(name, self.fns)

    def test_gui_widgets_wired(self):
        for name in ("btn_stop", "btn_sync", "btn_manage_sources",
                     "saved_menu", "btn_save_query", "btn_del_query",
                     "btn_export_project", "btn_import_project"):
            self.assertIn(name, self.src)

    def test_gui_calls_new_pipeline_api(self):
        for call in ("P.request_stop", "P.clear_stop", "P.delete_sources",
                     "store.list_saved_queries", "store.add_saved_query",
                     "store.delete_saved_query", "backup.export_project",
                     "backup.import_project", "backup.PROJECT_KIND"):
            self.assertIn(call, self.src)

    def test_stop_is_enabled_only_while_busy(self):
        # the busy-toggle must treat Stop inversely (enabled when busy)
        self.assertIn('stop.configure(state="normal" if busy else "disabled")',
                      self.src)


class TestPipelineWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (HERE / "pipeline.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)

    def test_check_stop_called_in_hot_loops(self):
        # ingest + export file loops and the embed batch loop each guard
        self.assertGreaterEqual(self.src.count("check_stop()"), 3)

    def test_cli_delete_subcommand(self):
        self.assertIn('add_parser("delete"', self.src)
        self.assertIn("cmd_delete", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
