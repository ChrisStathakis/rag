"""test_index.py -- offline checks for the vector-index maintenance layer.

Covers `vector_search` (metric + ANN knobs), `_index_health` / `_fmt_indices`
reporting, `_maybe_compact` thresholding, the `index` CLI wiring, and a real
(but throwaway) LanceDB round-trip that builds FTS + BTREE indexes and an ANN
index in a temp dir. Nothing touches the real `index/`.

Run:  python test_index.py
"""

import contextlib
import io
import pathlib
import re
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import pipeline as P

try:
    import lancedb  # noqa: F401
    import pyarrow as pa  # noqa: F401
    HAVE_LANCE = True
except Exception:                                        # noqa: BLE001
    HAVE_LANCE = False


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

class _QB:
    """Query-builder stub that records the chain and returns rows."""

    def __init__(self, log, rows=None, fail_on=()):
        self.log = log
        self.rows = rows if rows is not None else [{"id": "x"}]
        self.fail_on = fail_on

    def metric(self, m):
        self.log.append(("metric", m))
        return self

    def nprobes(self, n):
        if "nprobes" in self.fail_on:
            raise RuntimeError("no index")
        self.log.append(("nprobes", n))
        return self

    def refine_factor(self, n):
        if "refine" in self.fail_on:
            raise RuntimeError("no index")
        self.log.append(("refine_factor", n))
        return self

    def limit(self, k):
        self.log.append(("limit", k))
        return self

    def to_list(self):
        return self.rows


class _SearchTable:
    def __init__(self, rows=None, fail_on=()):
        self.log = []
        self.rows = rows
        self.fail_on = fail_on

    def search(self, vec, **kw):
        self.log.append(("search", kw.get("vector_column_name")))
        return _QB(self.log, self.rows, self.fail_on)


class _Lance:
    def __init__(self, n):
        self._n = n

    def get_fragments(self):
        return list(range(self._n))


class _Idx:
    def __init__(self, name, index_type, columns):
        self.name = name
        self.index_type = index_type
        self.columns = columns


class _HealthTable:
    def __init__(self, rows=10, frags=3, indices=()):
        self.rows = rows
        self.frags = frags
        self.indices = list(indices)

    def count_rows(self, *a, **kw):
        return self.rows

    def to_lance(self):
        return _Lance(self.frags)

    def list_indices(self):
        return list(self.indices)


class _NS:
    """argparse.Namespace stand-in with cmd_index's defaults."""

    def __init__(self, **kw):
        base = {"table": None, "ann": False, "type": "IVF_PQ",
                "sub_vectors": 0, "force": False, "no_compact": False,
                "no_fts": False, "no_scalar": False}
        base.update(kw)
        self.__dict__.update(base)


# --------------------------------------------------------------------------- #
# vector_search
# --------------------------------------------------------------------------- #

class TestVectorSearch(unittest.TestCase):
    def test_metric_is_always_cosine_and_limit_applied(self):
        t = _SearchTable()
        P.vector_search(t, [0.1], 7)
        self.assertIn(("metric", "cosine"), t.log)
        self.assertIn(("limit", 7), t.log)
        self.assertNotIn(("nprobes", 0), t.log)

    def test_knobs_omitted_when_zero(self):
        t = _SearchTable()
        P.vector_search(t, [0.1], 5, nprobes=0, refine=0)
        kinds = [k for k, *_ in t.log]
        self.assertNotIn("nprobes", kinds)
        self.assertNotIn("refine_factor", kinds)

    def test_knobs_applied_when_positive(self):
        t = _SearchTable()
        P.vector_search(t, [0.1], 5, nprobes=12, refine=3)
        self.assertIn(("nprobes", 12), t.log)
        self.assertIn(("refine_factor", 3), t.log)

    def test_unindexed_table_does_not_break_search(self):
        # LanceDB may reject the knobs on tables without an index; the helper
        # must swallow that and still return rows.
        t = _SearchTable(rows=[{"id": "a"}, {"id": "b"}],
                         fail_on=("nprobes", "refine"))
        out = P.vector_search(t, [0.1], 5, nprobes=9, refine=9)
        self.assertEqual([r["id"] for r in out], ["a", "b"])


# --------------------------------------------------------------------------- #
# health reporting / compaction
# --------------------------------------------------------------------------- #

class TestHealth(unittest.TestCase):
    def test_health_reads_rows_fragments_indices(self):
        t = _HealthTable(rows=42, frags=5, indices=[
            _Idx("text_raw_idx", "FTS", ["text_raw"]),
            _Idx("vector_idx", "IvfPq", ["vector"]),
        ])
        h = P._index_health(t)
        self.assertEqual(h["rows"], 42)
        self.assertEqual(h["fragments"], 5)
        self.assertEqual([i["type"] for i in h["indices"]], ["FTS", "IvfPq"])

    def test_health_survives_broken_table(self):
        class Broken:
            def count_rows(self, *a, **kw):
                raise RuntimeError("boom")

            def to_lance(self):
                raise RuntimeError("boom")

            def list_indices(self):
                raise RuntimeError("boom")
        h = P._index_health(Broken())
        self.assertIsNone(h["rows"])
        self.assertEqual(h["indices"], [])

    def test_fmt_indices(self):
        self.assertEqual(P._fmt_indices({"indices": []}), "none")
        self.assertEqual(
            P._fmt_indices({"indices": [
                {"type": "IvfPq", "columns": ["vector"]},
                {"type": "FTS", "columns": ["text_raw"]}]}),
            "IvfPq(vector), FTS(text_raw)")

    def test_maybe_compact_respects_threshold(self):
        calls: list = []
        orig = P._optimize_table
        P._optimize_table = lambda t: (calls.append(t), True)[1]
        self.addCleanup(lambda: setattr(P, "_optimize_table", orig))

        with contextlib.redirect_stdout(io.StringIO()):
            P._maybe_compact(_HealthTable(frags=P.COMPACT_FRAGMENT_THRESHOLD),
                             "t")
        self.assertEqual(calls, [], "at the threshold it must not compact")

        with contextlib.redirect_stdout(io.StringIO()):
            P._maybe_compact(
                _HealthTable(frags=P.COMPACT_FRAGMENT_THRESHOLD + 1), "t")
        self.assertEqual(len(calls), 1)


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #

class TestCli(unittest.TestCase):
    def _parse(self, *argv):
        return P.build_parser().parse_args(list(argv))

    def test_index_subcommand_defaults(self):
        a = self._parse("index")
        self.assertFalse(a.ann)
        self.assertEqual(a.type, "IVF_PQ")
        self.assertFalse(a.force)
        self.assertFalse(a.no_compact)
        self.assertFalse(a.no_fts)
        self.assertFalse(a.no_scalar)
        self.assertIsNone(a.table)

    def test_index_flags(self):
        a = self._parse("index", "--ann", "--force", "--sub-vectors", "8",
                        "--type", "IVF_HNSW_SQ", "--no-compact")
        self.assertTrue(a.ann)
        self.assertTrue(a.force)
        self.assertEqual(a.sub_vectors, 8)
        self.assertEqual(a.type, "IVF_HNSW_SQ")
        self.assertTrue(a.no_compact)

    def test_query_ann_knobs(self):
        a = self._parse("query", "hello")
        self.assertEqual(a.nprobes, 0)
        self.assertEqual(a.refine, 0)
        a = self._parse("query", "hello", "--nprobes", "20", "--refine", "5")
        self.assertEqual(a.nprobes, 20)
        self.assertEqual(a.refine, 5)


# --------------------------------------------------------------------------- #
# GUI + store glue (AST / source level, no Tk needed)
# --------------------------------------------------------------------------- #

class TestGuiAndStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        cls.flat = re.sub(r"\s+", "", cls.src)

    def test_settings_has_index_controls(self):
        self.assertIn("def do_index_health", self.src)
        self.assertIn("def _index_job", self.src)
        self.assertIn("def _index_tuning", self.src)
        self.assertIn("btn_index", self.flat)
        self.assertIn("ann_var", self.flat)
        self.assertIn('"vector_nprobes"', self.src)
        self.assertIn('"vector_refine_factor"', self.src)
        self.assertIn('"index"inresult', self.flat)     # _on_done branch

    def test_retrieve_uses_vector_search(self):
        self.assertIn("P.vector_search", self.src)

    def test_tuning_live_prefers_widgets(self):
        self.assertIn("def _tuning_live", self.src)

    def test_store_defaults(self):
        import store
        self.assertIn("vector_nprobes", store._SETTINGS_DEFAULTS)
        self.assertIn("vector_refine_factor", store._SETTINGS_DEFAULTS)


class TestTuningLive(unittest.TestCase):
    """The ANN knobs must apply from the boxes as typed (no Save needed).

    Needs a real window; skip cleanly when there is no display (CI).
    """

    def _app(self):
        try:
            import tk_app
        except Exception as e:                             # noqa: BLE001
            self.skipTest(f"GUI not available: {e}")
        tk_app.DB_PATH = pathlib.Path(
            tempfile.mkdtemp(prefix="ep-tuning-")) / "app.db"
        try:
            app = tk_app.App()
        except Exception as e:                             # noqa: BLE001
            self.skipTest(f"Tk not available: {e}")
        app.update()
        self.addCleanup(app.destroy)
        return app

    def test_reads_the_boxes_as_typed(self):
        app = self._app()
        app.sv["vector_nprobes"].set("24")
        app.sv["vector_refine_factor"].set("6")
        self.assertEqual(app._tuning_live(), (24, 6))

    def test_empty_or_junk_falls_back_to_zero(self):
        app = self._app()
        app.sv["vector_nprobes"].set("")
        app.sv["vector_refine_factor"].set("junk")
        self.assertEqual(app._tuning_live(), (0, 0))


# --------------------------------------------------------------------------- #
# real LanceDB round-trip in a temp dir (never the real index/)
# --------------------------------------------------------------------------- #

@unittest.skipUnless(HAVE_LANCE, "lancedb not installed")
class TestIndexIntegration(unittest.TestCase):
    def setUp(self):
        import lancedb
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="ep-test-index-"))
        self.old = P.INDEX_DIR
        P.INDEX_DIR = self.tmp
        self.addCleanup(lambda: setattr(P, "INDEX_DIR", self.old))

        self.dim = 16
        rows = []
        for i in range(600):
            rows.append({
                "id": str(i),
                "text_raw": f"chunk {i} about attention and retrieval",
                "embed_input": f"chunk {i}",
                "source": f"lab/f{i % 4}.md",
                "vector": [float((i + j) % 9) / 9.0 for j in range(self.dim)],
            })
        self.db = lancedb.connect(str(self.tmp))
        self.db.create_table("t", data=rows, schema=pa.schema([
            ("id", pa.string()),
            ("text_raw", pa.string()),
            ("embed_input", pa.string()),
            ("source", pa.string()),
            ("vector", pa.list_(pa.float32(), self.dim)),
        ]))

    def _run(self, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            res = P.cmd_index(_NS(**kw))
        return res, buf.getvalue()

    def _types(self):
        return {i.index_type for i in self.db.open_table("t").list_indices()}

    def test_fts_scalar_and_compaction(self):
        res, out = self._run(table="t")
        types = self._types()
        self.assertIn("FTS", types)
        self.assertIn("BTree", types)
        self.assertNotIn("IvfPq", types)                 # opt-in only
        self.assertIn("vector index: not requested", out)
        self.assertEqual(res["index"][0]["table"], "t")

    def test_ann_build_and_search(self):
        self._run(table="t", ann=True, force=True, sub_vectors=8)
        self.assertIn("IvfPq", self._types())
        hits = P.vector_search(self.db.open_table("t"),
                               [0.05] * self.dim, 5, nprobes=10, refine=5)
        self.assertEqual(len(hits), 5)

    def test_ann_skipped_on_small_table_without_force(self):
        _res, out = self._run(table="t", ann=True)       # no --force
        self.assertNotIn("IvfPq", self._types())
        self.assertIn("ANN skipped", out)

    def test_summary_reports_fragments_and_indices(self):
        res, _out = self._run(table="t")
        s = res["index"][0]
        self.assertIn("FTS", s["indices"])
        self.assertEqual(s["rows"], 600)
        self.assertIsNotNone(s["before_fragments"])
        self.assertIsNotNone(s["after_fragments"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
