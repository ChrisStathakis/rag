"""test_hybrid.py -- offline checks for the hybrid-search option.

Stdlib only (unittest). No Ollama / LanceDB / customtkinter needed:
  - pipeline.py top-level imports are stdlib-only (heavy deps are lazy),
  - tk_app.py is verified statically via ast (customtkinter is not installed
    in every environment).
Run:  python test_hybrid.py
"""

import argparse
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import pipeline as P


def _hit(rid, **kw):
    h = {"id": rid, "source": "docs/a.md", "title": "t",
         "section": "s", "text_raw": "body"}
    h.update(kw)
    return h


class TestRRF(unittest.TestCase):
    def test_overlap_fused_first_and_scored(self):
        vec = [_hit("a", _distance=0.2), _hit("b", _distance=0.4)]
        fts = [_hit("b", _score=5.0), _hit("c", _score=1.0)]
        out = P._rrf([vec, fts], 3)
        self.assertEqual([r["id"] for r in out], ["b", "a", "c"])
        self.assertTrue(all("_rrf" in r for r in out))

    def test_vector_distance_preserved_through_fts_overwrite(self):
        vec = [_hit("a", _distance=0.25)]
        fts = [_hit("a", _score=9.0)]
        out = P._rrf([vec, fts], 1)
        self.assertEqual(out[0].get("_distance"), 0.25)
        self.assertEqual(out[0].get("_score"), 9.0)

    def test_empty_fts_list_is_vector_order(self):
        vec = [_hit("a", _distance=0.1), _hit("b", _distance=0.3)]
        out = P._rrf([vec, []], 2)
        self.assertEqual([r["id"] for r in out], ["a", "b"])

    def test_k_truncates(self):
        vec = [_hit(str(i), _distance=0.1 * i) for i in range(5)]
        out = P._rrf([vec], 2)
        self.assertEqual(len(out), 2)


class TestWeights(unittest.TestCase):
    def test_boost_reorders(self):
        hits = [_hit("a", source="docs/a.md", _distance=0.5),   # cos 0.5
                _hit("b", source="code/b.py", _distance=0.2)]  # cos 0.8
        out = P.apply_weights(hits, {"docs": 2.0})
        self.assertEqual(out[0]["id"], "a")       # 0.5*2=1.0 wins
        self.assertEqual(out[0].get("_weight"), 2.0)

    def test_all_one_is_passthrough(self):
        hits = [_hit("a", _distance=0.5)]
        out = P.apply_weights(hits, {"docs": 1.0})
        self.assertEqual(out[0].get("_distance"), 0.5)
        self.assertNotIn("_weight", out[0])

    def test_missing_label_untouched(self):
        hits = [_hit("a", source="other/a.md", _distance=0.5)]
        out = P.apply_weights(hits, {"docs": 2.0})
        self.assertEqual(out[0].get("_distance"), 0.5)

    def test_fts_row_without_distance_passes_through(self):
        hits = [_hit("a", source="docs/a.md")]
        out = P.apply_weights(hits, {"docs": 2.0})
        self.assertNotIn("_distance", out[0])


class TestFTSFallback(unittest.TestCase):
    def test_broken_fts_returns_empty(self):
        class BadTable:
            def search(self, *a, **k):
                raise RuntimeError("no fts here")
        self.assertEqual(P._fts_search(BadTable(), "q", 5), [])

    def test_hybrid_branch_skips_fts_when_off(self):
        # mirror of the tk_app._retrieve branch: hybrid=False must never
        # call the FTS half.
        calls = []

        def fake_fts(table, text, k):
            calls.append(text)
            return [_hit("x")]

        vec = [_hit("a", _distance=0.1), _hit("b", _distance=0.2)]
        hybrid = False
        if hybrid:
            hits = P._rrf([vec, fake_fts(None, "q", 6)], 2)
        else:
            hits = vec[:2]
        self.assertEqual(calls, [])
        self.assertEqual([h["id"] for h in hits], ["a", "b"])


class TestCLI(unittest.TestCase):
    def _parse(self, *argv):
        return P.build_parser().parse_args(list(argv))

    def test_query_defaults_to_hybrid(self):
        self.assertTrue(self._parse("query", "hello").hybrid)

    def test_no_hybrid_disables(self):
        self.assertFalse(self._parse("query", "hello", "--no-hybrid").hybrid)

    def test_explicit_hybrid_flag(self):
        self.assertTrue(self._parse("query", "hello", "--hybrid").hybrid)

    def test_eval_defaults_to_vector(self):
        self.assertFalse(self._parse("eval", "--queries", "q.txt").hybrid)

    def test_eval_hybrid_opt_in(self):
        self.assertTrue(
            self._parse("eval", "--queries", "q.txt", "--hybrid").hybrid)


class TestGUIStatic(unittest.TestCase):
    """tk_app needs customtkinter + a display; check the wiring by AST."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((HERE / "tk_app.py").read_text(encoding="utf-8"))
        cls.fns = {n.name: n for n in ast.walk(cls.tree)
                   if isinstance(n, ast.FunctionDef)}

    def _arg_names(self, fn):
        return [a.arg for a in fn.args.args] + [
            a.arg for a in fn.args.kwonlyargs]

    def test_retrieve_has_hybrid_default_true(self):
        fn = self.fns["_retrieve"]
        self.assertIn("hybrid", self._arg_names(fn))
        d = fn.args.defaults[-1]
        self.assertTrue(isinstance(d, ast.Constant) and d.value is True)

    def test_search_and_chat_accept_hybrid(self):
        self.assertIn("hybrid", self._arg_names(self.fns["_search"]))
        self.assertIn("hybrid", self._arg_names(self.fns["_chat"]))

    def test_retrieve_branches_on_hybrid(self):
        src = ast.get_source_segment(
            (HERE / "tk_app.py").read_text(encoding="utf-8"),
            self.fns["_retrieve"])
        self.assertIn("if hybrid:", src)
        self.assertIn("_fts_search", src)
        self.assertIn("hits[:k]", src)

    def test_callers_pass_hybrid(self):
        src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        self.assertIn("hybrid=bool(self.hybrid_var.get())", src)

    def test_checkboxes_exist(self):
        src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        self.assertIn("self.hybrid_var = ctk.BooleanVar(value=True)", src)
        self.assertIn('var("hybrid_search", "1")', src)
        self.assertIn("_on_hybrid_setting", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
