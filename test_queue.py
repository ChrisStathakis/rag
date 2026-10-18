"""test_queue.py -- job-queue behaviour of the GUI Runner.

`Runner` used to refuse a second job ("Another step is still running"), which
froze the window for the whole of a long embed. It now queues jobs: you can
switch projects, set up the next step, and the job starts by itself when the
current one finishes. These tests pin that down -- including that the queue is
strictly serial (only one worker, because pipeline.log/set_progress/stop are
module-level globals) and that a result arriving after a project switch is
logged, not painted into the wrong project's view.

No Tk needed for the Runner tests (a fake app stands in); the GUI-level tests
skip cleanly when there is no display.

Run:  python test_queue.py
"""

import ast
import pathlib
import queue
import re
import sys
import threading
import time
import unittest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import tk_app
from tk_app import BUS, Runner


# --------------------------------------------------------------------------- #
# a minimal App stand-in (only what Runner touches)
# --------------------------------------------------------------------------- #

class _Var:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class _Btn:
    def __init__(self) -> None:
        self.text = "Stop"
        self.state = "disabled"

    def configure(self, **kw) -> None:
        for key, value in kw.items():
            setattr(self, key, value)


class FakeApp:
    """Just enough `App` surface for Runner + the Stop-button badge."""

    def __init__(self, pid: int | None = 1, table: str = "t") -> None:
        self.project_id = pid
        self.table_var = _Var(table)
        self.btn_embed = _Btn()
        self.btn_stop = _Btn()
        self.busy_calls: list = []
        self.progress_calls: list = []
        self.said: list = []
        self._last_job = ""
        self._job_ctx: dict = {}
        self.runner: Runner | None = None

    def _say(self, msg: str) -> None:
        self.said.append(msg)

    def set_busy(self, busy: bool) -> None:
        self.busy_calls.append(busy)
        self.btn_stop.state = "normal" if busy else "disabled"
        self._update_queue_badge()

    def set_progress(self, done: int, total: int, title: str = "") -> None:
        self.progress_calls.append((done, total, title))

    def _update_queue_badge(self) -> None:
        n = self.runner.pending if self.runner else 0
        self.btn_stop.text = f"Stop ({n})" if n else "Stop"


# --------------------------------------------------------------------------- #
# Runner queue
# --------------------------------------------------------------------------- #

class TestRunnerQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FakeApp()
        self.app.runner = Runner(self.app)
        # drain anything a previous test left on the bus
        self._drain()

    def tearDown(self) -> None:
        self._drain()

    def _drain(self) -> list:
        """Consume BUS, feeding 'finished' back to the runner like App does."""
        seen = []
        try:
            while True:
                kind, payload = BUS.get_nowait()
                seen.append((kind, payload))
                if kind == "finished":
                    self.app.runner.finish()
        except queue.Empty:
            pass
        return seen

    def _settle(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self._drain()
            if not self.app.runner.busy and not self.app.runner.pending:
                return True
            time.sleep(0.005)
        return False

    def _wait(self, cond, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.005)
        return False

    # -- basics ---------------------------------------------------------- #

    def test_idle_run_starts_immediately(self):
        ran = []
        self.app.runner.run("a", lambda: ran.append("a") or {"ok": 1})
        self.assertTrue(self.app.runner.busy)
        self.assertEqual(self.app.runner.current, "a")
        self.assertEqual(self.app.runner.pending, 0)
        self.assertTrue(self._settle())
        self.assertEqual(ran, ["a"])
        self.assertFalse(self.app.runner.busy)
        self.assertEqual(self.app.busy_calls, [True, False])

    def test_second_job_is_queued_not_refused(self):
        gate = threading.Event()
        order = []

        def blocked():
            order.append("a")
            gate.wait(5)
            return {}

        self.app.runner.run("a", blocked)
        self.assertTrue(self._wait(lambda: order == ["a"]))
        self.app.runner.run("b", lambda: order.append("b") or {})

        # b waits, and the UI was told so -- no warning dialog, no refusal
        self.assertEqual(order, ["a"])
        self.assertEqual(self.app.runner.pending, 1)
        self.assertIn("Stop (1)", self.app.btn_stop.text)
        self.assertTrue(any("queued: b" in m for m in self.app.said))

        gate.set()
        self.assertTrue(self._settle())
        self.assertEqual(order, ["a", "b"])

    def test_queue_drains_in_order_and_unlocks_once(self):
        order = []
        gate = threading.Event()

        def job(tag):
            def _run():
                order.append(tag)
                if tag == "a":
                    gate.wait(5)
                return {}
            return _run

        self.app.runner.run("a", job("a"))
        self.assertTrue(self._wait(lambda: order == ["a"]))
        self.app.runner.run("b", job("b"))
        self.app.runner.run("c", job("c"))
        self.assertEqual(self.app.runner.pending, 2)

        gate.set()
        self.assertTrue(self._settle())
        self.assertEqual(order, ["a", "b", "c"])
        # The window must not be unlocked *between* jobs: set_busy(False) may
        # fire only once, at the very end of the drain.
        self.assertEqual(self.app.busy_calls.count(False), 1)
        self.assertIs(self.app.busy_calls[-1], False)
        self.assertNotIn(False, self.app.busy_calls[:-1])

    def test_goes_idle_when_queue_empty(self):
        self.app.runner.run("a", lambda: {})
        self.assertTrue(self._settle())
        self.assertEqual(self.app.progress_calls[-1], (1, 1, "idle"))
        self.assertEqual(self.app.btn_stop.text, "Stop")

    def test_clear_queue_drops_waiting_only(self):
        gate = threading.Event()
        ran = []

        def blocked():
            ran.append("a")
            gate.wait(5)
            return {}

        self.app.runner.run("a", blocked)
        self.assertTrue(self._wait(lambda: ran == ["a"]))
        self.app.runner.run("b", lambda: ran.append("b") or {})
        self.app.runner.run("c", lambda: ran.append("c") or {})
        self.assertEqual(self.app.runner.pending, 2)

        self.assertEqual(self.app.runner.clear_queue(), 2)
        self.assertEqual(self.app.runner.pending, 0)
        self.assertEqual(self.app.btn_stop.text, "Stop")

        gate.set()
        self.assertTrue(self._settle())
        self.assertEqual(ran, ["a"], "queued jobs must not run after Stop")

    # -- context tagging -------------------------------------------------- #

    def test_ctx_is_captured_at_enqueue(self):
        gate = threading.Event()
        self.app.runner.run("a", lambda: gate.wait(5) or {})
        self.assertEqual(self.app._job_ctx["pid"], 1)
        self.assertEqual(self.app._job_ctx["table"], "t")

        # A queued job remembers the project it was queued *for*, even though
        # the user switches in the meantime.
        self.app.runner.run("b", lambda: {})
        self.app.project_id = 42
        self.app.table_var.set("other")
        gate.set()
        self.assertTrue(self._settle())
        self.assertEqual(self.app._job_ctx["pid"], 1)
        self.assertEqual(self.app._job_ctx["table"], "t")

    def test_failure_does_not_stop_the_queue(self):
        order = []
        gate = threading.Event()

        def boom():
            gate.wait(5)
            raise RuntimeError("nope")

        self.app.runner.run("a", boom)
        self.app.runner.run("b", lambda: order.append("b") or {})
        gate.set()
        self.assertTrue(self._settle())
        self.assertEqual(order, ["b"])


# --------------------------------------------------------------------------- #
# GUI wiring (source level)
# --------------------------------------------------------------------------- #

class TestGuiWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        cls.flat = re.sub(r"\s+", "", cls.src)
        tree = ast.parse(cls.src)
        cls.fns = {n.name: n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}

    def _src_of(self, name: str) -> str:
        return ast.get_source_segment(self.src, self.fns[name])

    def test_set_busy_no_longer_locks_the_window(self):
        body = self._src_of("set_busy")
        self.assertNotIn("project_menu", body)
        self.assertNotIn("btn_ingest", body)
        self.assertIn("btn_stop", body)

    def test_stop_clears_the_queue(self):
        body = self._src_of("do_stop")
        self.assertIn("clear_queue", body)

    def test_finished_handler_delegates_to_finish(self):
        # finish() owns the unlock; the handler must not reset the UI itself.
        body = self._src_of("_handle")
        branch = body[body.index('elif kind == "finished":'):]
        self.assertIn("self.runner.finish()", branch)
        self.assertNotIn("set_busy(False)", branch)

    def test_done_handler_guards_foreign_results(self):
        body = self._src_of("_on_done")
        self.assertIn("foreign", body)
        self.assertIn("self.project_id", body)

    def test_runner_is_a_queue(self):
        self.assertIn("def clear_queue", self.src)
        self.assertIn("def _start_next", self.src)
        self.assertIn("def pending", self.src)
        self.assertNotIn("Another step is still running", self.src)


# --------------------------------------------------------------------------- #
# GUI-level checks (need a display; skipped otherwise)
# --------------------------------------------------------------------------- #

class TestGuiLive(unittest.TestCase):
    def _app(self):
        try:
            import tk_app as mod
        except Exception as e:                             # noqa: BLE001
            self.skipTest(f"GUI not available: {e}")
        import tempfile
        mod.DB_PATH = pathlib.Path(
            tempfile.mkdtemp(prefix="ep-queue-")) / "app.db"
        try:
            app = mod.App()
        except Exception as e:                             # noqa: BLE001
            self.skipTest(f"Tk not available: {e}")
        app.probe_ollama = lambda: None       # no network chatter in tests
        app.update()
        self.addCleanup(app.destroy)
        return app

    def test_project_menu_and_buttons_stay_live_while_busy(self):
        app = self._app()
        app.set_busy(True)
        self.assertEqual(str(app.project_menu.cget("state")), "normal")
        self.assertEqual(str(app.btn_stop.cget("state")), "normal")
        self.assertEqual(str(app.btn_ingest.cget("state")), "normal")
        app.set_busy(False)
        self.assertEqual(str(app.btn_stop.cget("state")), "disabled")

    def test_badge_counts_waiting_jobs(self):
        app = self._app()
        app.runner._queue.append(("x", None, (), {}, {}))
        app._update_queue_badge()
        self.assertEqual(app.btn_stop.cget("text"), "Stop (1)")
        app.runner._queue.clear()
        app._update_queue_badge()
        self.assertEqual(app.btn_stop.cget("text"), "Stop")

    def test_switch_mid_job_keeps_the_job_on_its_own_table(self):
        app = self._app()
        seen = {}
        gate = threading.Event()

        def job(**kw):
            seen.update(kw)
            gate.wait(5)
            return {}

        app.table_var.set("alpha")
        app.runner.run("embed", job, table="alpha")
        end = time.time() + 5
        while not seen and time.time() < end:
            time.sleep(0.01)
        self.assertEqual(seen.get("table"), "alpha")

        app.table_var.set("beta")          # the user switches project
        gate.set()
        for _ in range(400):
            app.update()
            if not app.runner.busy:
                break
            time.sleep(0.01)
        # The job kept the table it was queued for -- a lazy read of
        # self.table_var would have retargeted it at "beta".
        self.assertEqual(seen.get("table"), "alpha")

    def test_answer_from_another_project_is_not_shown(self):
        app = self._app()
        app.project_id = 1
        app._job_ctx = {"title": "chat", "pid": 2, "table": "other"}
        before = len(app._history)
        app._on_done({"answer": "hello from another project", "hits": [],
                      "history": []})
        self.assertEqual(len(app._history), before)
        self.assertIn("another project", app.log.get("1.0", "end"))

    def test_answer_for_current_project_is_shown(self):
        app = self._app()
        app.project_id = 1
        app._job_ctx = {"title": "chat", "pid": 1, "table": "t"}
        before = len(app._history)
        app._on_done({"answer": "hello", "hits": [], "history": []})
        self.assertGreater(len(app._history), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
