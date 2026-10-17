"""test_providers.py -- offline checks for the hosted chat providers.

Covers the OpenAI-compatible gateways added on top of Anthropic/OpenAI
(OpenCode, The Grid, custom), the manual 3xx redirect following that The
Grid needs, and the GUI/config wiring. No network: `_post` is replaced by
a recorder, and the redirect test uses a throwaway local HTTP server.

Run:  python test_providers.py
"""

import ast
import http.server
import json
import re
import sys
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import chat_providers as C


class _Recorder:
    """Stand-in for chat_providers._post that records the request."""

    def __init__(self, result=None):
        self.calls: list[tuple] = []
        self.result = result or {"choices": [{"message": {"content": "ok"}}]}

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), dict(payload)))
        return self.result


class TestRouting(unittest.TestCase):
    def setUp(self):
        self._orig = C._post

    def tearDown(self):
        C._post = self._orig

    def _run(self, provider, result=None, **kw):
        rec = _Recorder(result)
        C._post = rec
        text = C.complete(provider, model="m1", api_key="k1",
                          system="sys", user="hi", **kw)
        return text, rec.calls[0]

    def test_openai(self):
        text, (url, h, body) = self._run("openai")
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(h["Authorization"], "Bearer k1")
        self.assertEqual(body["model"], "m1")
        self.assertEqual(body["messages"][0],
                         {"role": "system", "content": "sys"})
        self.assertEqual(body["messages"][-1],
                         {"role": "user", "content": "hi"})
        self.assertEqual(text, "ok")

    def test_opencode(self):
        _, (url, h, _b) = self._run("opencode")
        self.assertEqual(
            url, "https://opencode.ai/inference/openai/v1/chat/completions")
        self.assertEqual(h["Authorization"], "Bearer k1")

    def test_thegrid(self):
        _, (url, h, _b) = self._run("thegrid")
        self.assertEqual(url, "https://api.thegrid.ai/v1/chat/completions")
        self.assertEqual(h["Authorization"], "Bearer k1")

    def test_custom_uses_base_url(self):
        _, (url, _h, _b) = self._run("custom",
                                     base_url="https://host/v1/")
        # trailing slash stripped, path appended once
        self.assertEqual(url, "https://host/v1/chat/completions")

    def test_custom_requires_base_url(self):
        C._post = _Recorder()
        with self.assertRaises(C.ProviderError):
            C.complete("custom", model="m", api_key="k",
                       system="s", user="u")

    def test_anthropic_shape(self):
        text, (url, h, body) = self._run(
            "anthropic",
            result={"content": [{"type": "text", "text": "claude"}]})
        self.assertEqual(url, C.ANTHROPIC_URL)
        self.assertEqual(h["x-api-key"], "k1")
        self.assertEqual(h["anthropic-version"], C.ANTHROPIC_VERSION)
        self.assertIn("messages", body)
        self.assertEqual(text, "claude")

    def test_history_inserted_before_user(self):
        hist = [{"role": "user", "content": "prev"},
                {"role": "assistant", "content": "a"}]
        _, (_u, _h, body) = self._run("openai", history=hist)
        self.assertEqual(body["messages"][1:3], hist)
        self.assertEqual(body["messages"][-1]["content"], "hi")

    def test_unknown_provider(self):
        C._post = _Recorder()
        with self.assertRaises(C.ProviderError):
            C.complete("nope", model="m", api_key="k", system="s", user="u")

    def test_missing_key(self):
        C._post = _Recorder()
        with self.assertRaises(C.ProviderError):
            C.complete("opencode", model="m", api_key="", system="s", user="u")

    def test_missing_model(self):
        C._post = _Recorder()
        with self.assertRaises(C.ProviderError):
            C.complete("thegrid", model="", api_key="k", system="s", user="u")

    def test_http_messages_are_actionable(self):
        self.assertIn("API key", C._http_message(401, "x", "u"))
        self.assertIn("credits", C._http_message(402, "x", "u"))
        self.assertIn("instrument", C._http_message(404, "x", "u"))

    def test_connection_pings_the_real_path(self):
        rec = _Recorder({"choices": [{"message": {"content": "OK"}}]})
        C._post = rec
        msg = C.test_connection("opencode", model="m1", api_key="k1")
        self.assertIn("OK", msg)
        self.assertIn("m1", msg)
        self.assertEqual(
            rec.calls[0][0],
            "https://opencode.ai/inference/openai/v1/chat/completions")

    def test_connection_rejects_ollama(self):
        with self.assertRaises(C.ProviderError):
            C.test_connection("ollama", model="m", api_key="k")


class TestListModels(unittest.TestCase):
    """`list_models` routing + tolerant parsing, all offline."""

    def setUp(self):
        self._orig = C._request

    def tearDown(self):
        C._request = self._orig

    def _stub(self, payload):
        calls: list[tuple] = []

        def fake(method, url, headers, body, timeout):
            calls.append((method, url, dict(headers), body))
            return json.dumps(payload)

        C._request = fake
        return calls

    def test_openai_url_headers_and_shape(self):
        calls = self._stub({"data": [{"id": "gpt-4o-mini"}]})
        rows = C.list_models("openai", api_key="k1")
        self.assertEqual(rows, [("gpt-4o-mini", "")])
        method, url, h, body = calls[0]
        self.assertEqual((method, url),
                         ("GET", "https://api.openai.com/v1/models"))
        self.assertEqual(h["Authorization"], "Bearer k1")
        self.assertIsNone(body)

    def test_opencode_uses_the_gateway_catalog(self):
        calls = self._stub({"data": [{"id": "deepseek-v4.1-flash-free"}]})
        C.list_models("opencode", api_key="k1")
        self.assertEqual(calls[0][1],
                         "https://opencode.ai/inference/v1/models")

    def test_anthropic_headers(self):
        calls = self._stub({"data": [{"id": "claude-3-5-sonnet"}]})
        C.list_models("anthropic", api_key="k1")
        _m, url, h, _b = calls[0]
        self.assertEqual(url, "https://api.anthropic.com/v1/models")
        self.assertEqual(h["x-api-key"], "k1")
        self.assertEqual(h["anthropic-version"], C.ANTHROPIC_VERSION)

    def test_thegrid_is_public_and_ids_are_instruments(self):
        payload = {"object": "list", "data": [
            {"id": "text-prime", "name": "The Grid: Text Prime"},
            {"id": "code-max", "name": "The Grid: Code Max"}]}
        calls = self._stub(payload)
        rows = C.list_models("thegrid")              # no key needed
        self.assertEqual(rows, [("code-max", "The Grid: Code Max"),
                                ("text-prime", "The Grid: Text Prime")])
        self.assertEqual(calls[0][1], "https://api.thegrid.ai/v1/models")

    def test_custom_uses_the_base_url(self):
        calls = self._stub(["m-a", "m-b"])
        rows = C.list_models("custom", api_key="k", base_url="https://h/v1/")
        self.assertEqual(rows, [("m-a", ""), ("m-b", "")])
        self.assertEqual(calls[0][1], "https://h/v1/models")

    def test_tolerates_other_shapes(self):
        self._stub({"models": [{"model": "x"}]})
        self.assertEqual(C.list_models("openai", api_key="k"), [("x", "")])

    def test_key_required_except_for_thegrid(self):
        self._stub({"data": [{"id": "x"}]})
        with self.assertRaises(C.ProviderError):
            C.list_models("openai")
        with self.assertRaises(C.ProviderError):
            C.list_models("opencode")
        C.list_models("thegrid")                     # public: no raise

    def test_custom_requires_base_url(self):
        self._stub(["x"])
        with self.assertRaises(C.ProviderError):
            C.list_models("custom", api_key="k")

    def test_empty_catalog_raises(self):
        self._stub({"data": []})
        with self.assertRaises(C.ProviderError):
            C.list_models("openai", api_key="k")

    def test_unknown_provider(self):
        with self.assertRaises(C.ProviderError):
            C.list_models("nope")


class TestRedirect(unittest.TestCase):
    """The Grid 307s; the POST body must survive the hop."""

    def test_307_is_followed_and_body_preserved(self):
        seen: dict = {}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):                   # keep test output quiet
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(n)
                if self.path == "/redirect":
                    self.send_response(307)
                    self.send_header("Location", "/final")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                seen["path"] = self.path
                seen["body"] = body
                payload = json.dumps(
                    {"choices": [{"message": {"content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{srv.server_port}/redirect"
            out = C._post(url, {"Content-Type": "application/json"},
                          {"a": 1}, timeout=5)
        finally:
            srv.shutdown()
            srv.server_close()

        self.assertEqual(out["choices"][0]["message"]["content"], "ok")
        self.assertEqual(seen["path"], "/final")
        self.assertEqual(json.loads(seen["body"]), {"a": 1})

    def test_307_is_followed_on_get_too(self):
        seen: dict = {}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/redirect":
                    self.send_response(307)
                    self.send_header("Location", "/final")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                seen["path"] = self.path
                payload = b'{"data": [{"id": "m"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_port}/redirect"
            raw = C._request("GET", url, {}, None, timeout=5)
        finally:
            srv.shutdown()
            srv.server_close()

        self.assertEqual(seen["path"], "/final")
        self.assertEqual(json.loads(raw)["data"][0]["id"], "m")


class TestConfig(unittest.TestCase):
    def test_secret_keys_cover_new_providers(self):
        import backup
        for k in ("anthropic_key", "openai_key", "opencode_key",
                  "thegrid_key", "custom_key"):
            self.assertIn(k, backup._SECRET_KEYS)

    def test_store_defaults(self):
        import store
        for k in ("opencode_key", "thegrid_key", "custom_key",
                  "custom_base_url"):
            self.assertIn(k, store._SETTINGS_DEFAULTS)

    def test_provider_registry(self):
        self.assertEqual(C.PROVIDERS["openai"], "https://api.openai.com/v1")
        self.assertIn("opencode", C.PROVIDERS)
        self.assertIn("thegrid", C.PROVIDERS)


class TestGuiWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (HERE / "tk_app.py").read_text(encoding="utf-8")
        tree = ast.parse(cls.src)
        cls.consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in (
                            "CHAT_PROVIDERS", "PROVIDER_KEY_SETTING"):
                        cls.consts[tgt.id] = ast.literal_eval(node.value)

    def test_chat_providers_constant(self):
        self.assertEqual(
            self.consts.get("CHAT_PROVIDERS"),
            ["ollama", "anthropic", "openai", "opencode", "thegrid", "custom"])

    def test_key_setting_map(self):
        self.assertEqual(self.consts.get("PROVIDER_KEY_SETTING"), {
            "anthropic": "anthropic_key",
            "openai": "openai_key",
            "opencode": "opencode_key",
            "thegrid": "thegrid_key",
            "custom": "custom_key",
        })

    def test_no_hardcoded_provider_lists_left(self):
        self.assertNotIn('values=["ollama", "anthropic", "openai"]', self.src)
        self.assertNotIn('("ollama", "anthropic", "openai")', self.src)

    def test_do_send_uses_the_map_and_passes_base_url(self):
        self.assertIn("PROVIDER_KEY_SETTING.get(provider", self.src)
        self.assertIn("custom_base_url", self.src)
        self.assertIn("base_url: str = \"\"", self.src)   # _chat signature

    def test_settings_has_a_test_button_per_provider(self):
        self.assertIn("def do_test_provider", self.src)
        self.assertIn("def _test_model_for", self.src)
        self.assertIn("provider_test_lbl", self.src)
        self.assertIn('kind == "provider_test"', self.src)
        flat = re.sub(r"\s+", "", self.src)       # ignore wrapping
        for prov in ("anthropic", "openai", "opencode", "thegrid", "custom"):
            self.assertIn(f'"{prov}");r+=1', flat)

    def test_settings_has_a_models_button(self):
        self.assertIn("def do_list_models", self.src)
        self.assertIn("def _render_provider_models", self.src)
        self.assertIn("provider_models_box", self.src)
        self.assertIn("provider_models_lbl", self.src)
        self.assertIn('kind == "provider_models"', self.src)
        flat = re.sub(r"\s+", "", self.src)
        self.assertIn("self.do_list_models(p)", flat)


class TestProviderTestButton(unittest.TestCase):
    """The Settings 'Test' button, exercised off the network.

    Constructing a real window needs a display; skip cleanly when there is
    none (CI) instead of failing the suite.
    """

    def _app(self):
        import tempfile
        import pathlib
        try:
            import tk_app
        except Exception as e:                             # noqa: BLE001
            self.skipTest(f"GUI not available: {e}")
        tk_app.DB_PATH = pathlib.Path(
            tempfile.mkdtemp(prefix="ep-ptest-")) / "app.db"
        try:
            app = tk_app.App()
        except Exception as e:                             # noqa: BLE001
            self.skipTest(f"Tk not available: {e}")
        app.update()
        self.addCleanup(app.destroy)
        return app

    def test_missing_key_is_reported_before_anything_else(self):
        app = self._app()
        app.sv["opencode_key"].set("")
        app.provider_var.set("ollama")
        app.do_test_provider("opencode")
        self.assertIn("no API key", app.provider_test_lbl.cget("text"))

    def test_missing_model_is_reported(self):
        app = self._app()
        app.sv["opencode_key"].set("k")
        app.sv["default_chat_model"].set("")
        app.chat_entry_var.set("")
        app.provider_var.set("ollama")
        app.do_test_provider("opencode")
        self.assertIn("no model", app.provider_test_lbl.cget("text"))

    def test_custom_without_base_url_is_reported(self):
        app = self._app()
        app.sv["custom_key"].set("k")
        app.sv["custom_base_url"].set("")
        app.do_test_provider("custom")
        self.assertIn("base URL", app.provider_test_lbl.cget("text"))

    def test_model_resolver_prefers_the_chat_tab(self):
        app = self._app()
        app.sv["default_chat_model"].set("fallback")
        app.provider_var.set("thegrid")
        app.chat_entry_var.set("text-prime")
        self.assertEqual(app._test_model_for("thegrid"), "text-prime")
        app.provider_var.set("ollama")
        self.assertEqual(app._test_model_for("thegrid"), "fallback")

    def test_models_button_warns_without_a_key(self):
        app = self._app()
        app.sv["openai_key"].set("")
        app.do_list_models("openai")
        self.assertIn("no API key", app.provider_models_lbl.cget("text"))

    def test_models_button_warns_without_base_url(self):
        app = self._app()
        app.sv["custom_key"].set("k")
        app.sv["custom_base_url"].set("")
        app.do_list_models("custom")
        self.assertIn("base URL", app.provider_models_lbl.cget("text"))

    def test_models_render_lists_ids_and_names(self):
        app = self._app()
        app._render_provider_models("thegrid", [("text-prime", "Text Prime"),
                                                ("code-max", "")])
        text = app.provider_models_box.get("1.0", "end")
        self.assertIn("text-prime", text)
        self.assertIn("# Text Prime", text)
        self.assertIn("code-max", text)
        self.assertIn("2 model(s)", app.provider_models_lbl.cget("text"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
