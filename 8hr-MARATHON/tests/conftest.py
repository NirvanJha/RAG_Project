"""
Offline test setup: stub out tracing and the LLM gateway so unit tests run
without API keys, network access, or the heavy ML dependencies installed.
"""
import contextlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import logfire  # noqa: F401
except ImportError:
    stub = types.ModuleType("logfire")
    stub.span = lambda *a, **k: contextlib.nullcontext()
    for name in ("info", "warning", "warn", "error", "configure"):
        setattr(stub, name, lambda *a, **k: None)
    sys.modules["logfire"] = stub

# The planner builds a Portkey-backed LLM at import time.
gateway = types.ModuleType("app.gateway")
gateway.get_langchain_llm = lambda **k: None
gateway.portkey_client = None
gateway.extract_cache_status = lambda r: "MISS"
sys.modules["app.gateway"] = gateway
