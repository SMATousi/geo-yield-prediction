"""V&V entry point (fallback, authored by the pipeline).

The V&V runner loads ``run(config)`` from here. This adapter delegates to the
verification battery committed under ``.primae/vandv/`` — it does not invent
results: when the battery is absent or cannot run, it says so and returns no
checks rather than reporting success.
"""
from __future__ import annotations

import json
import pathlib
import runpy
from typing import Any, Dict

_ROOT = pathlib.Path(__file__).resolve().parent
_PROBE = _ROOT / ".primae" / "vandv" / "probe.py"
_REPORT = _ROOT / ".primae" / "vandv" / "probe_report.json"


def run(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute the committed battery and return its report as a dict."""
    if not _PROBE.is_file():
        return {
            "ok": False,
            "reason": "no verification battery committed under .primae/vandv",
            "checks": [],
        }
    try:
        runpy.run_path(str(_PROBE), run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 — report, never raise into V&V
        return {"ok": False, "reason": f"battery failed: {exc}", "checks": []}
    if _REPORT.is_file():
        try:
            report = json.loads(_REPORT.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"unreadable report: {exc}", "checks": []}
        checks = report.get("checks") or report.get("results") or []
        return {"ok": True, "checks": checks, "report": report}
    return {"ok": False, "reason": "battery produced no report", "checks": []}
