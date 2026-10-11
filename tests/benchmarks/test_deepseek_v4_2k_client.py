# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check HTTP timing, distinct sampling and offline JSON analysis."""

import importlib.util
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/benchmark_deepseek_v4_2k.py"
SPEC = importlib.util.spec_from_file_location("deepseek_2k_client", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


@contextmanager
def completion_server(status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            self.send_response(status)
            self.end_headers()
            # Delay after headers: E2E must include reading the response body.
            time.sleep(0.02)
            self.wfile.write(
                json.dumps(
                    {
                        "usage": {
                            "prompt_tokens": len(payload["prompt"]),
                            "completion_tokens": 1,
                        },
                        "choices": [{"text": "测", "finish_reason": "length"}],
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_packaged_dataset_has_100_distinct_exact_2k_inputs():
    cases = client.select_cases(client.DEFAULT_DATASET, 100, 19)
    assert len({case["case_id"] for case in cases}) == 100
    assert len({tuple(case["prompt_token_ids"]) for case in cases}) == 100
    assert {len(case["prompt_token_ids"]) for case in cases} == {2048}
    assert cases == client.select_cases(client.DEFAULT_DATASET, 100, 19)
    assert cases != client.select_cases(client.DEFAULT_DATASET, 100, 20)


def test_statistics_exclude_failed_requests_and_interpolate_percentiles():
    rows = [{"success": True, "e2e_s": n} for n in [1, 2, 3, 4]]
    rows.append({"success": False, "e2e_s": 240})
    summary = client.summarize(rows)
    assert summary["attempted"] == 5
    assert summary["failed"] == 1
    assert summary["avg_s"] == pytest.approx(2.5)
    assert summary["p50_s"] == pytest.approx(2.5)
    assert summary["p95_s"] == pytest.approx(3.85)
    assert summary["p99_s"] == pytest.approx(3.97)


def test_http_run_records_requests_timing_order_and_excludes_warmup(
    tmp_path, monkeypatch
):
    sleeps: list[float] = []
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(
            time=time.time, perf_counter=time.perf_counter, sleep=sleeps.append
        ),
    )
    with completion_server() as (url, received):
        rc = client.main(
            [
                "--requests",
                "3",
                "--warmup",
                "1",
                "--seed",
                "19",
                "--prepack-cache-version",
                "v028-482e03e",
                "--interval-seconds",
                "5",
                "--base-url",
                url,
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
    folder = tmp_path / "result"
    run = json.loads((folder / "requests.json").read_text())
    rows = run["requests"]
    assert run["protocol"]["prepack_cache_version"] == "v028-482e03e"
    expected = client.select_cases(client.DEFAULT_DATASET, 3, 19)
    assert rc == 0
    assert [r["case_id"] for r in rows] == [c["case_id"] for c in expected]
    assert [r["prompt"] for r in received] == [expected[0]["prompt_token_ids"]] + [
        c["prompt_token_ids"] for c in expected
    ]
    assert [r["request"] for r in run["warmups"] + rows] == received
    assert all(r["max_tokens"] == 1 and r["ignore_eos"] for r in received)
    assert all(r["e2e_s"] >= 0.02 and r["end_unix"] >= r["start_unix"] for r in rows)
    assert sleeps == [5, 5, 5]
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["successful"] == 3
    assert summary["warmup_attempted"] == 1
    assert summary["complete"]
    assert summary["avg_s"] == pytest.approx(sum(r["e2e_s"] for r in rows) / 3)
    assert (
        client.main(
            [
                "--analyze",
                str(folder / "requests.json"),
                "--output-dir",
                str(tmp_path / "recomputed"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "recomputed/summary.json").read_text()) == summary


@pytest.mark.parametrize("warmup", [0, 1])
def test_http_failure_saves_partial_json_and_returns_nonzero(tmp_path, warmup):
    with completion_server(status=500) as (url, received):
        rc = client.main(
            [
                "--requests",
                "3",
                "--warmup",
                str(warmup),
                "--interval-seconds",
                "0",
                "--base-url",
                url,
                "--output-dir",
                str(tmp_path / "failed"),
            ]
        )
    summary = json.loads((tmp_path / "failed/summary.json").read_text())
    run = json.loads((tmp_path / "failed/requests.json").read_text())
    assert rc == 1
    assert len(received) == 1
    assert not summary["complete"]
    assert summary["successful"] == 0
    assert summary["avg_s"] is None
    assert summary["warmup_failed"] == warmup
    assert summary["failed"] == 1 - warmup
    assert client.analyze_results(run) == summary
    saved = (run["warmups"] + run["requests"])[0]
    assert saved["error"].startswith("HTTP 500:")


def test_requested_count_cannot_repeat_dataset_cases():
    with pytest.raises(ValueError, match="1–100"):
        client.select_cases(client.DEFAULT_DATASET, 101, 0)
