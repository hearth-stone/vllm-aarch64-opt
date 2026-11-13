# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate answer and stream checks independently of model performance."""

import importlib.util
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/accept_deepseek_v4_cpu.py"
SPEC = importlib.util.spec_from_file_location("cpu_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


@pytest.mark.parametrize(
    "text,finish,correct",
    [
        ('{"编号":"验收001"}', "stop", True),
        ('```json\n{"编号":"验收001"}\n```', "stop", True),
        ('{"编号":"验收002"}', "stop", False),
        ('{"编号":"验收001"}', "length", False),
        ('{"编号":"验收001","额外":"解释"}', "stop", False),
        ('说明：{"编号":"验收001"}', "stop", False),
        ('{"编号":"验收001�"}', "stop", False),
        ("", "stop", False),
    ],
)
def test_answer_requires_gold_fields_and_complete_output(text, finish, correct):
    assert acceptance.judge(text, {"编号": "验收001"}, finish)["correct"] is correct


@contextmanager
def stream_server(events, done=True, status=200):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.end_headers()
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            if done:
                self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    "done,status,error", [(True, 200, False), (False, 200, True), (True, 500, True)]
)
def test_stream_requires_success_and_done_marker(done, status, error):
    events = [
        {
            "choices": [
                {
                    "text": '{"编号":"验收001"}',
                    "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    "finish_reason": "stop",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
    ]
    case = {
        "case_id": "验收001",
        "input_tokens": 2,
        "prompt_token_ids": [1, 2],
        "expected": {"编号": "验收001"},
    }
    with stream_server(events, done, status) as url:
        row = acceptance.complete(url, "model", case, 64, 10)
    assert (row["error"] is not None) is error
    assert row["judgment"]["correct"] is (not error)
    if not error:
        assert row["ttft_s"] is not None and row["tpot_s"] is not None


def test_native_batch_maps_out_of_order_choices_to_their_own_prompt():
    cases = [
        {"case_id": f"验收{i}", "input_tokens": 2, "expected": {"编号": f"验收{i}"}}
        for i in range(2)
    ]
    response = {
        "choices": [
            {
                "index": i,
                "text": json.dumps(cases[i]["expected"]),
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1]},
            }
            for i in (1, 0)
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    rows = acceptance.validate_api_batch(response, cases)
    assert [r["case_id"] for r in rows] == ["验收0", "验收1"]
    assert all(r["judgment"]["correct"] for r in rows)
    response["choices"][0]["text"] = response["choices"][1]["text"]
    rows = acceptance.validate_api_batch(response, cases)
    assert [r["judgment"]["correct"] for r in rows] == [True, False]


@pytest.mark.parametrize("indices", [[0], [0, 0]])
def test_native_batch_rejects_missing_or_duplicate_answer_indices(indices):
    response = {"choices": [{"index": i} for i in indices]}
    with pytest.raises(ValueError, match="missing or duplicate"):
        acceptance.validate_api_batch(response, [{}, {}])


@pytest.mark.parametrize("difference", [None, "model", "missing", "length", "gold"])
def test_continued_phase_requires_matching_single_baseline(tmp_path, difference):
    case = {"case_id": "验收001", "input_tokens": 2, "expected": {"编号": "验收001"}}
    protocol = {"model": "bf16"}
    row = dict(case)
    if difference == "model":
        protocol["model"] = "int8"
    elif difference == "missing":
        row["case_id"] = "验收002"
    elif difference == "length":
        row["input_tokens"] = 3
    elif difference == "gold":
        row["expected"] = {"编号": "验收002"}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    path = tmp_path / "batch1-requests.json"
    path.write_text(json.dumps([row]))
    if difference is not None:
        with pytest.raises(ValueError):
            acceptance.load_baseline(path, [case], "bf16")
    else:
        assert acceptance.load_baseline(path, [case], "bf16") == {"验收001": row}


def test_pressure_failure_keeps_outcomes_of_already_started_requests(
    tmp_path, monkeypatch
):
    cases = [
        {
            "case_id": str(i),
            "input_tokens": 2,
            "prompt_token_ids": [1, i],
            "expected": {"编号": str(i)},
        }
        for i in (1, 2)
    ]
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text("".join(json.dumps(c) + "\n" for c in cases))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "protocol.json").write_text(json.dumps({"model": "test"}))
    baseline_rows = [
        {**case, "error": None, "judgment": {"parsed": case["expected"]}}
        for case in cases
    ]
    (baseline / "batch1-requests.json").write_text(json.dumps(baseline_rows))
    service = tmp_path / "service"
    service.mkdir()
    (service / "server.log").write_text(
        "\n".join(f"Worker_TP{i} pid={os.getpid()}" for i in range(4))
    )
    release_success = threading.Event()
    real_save = acceptance.save

    def save_and_release(path, value):
        real_save(path, value)
        if path.name == "stress-requests.json" and any(r["error"] for r in value):
            release_success.set()

    def complete_request(_url, _model, case, _max_tokens, _timeout):
        failed = case["case_id"] == "1"
        if not failed:
            assert release_success.wait(timeout=5)
        return {
            "case_id": case["case_id"],
            "error": "HTTP 500" if failed else None,
            "elapsed_s": 0.1,
            "ttft_s": 0.01,
            "tpot_s": 0.01,
            "usage": {"completion_tokens": 2},
            "judgment": {
                "correct": not failed,
                "parsed": case["expected"],
                "garbled": False,
                "empty": False,
                "truncated": False,
            },
        }

    monkeypatch.setattr(acceptance, "save", save_and_release)
    monkeypatch.setattr(acceptance, "complete", complete_request)
    output = tmp_path / "result"
    with stream_server([]) as url:
        args = SimpleNamespace(
            cases=cases_file,
            model="test",
            service_run=service,
            output_dir=output,
            baseline=baseline / "batch1-requests.json",
            base_url=url,
            warmup=0,
            batch_sizes=[],
            stress_concurrency=2,
            stress_seconds=60,
            max_tokens=64,
            timeout=10,
        )
        with pytest.raises(RuntimeError, match="Sustained-load request failed"):
            acceptance.run(args)
    rows = json.loads((output / "stress-requests.json").read_text())
    assert {r["case_id"] for r in rows} == {"1", "2"}
    summary = json.loads((output / "summary.json").read_text())["stress2"]
    assert summary["started_requests"] == summary["requests"] == 2
    assert summary["errors"] == summary["correct"] == 1
    assert summary["aborted_due_to_error"] is True
    assert summary["load_duration_s"] < 60
