from __future__ import annotations

from pathlib import Path

import pytest

import main
from main import create_app, load_api_key


def test_load_api_key_prefers_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    key_file = tmp_path / "openai_api_key.txt"
    key_file.write_text("from-file", encoding="utf-8")

    assert load_api_key(key_file=key_file) == "from-env"


def test_load_api_key_reads_file_when_env_missing(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    key_file = tmp_path / "openai_api_key.txt"
    key_file.write_text("from-file", encoding="utf-8")

    assert load_api_key(env=env, key_file=key_file) == "from-file"


def test_load_api_key_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    key_file = tmp_path / "openai_api_key.txt"
    key_file.write_text("\n# comment\n  sk-test\n", encoding="utf-8")

    assert load_api_key(env={}, key_file=key_file) == "sk-test"


def test_load_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        load_api_key(env={}, key_file=tmp_path / "openai_api_key.txt")


def test_normalise_messages_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError):
        main._normalise_messages([{"role": "user", "content": ""}])

    with pytest.raises(ValueError):
        main._normalise_messages([{"role": "invalid", "content": "question"}])


def test_create_app_health_endpoint() -> None:
    app = create_app()
    client = app.test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_chat_endpoint_validates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app()
    client = app.test_client()

    response = client.post("/chat", json={"messages": "invalid"})
    assert response.status_code == 400

    monkeypatch.setattr(main, "run_deep_research", lambda messages: {"answer": "ok", "trace": []})
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.get_json() == {"answer": "ok", "trace": []}


def test_run_deep_research_returns_clarifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "_triage_agent",
        lambda messages, trace: {
            "decision": "clarify",
            "missing_information": "詳細",
            "summary": "要約",
        },
    )
    monkeypatch.setattr(
        main,
        "_clarifier_agent",
        lambda messages, triage, trace: {"question": "補足情報を教えてください。"},
    )

    result = main.run_deep_research([{"role": "user", "content": "テスト"}])

    assert result["answer"].startswith("Clarifier Agent:")
    assert isinstance(result["trace"], list)


def test_run_deep_research_proceeds_to_research(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "_triage_agent",
        lambda messages, trace: {
            "decision": "proceed",
            "missing_information": "",
            "summary": "要約",
        },
    )
    monkeypatch.setattr(
        main,
        "_instruction_builder_agent",
        lambda messages, triage, trace: {
            "research_question": "RQ",
            "background_context": "背景",
            "deliverable_expectations": "期待",
            "subtopics": ["A"],
            "tone": "丁寧",
        },
    )
    monkeypatch.setattr(main, "_research_agent", lambda brief, trace: "結果")

    result = main.run_deep_research([{"role": "user", "content": "テスト"}])

    assert result["answer"] == "結果"
    assert isinstance(result["trace"], list)


def test_research_agent_uses_web_search_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyClient:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.responses = self

        def create(self, **kwargs):
            self.kwargs = kwargs

            class DummyResponse:
                output_text = "ok"

            captured.update(kwargs)
            return DummyResponse()

    monkeypatch.setattr(main, "get_openai_client", lambda: DummyClient())

    brief = {
        "research_question": "RQ",
        "background_context": "背景",
        "deliverable_expectations": "期待",
        "subtopics": ["A"],
        "tone": "丁寧",
    }

    trace: list[main.TraceEntry] = []

    assert main._research_agent(brief, trace) == "ok"
    assert "tools" in captured
    assert captured["tools"] == main.DEEP_RESEARCH_TOOLS
    assert trace[-1]["content"] == "ok"


def test_research_agent_handles_incomplete_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the initial streamed response is incomplete with no text, the agent
    should request a continuation with reduced verbosity and effort.
    """

    # Simple event objects for the stream
    class ECreated:
        def __init__(self, rid: str) -> None:
            self.type = "response.created"
            self.response = type("Resp", (), {"id": rid})()

    class ECompleted:
        type = "response.completed"

    class DummyResponseNoText:
        def __init__(self, rid: str) -> None:
            self.id = rid
            self.output_text = None
            self.status = "incomplete"

            class Incomplete:
                def model_dump(self):
                    return {"reason": "max_output_tokens"}

            self.incomplete_details = Incomplete()

        def to_json(self, **_: object):  # used in _log_raw
            return {"status": self.status, "incomplete_details": {"reason": "max_output_tokens"}}

    class DummyResponseText:
        def __init__(self, rid: str) -> None:
            self.id = rid
            self.output_text = "FINAL"

        def to_json(self, **_: object):
            return {"status": "completed"}

    class DummyStream:
        def __init__(self, rid: str) -> None:
            self._rid = rid

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - not needed
            return False

        def __iter__(self):
            # Created then completed, but final has no text and is incomplete
            yield ECreated(self._rid)
            yield ECompleted()

        def get_final_response(self):
            return DummyResponseNoText(self._rid)

    captured_calls: list[dict[str, object]] = []

    class DummyClient:
        def __init__(self) -> None:
            self.responses = self

        # stream() used first
        def stream(self, **kwargs):  # type: ignore[override]
            return DummyStream("resp_1")

        # create() used for continuation
        def create(self, **kwargs):  # type: ignore[override]
            captured_calls.append(kwargs)
            # return a response that now has text so the agent can finish
            return DummyResponseText("resp_2")

    monkeypatch.setattr(main, "get_openai_client", lambda: DummyClient())

    brief = {
        "research_question": "RQ",
        "background_context": "背景",
        "deliverable_expectations": "期待",
        "subtopics": ["A"],
        "tone": "丁寧",
    }

    trace: list[main.TraceEntry] = []
    result = main._research_agent(brief, trace)

    assert result == "FINAL"
    # Ensure we did one continuation call and used a supported reasoning setting
    assert captured_calls, "No continuation call captured"
    cont = captured_calls[0]
    assert cont.get("previous_response_id") == "resp_1"
    # ユーザー要望により reasoning/text は送信しない
    assert "reasoning" not in cont
    assert "text" not in cont


def test_extract_answer_handles_summary_text() -> None:
    class DummyResponse:
        output_text = None

        def model_dump(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "summary_text", "text": "Summarized result"},
                            {"type": "output_text", "text": "Duplicate"},
                        ],
                    }
                ]
            }

    result = main._extract_answer(DummyResponse())
    assert result.startswith("Summarized result")


def test_extract_answer_recovers_from_object_output() -> None:
    class DummySegment:
        def __init__(self, text: str) -> None:
            self.type = "analysis"
            self.text = text

    class DummyMessage:
        def __init__(self, segments) -> None:
            self.type = "message"
            self.content = segments

    class DummyResponse:
        def __init__(self) -> None:
            self.output_text = ""
            self.output = [DummyMessage([DummySegment("最終結果: OK")])]

        def model_dump(self, **_: object):
            return {"output": []}

    result = main._extract_answer(DummyResponse())
    assert result == "最終結果: OK"


def test_extract_answer_handles_nested_report_structure() -> None:
    class DummyReport:
        def __init__(self) -> None:
            self.sections = [
                {
                    "title": "概要",
                    "paragraphs": [
                        {
                            "sentences": [
                                "1. 調査は成功しました。",
                                "2. 主要な結論: サンプル。",
                            ]
                        }
                    ],
                }
            ]

    class DummySegment:
        def __init__(self, report: DummyReport) -> None:
            self.type = "report"
            self.report = report

    class DummyMessage:
        def __init__(self, content) -> None:
            self.type = "message"
            self.content = content

    class DummyResponse:
        def __init__(self) -> None:
            self.output_text = None
            self.output = [DummyMessage([DummySegment(DummyReport())])]

    result = main._extract_answer(DummyResponse())
    assert "調査は成功しました" in result


def test_snapshot_rate_limits_formats_headers() -> None:
    headers = {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "0",
        "x-request-id": "req_test",
    }

    class DummyError(Exception):
        def __init__(self) -> None:
            self.response = type("Resp", (), {"headers": headers})()

    snapshot = main._snapshot_rate_limits(DummyError())
    assert snapshot is not None
    assert "x-ratelimit-limit-requests=60" in snapshot
    assert "x-ratelimit-remaining-requests=0" in snapshot
    assert "x-request-id=req_test" in snapshot
