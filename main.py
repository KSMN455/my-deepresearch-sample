from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, TypedDict

from flask import Flask, jsonify, render_template_string, request
from openai import OpenAI, OpenAIError

SYSTEM_PROMPT = (
    "You are an expert research assistant who uses the Deep Research API. "
    "Summarize findings clearly, cite important sources inline when provided, "
    "and answer concisely in Japanese unless the user explicitly asks for another language."
)

TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "gpt-4o-mini")
CLARIFIER_MODEL = os.getenv("CLARIFIER_MODEL", "gpt-4o-mini")
INSTRUCTION_MODEL = os.getenv("INSTRUCTION_MODEL", "gpt-4o-mini")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "o4-mini-deep-research-2025-06-26")


def _load_research_tools() -> list[dict[str, object]]:
    raw = os.getenv("DEEP_RESEARCH_TOOLS")
    if not raw:
        return [{"type": "web_search_preview"}]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
        raise RuntimeError("DEEP_RESEARCH_TOOLS must be valid JSON.") from exc
    if not isinstance(parsed, list):  # pragma: no cover - invalid config
        raise RuntimeError("DEEP_RESEARCH_TOOLS must be a JSON array.")
    return parsed


DEEP_RESEARCH_TOOLS = _load_research_tools()

LOGGER = logging.getLogger("deepresearch")

RATE_LIMIT_HEADER_KEYS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset-requests",
)


def configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


HTML_TEMPLATE = """<!doctype html>\n<html lang=\"ja\">\n<head>\n  <meta charset=\"utf-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n  <title>DeepResearch Sample</title>\n  <style>\n    :root {\n      color-scheme: light dark;\n      font-family: system-ui, -apple-system, BlinkMacSystemFont, \\"Segoe UI\\", sans-serif;\n    }\n    body {\n      margin: 0;\n      padding: 0 1rem;\n      background: #f5f5f5;\n    }\n    main {\n      max-width: 960px;\n      margin: 2rem auto;\n      background: #ffffffaa;\n      border-radius: 12px;\n      padding: 1.5rem;\n      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);\n    }\n    h1 {\n      margin-top: 0;\n      text-align: center;\n    }\n    #conversation {\n      display: flex;\n      flex-direction: column;\n      gap: 1rem;\n      margin: 1rem 0;\n      max-height: 60vh;\n      overflow-y: auto;\n      padding-right: 0.5rem;\n    }\n    .message {\n      border-radius: 10px;\n      padding: 0.75rem 0.9rem;\n      background: #ffffff;\n      border: 1px solid #d0d7ff;\n      color: #111827;\n      line-height: 1.6;\n    }\n    .message p,\n    .message pre {\n      margin: 0;\n    }\n    .message-user {\n      background: #1d4ed8;\n      border-color: #1d4ed8;\n      color: #f8fafc;\n      align-self: flex-end;\n    }\n    .message-assistant {\n      background: #7c3aed;\n      border-color: #7c3aed;\n      color: #fdf4ff;\n      align-self: flex-start;\n    }\n    .message-trace {\n      background: #0f172a;\n      border: 1px solid #38bdf8;\n      border-left-width: 6px;\n      align-self: stretch;\n      color: #e2e8f0;\n    }\n    .message-heading {\n      display: flex;\n      align-items: center;\n      gap: 0.5rem;\n      font-size: 0.9rem;\n      margin-bottom: 0.35rem;\n    }\n    .message-heading strong {\n      font-weight: 600;\n    }\n    .message-badge {\n      font-size: 0.7rem;\n      text-transform: uppercase;\n      letter-spacing: 0.04em;\n      background: rgba(17, 24, 39, 0.12);\n      padding: 0.15rem 0.45rem;\n      border-radius: 999px;\n      color: #0f172a;\n    }\n    .message-user .message-badge,\n    .message-assistant .message-badge {\n      background: rgba(248, 250, 252, 0.25);\n      color: inherit;\n    }\n    .message-trace .message-badge {\n      background: rgba(56, 189, 248, 0.35);\n      color: #e0f2fe;\n    }\n    .message-trace pre {\n      margin: 0;\n      white-space: pre-wrap;\n      word-break: break-word;\n      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;\n      font-size: 0.9rem;\n      color: inherit;\n    }\n    form {\n      display: flex;\n      flex-direction: column;\n      gap: 0.75rem;\n    }\n    textarea {\n      resize: vertical;\n      min-height: 120px;\n      padding: 0.75rem;\n      font-size: 1rem;\n      border-radius: 10px;\n      border: 1px solid #ccc;\n    }\n    button {\n      cursor: pointer;\n      border: none;\n      background: #4c6ef5;\n      color: #fff;\n      padding: 0.75rem 1.5rem;\n      font-size: 1rem;\n      border-radius: 999px;\n      transition: background 0.2s ease-in-out;\n    }\n    button:disabled {\n      cursor: wait;\n      opacity: 0.7;\n    }\n    footer {\n      margin-top: 1.5rem;\n      font-size: 0.85rem;\n      color: #555;\n      text-align: center;\n    }\n    .status {\n      min-height: 1.25rem;\n      color: #555;\n    }\n  </style>\n</head>\n<body>\n  <main>\n    <h1>DeepResearch Chat Demo</h1>\n    <p>調査したいテーマを入力すると、四段構成のエージェント (Triage → Clarifier → Instruction Builder → Research) が情報整理と補足質問を行った上で o4-mini-deep-research が調査します。</p>\n    <section id=\"conversation\" aria-live=\"polite\"></section>\n    <form id=\"chat-form\" autocomplete=\"off\">\n      <label for=\"message\">質問や調査テーマ</label>\n      <textarea id=\"message\" placeholder=\"例: 最新の大規模言語モデル評価指標の動向を調べて\" required></textarea>\n      <button type=\"submit\" id=\"submit-button\">調査する</button>\n      <div class=\"status\" id=\"status\"></div>\n    </form>\n    <footer>API Key は環境変数 <code>OPENAI_API_KEY</code> または <code>openai_api_key.txt</code> から読み込みます。</footer>\n  </main>\n  <script>\n    const conversation = [];\n    const conversationEl = document.getElementById('conversation');\n    const formEl = document.getElementById('chat-form');\n    const textareaEl = document.getElementById('message');\n    const statusEl = document.getElementById('status');\n    const buttonEl = document.getElementById('submit-button');\n\n    const ROLE_LABELS = { user: 'You', assistant: 'Researcher', trace: 'Pipeline' };\n    const ROLE_CLASSES = { user: 'message-user', assistant: 'message-assistant', trace: 'message-trace' };\n\n    function addMessage(role, text, options = {}) {\n      const { label, usePre = false, meta } = options;\n      const wrapper = document.createElement('div');\n      const roleClass = ROLE_CLASSES[role];\n      wrapper.className = ['message', roleClass].filter(Boolean).join(' ');\n\n      const heading = document.createElement('div');\n      heading.className = 'message-heading';\n      const titleEl = document.createElement('strong');\n      titleEl.textContent = label ?? ROLE_LABELS[role] ?? role;\n      heading.appendChild(titleEl);\n\n      if (meta) {\n        const badge = document.createElement('span');\n        badge.className = 'message-badge';\n        badge.textContent = meta;\n        heading.appendChild(badge);\n      }\n\n      const body = document.createElement(usePre ? 'pre' : 'p');\n      body.textContent = text;\n\n      wrapper.appendChild(heading);\n      wrapper.appendChild(body);\n      conversationEl.appendChild(wrapper);\n      conversationEl.scrollTop = conversationEl.scrollHeight;\n    }\n\n    function addTrace(entry) {\n      if (!entry || !entry.content) {\n        return;\n      }\n      const labelParts = [];\n      if (entry.stage) labelParts.push(entry.stage);\n      if (entry.title) labelParts.push(entry.title);\n      const label = labelParts.join(' · ') || 'Pipeline';\n      const meta = entry.kind ? entry.kind.toUpperCase() : undefined;\n      addMessage('trace', entry.content, { label, usePre: true, meta });\n    }\n\n    async function submitPrompt(event) {\n      event.preventDefault();\n      const content = textareaEl.value.trim();\n      if (!content) {\n        return;\n      }\n\n      textareaEl.value = '';\n      textareaEl.focus();\n      buttonEl.disabled = true;\n      statusEl.textContent = '調査パイプラインを実行しています…';\n      addMessage('user', content);\n      conversation.push({ role: 'user', content });\n\n      try {\n        const response = await fetch('/chat', {\n          method: 'POST',\n          headers: { 'Content-Type': 'application/json' },\n          body: JSON.stringify({ messages: conversation }),\n        });\n\n        if (!response.ok) {\n          throw new Error(`Request failed with status ${response.status}`);\n        }\n\n        const payload = await response.json();\n        const traceEntries = Array.isArray(payload.trace) ? payload.trace : [];\n        traceEntries.forEach(addTrace);\n        const answer = payload.answer ?? '回答を取得できませんでした。';\n        addMessage('assistant', answer);\n        conversation.push({ role: 'assistant', content: answer });\n        statusEl.textContent = '';\n      } catch (error) {\n        console.error(error);\n        statusEl.textContent = 'エラーが発生しました。しばらく経ってから再度お試しください。';\n      } finally {\n        buttonEl.disabled = false;\n      }\n    }\n\n    formEl.addEventListener('submit', submitPrompt);\n  </script>\n</body>\n</html>\n"""


class ConversationMessage(TypedDict):
    role: str
    content: str


class TriageDecision(TypedDict, total=False):
    decision: Literal["clarify", "proceed"]
    missing_information: str
    summary: str


class ClarifierPlan(TypedDict, total=False):
    question: str
    rationale: str


class InstructionBrief(TypedDict, total=False):
    research_question: str
    background_context: str
    deliverable_expectations: str
    subtopics: list[str]
    tone: str


TraceKind = Literal["system", "user", "response", "info", "debug"]


class TraceEntry(TypedDict, total=False):
    stage: str
    kind: TraceKind
    title: str
    content: str


TraceLog = list[TraceEntry]


def _log_trace(
    trace: TraceLog,
    stage: str,
    *,
    kind: TraceKind,
    title: str,
    content: str,
) -> None:
    """Append a structured trace entry to help debug the four-agent pipeline.

    四段のエージェントパイプラインを後から追跡しやすくするためのトレースを記録する。
    """

    trace.append(
        {
            "stage": stage,
            "kind": kind,
            "title": title,
            "content": content,
        }
    )


def _log_prompts(
    trace: TraceLog,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    *,
    user_title: str = "User Prompt",
) -> None:
    """Record the prompts sent to a model so we can replay pipeline decisions later.

    モデルに送ったプロンプトを記録して、後で意思決定の流れを再現できるようにする。
    """

    _log_trace(trace, stage, kind="system", title="System Prompt", content=system_prompt)
    _log_trace(trace, stage, kind="user", title=user_title, content=user_prompt)


def load_api_key(
    env: Mapping[str, str] | None = None,
    key_file: str | Path | None = None,
) -> str:
    env_mapping = env if env is not None else os.environ
    env_value = env_mapping.get("OPENAI_API_KEY", "").strip()
    if env_value:
        LOGGER.debug("Using API key from environment variable")
        return env_value

    candidate = (
        Path(key_file)
        if key_file is not None
        else Path(__file__).resolve().parent / "openai_api_key.txt"
    )
    if candidate.exists():
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            LOGGER.debug("Using API key from %s", candidate)
            return stripped

    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or create openai_api_key.txt with the key."
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    max_retries_env = os.getenv("OPENAI_MAX_RETRIES")
    kwargs: dict[str, object] = {"api_key": load_api_key()}
    if max_retries_env is not None:
        try:
            kwargs["max_retries"] = int(max_retries_env)
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid OPENAI_MAX_RETRIES=%r (expected integer)",
                max_retries_env,
            )
    return OpenAI(**kwargs)


def _normalise_messages(messages: Iterable[Mapping[str, str]]) -> list[ConversationMessage]:
    """Validate the chat payload shape and strip whitespace for model readiness.

    モデルに投入しやすいように会話メッセージを検証し、余分な空白を削除する。
    """

    normalised: list[ConversationMessage] = []
    for item in messages:
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role in conversation payload: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Conversation entries must include non-empty string content.")
        normalised.append({"role": role, "content": content.strip()})
    return normalised


def _extract_answer(response: object) -> str:
    """Pull human-readable text from SDK responses, handling many output formats.

    SDK から返るさまざまな形式のレスポンスから、人間が読めるテキストを取り出す。
    """

    fragments: list[str] = []

    def _append_text(value: object) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                fragments.append(stripped)

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(output_text, SequenceABC) and not isinstance(
        output_text, (str, bytes, bytearray)
    ):
        for item in output_text:
            _append_text(item)
        if fragments:
            deduped = list(dict.fromkeys(fragments))
            return "\n\n".join(deduped)

    # Track visited objects to avoid infinite recursion on cyclic structures.
    # 循環参照が存在しても再帰が無限に続かないようにする。
    visited: set[int] = set()

    def _collect(node: object) -> None:
        if node is None:
            return
        if isinstance(node, str):
            _append_text(node)
            return
        if isinstance(node, (bytes, bytearray)):
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, MappingABC):
            for key, value in node.items():
                key_lower = str(key).lower()
                if isinstance(value, str):
                    if key_lower not in {
                        "id",
                        "type",
                        "status",
                        "role",
                        "name",
                        "call_id",
                        "url",
                        "pattern",
                        "query",
                        "server_label",
                        "model",
                    }:
                        _append_text(value)
                    continue

                if (
                    key_lower.endswith("text")
                    or "summary" in key_lower
                    or key_lower
                    in {
                        "report",
                        "content",
                        "answer",
                        "message",
                        "result",
                        "body",
                        "sections",
                        "paragraphs",
                        "bullets",
                        "sentences",
                        "analysis",
                        "entries",
                        "steps",
                        "outputs",
                    }
                ):
                    _collect(value)
                elif key_lower not in {"id", "type", "status", "role"}:
                    _collect(value)
            return

        if isinstance(node, SequenceABC) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                _collect(item)
            return

        for attr_name in (
            "text",
            "summary",
            "content",
            "report",
            "answer",
            "result",
            "message",
            "body",
        ):
            if not hasattr(node, attr_name):
                continue
            value = getattr(node, attr_name)
            if attr_name == "text":
                _append_text(value)
            else:
                _collect(value)

        dump_method = getattr(node, "model_dump", None)
        if callable(dump_method):
            try:
                dumped = dump_method(warnings=False)
            except TypeError:
                dumped = dump_method()
            except Exception:
                dumped = None
            if dumped is not None:
                _collect(dumped)
                return

        if hasattr(node, "__dict__"):
            _collect(vars(node))

    _collect(getattr(response, "output", None))

    if not fragments:
        _collect(getattr(response, "output_parsed", None))

    if not fragments:
        dump_method = getattr(response, "model_dump", None)
        if callable(dump_method):
            try:
                _collect(dump_method(warnings=False))
            except Exception:
                pass

    if not fragments:
        _collect(response)

    if not fragments:
        raise RuntimeError("OpenAI response did not include textual output.")

    deduped = list(dict.fromkeys(fragments))
    return "\n\n".join(deduped)


def _format_conversation(messages: Sequence[ConversationMessage], limit: int = 12) -> str:
    recent = messages[-limit:]
    lines = []
    for message in recent:
        prefix = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }.get(message["role"], message["role"])
        lines.append(f"{prefix}: {message['content']}")
    return "\n".join(lines)


def _input_text_block(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def _complete(model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
    client = get_openai_client()
    LOGGER.debug(
        "Calling model %s with options %s", model, {k: v for k, v in kwargs.items() if k != "input"}
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    _input_text_block(system_prompt),
                ],
            },
            {
                "role": "user",
                "content": [
                    _input_text_block(user_prompt),
                ],
            },
        ],
        **kwargs,
    )
    text = _extract_answer(response)
    LOGGER.debug("Model %s raw output: %s", model, text)
    return text


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```", 2)
        if len(parts) >= 2:
            stripped = parts[1]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()


def _snapshot_rate_limits(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        items = {key: headers.get(key) for key in RATE_LIMIT_HEADER_KEYS if headers.get(key)}
    except Exception:  # pragma: no cover - defensive against unexpected header types
        return None
    if not items:
        return None
    formatted = ", ".join(f"{key}={value}" for key, value in items.items())
    request_id = headers.get("x-request-id")
    if request_id:
        formatted = f"{formatted}, x-request-id={request_id}"
    return formatted


def _parse_json_response(raw: str) -> dict[str, object]:
    """Parse agent output, tolerating Markdown fences or stray text around the JSON.

    マークダウンのフェンスや JSON 外のノイズが混ざっていてもパースできるようにする。
    """

    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _call_json_agent(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    agent_name: str,
) -> dict[str, object]:
    raw = _complete(model, system_prompt, user_prompt)
    try:
        return _parse_json_response(raw)
    except json.JSONDecodeError as exc:
        LOGGER.error("%s agent failed to return JSON: %s", agent_name, raw)
        raise RuntimeError(f"{agent_name} が JSON 形式で応答しませんでした。") from exc


def _triage_agent(
    conversation: Sequence[ConversationMessage],
    trace: TraceLog,
) -> TriageDecision:
    """Decide whether to run research or ask the user for more context.

    調査を継続するか、ユーザーへ追加のコンテキスト確認を促すかを判定する。
    """

    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Latest conversation entry must be a user message.")

    transcript = _format_conversation(conversation)
    system_prompt = (
        "You are the Triage Agent in a Four-Agent Deep Research pipeline. "
        "Inspect the conversation, decide if the latest user query has enough context for research, "
        "and respond in strict JSON with keys decision, missing_information, summary. "
        "Use decision='clarify' if more details are required, otherwise 'proceed'. "
        "Provide concise Japanese text for missing_information and summary."
    )
    user_prompt = (
        "これまでの会話:\n"
        f"{transcript}\n\n"
        "最新のユーザークエリに対して調査を進める準備が整っているか評価してください。"
        "必要な場合は不足しているコンテキストをまとめてください。"
        "JSON 形式のみで回答してください。例:\n"
        '{"decision": "clarify", "missing_information": "...", "summary": "..."}'
    )
    stage = "Triager"
    _log_prompts(trace, stage, system_prompt, user_prompt)
    data = _call_json_agent(
        TRIAGE_MODEL,
        system_prompt,
        user_prompt,
        agent_name="Triager",
    )

    decision = data.get("decision")
    if decision not in {"clarify", "proceed"}:
        raise RuntimeError("Triager の decision フィールドが不正です。")

    decision_payload = {
        "decision": decision,
        "missing_information": data.get("missing_information", "").strip(),
        "summary": data.get("summary", "").strip(),
    }
    _log_trace(
        trace,
        stage,
        kind="response",
        title="Decision",
        content=json.dumps(decision_payload, ensure_ascii=False, indent=2),
    )
    LOGGER.info("Triager decision: %s", decision_payload["decision"])
    LOGGER.debug("Triager details: %s", decision_payload)
    return decision_payload


def _clarifier_agent(
    conversation: Sequence[ConversationMessage],
    triage: TriageDecision,
    trace: TraceLog,
) -> ClarifierPlan:
    """Draft a single follow-up question to resolve missing context.

    不足している情報を補うために、フォローアップ質問を 1 件だけ生成する。
    """

    transcript = _format_conversation(conversation)
    missing_info = triage.get("missing_information", "")
    system_prompt = (
        "You are the Clarifier Agent in a Four-Agent Deep Research pipeline. "
        "Create at most one concise follow-up question in Japanese to obtain missing context. "
        "Return strict JSON with keys question (required) and rationale (optional)."
    )
    user_prompt = (
        "不足している情報: "
        f"{missing_info or '特になし'}\n"
        "会話ログ:\n"
        f"{transcript}\n\n"
        "Clarifier としてユーザーに尋ねるべき質問を 1 つ提案してください。"
        '回答は JSON のみ: {"question": "...", "rationale": "..."}'
    )
    stage = "Clarifier"
    _log_prompts(trace, stage, system_prompt, user_prompt)
    data = _call_json_agent(
        CLARIFIER_MODEL,
        system_prompt,
        user_prompt,
        agent_name="Clarifier",
    )

    question = data.get("question", "").strip()
    if not question:
        raise RuntimeError("Clarifier の質問が空でした。")

    clarifier_response = {
        "question": question,
        "rationale": data.get("rationale", "").strip(),
    }
    _log_trace(
        trace,
        stage,
        kind="response",
        title="Clarifier Output",
        content=json.dumps(clarifier_response, ensure_ascii=False, indent=2),
    )
    LOGGER.info("Clarifier generated follow-up question")
    LOGGER.debug("Clarifier details: %s", clarifier_response)
    return clarifier_response


def _instruction_builder_agent(
    conversation: Sequence[ConversationMessage],
    triage: TriageDecision,
    trace: TraceLog,
) -> InstructionBrief:
    """Transform conversation context into a structured brief for the researcher.

    会話とトリアージ結果を元に、リサーチャー向けの構造化ブリーフへ変換する。
    """

    transcript = _format_conversation(conversation)
    user_query = conversation[-1]["content"]
    summary = triage.get("summary", "")
    system_prompt = (
        "You are the Instruction Builder Agent. "
        "Transform the conversation and triage summary into a structured research brief for the Research Agent. "
        "Respond in JSON with keys research_question, background_context, deliverable_expectations, subtopics (array of strings), tone. "
        "Use clear Japanese."
    )
    user_prompt = (
        "会話ログ:\n"
        f"{transcript}\n\n"
        f"Triager の要約: {summary or '未提供'}\n"
        f"最新のユーザー要求: {user_query}\n\n"
        "Research Agent が効率的に調査できるように指示書を作成してください。"
    )
    stage = "Instruction Builder"
    _log_prompts(trace, stage, system_prompt, user_prompt)
    data = _call_json_agent(
        INSTRUCTION_MODEL,
        system_prompt,
        user_prompt,
        agent_name="Instruction Builder",
    )

    research_question = data.get("research_question", "").strip()
    deliverable = data.get("deliverable_expectations", "").strip()
    if not research_question or not deliverable:
        raise RuntimeError("Instruction Builder の出力が不完全です。")

    subtopics = data.get("subtopics") or []
    if not isinstance(subtopics, list):
        raise RuntimeError("Instruction Builder の subtopics は配列である必要があります。")

    brief = {
        "research_question": research_question,
        "background_context": data.get("background_context", "").strip(),
        "deliverable_expectations": deliverable,
        "subtopics": [str(item).strip() for item in subtopics if str(item).strip()],
        "tone": data.get("tone", "日本語で簡潔にまとめる"),
    }
    _log_trace(
        trace,
        stage,
        kind="response",
        title="Research Brief",
        content=json.dumps(brief, ensure_ascii=False, indent=2),
    )
    LOGGER.info(
        "Instruction Builder produced research brief with %d subtopics", len(brief["subtopics"])
    )
    LOGGER.debug("Instruction Builder brief: %s", brief)
    return brief


def _build_research_prompt(brief: InstructionBrief) -> str:
    """Convert the instruction brief into the prompt expected by the research model.

    インストラクションブリーフをリサーチモデルが解釈しやすいプロンプトに整形する。
    """

    agenda = (
        "\n".join(f"- {item}" for item in brief.get("subtopics", []))
        or "- 重要な小項目を自ら特定してください"
    )
    background = brief.get("background_context") or "追加の背景情報は提供されていません。"
    return (
        "以下の指示に従ってオンライン調査を実施し、引用を付与した要約を作成してください。\n"
        f"主題: {brief['research_question']}\n"
        f"背景: {background}\n"
        f"期待する成果物: {brief['deliverable_expectations']}\n"
        f"推奨サブトピック:\n{agenda}\n"
        f"口調: {brief.get('tone', '日本語で簡潔かつ洞察的')}\n"
        "必要に応じて追加のサブトピックを自律的に検討して構いません。"
    )


def _research_agent(brief: InstructionBrief, trace: TraceLog) -> str:
    prompt = _build_research_prompt(brief)
    client = get_openai_client()
    LOGGER.info("Invoking research model %s", RESEARCH_MODEL)
    stage = "Research"

    _log_trace(trace, stage, kind="system", title="System Prompt", content=SYSTEM_PROMPT)
    _log_trace(trace, stage, kind="user", title="Instruction Builder Prompt", content=prompt)

    # 調整可能なパラメータ（環境変数でオーバーライドしてもOK）
    max_tokens = int(os.getenv("RESEARCH_MAX_OUTPUT_TOKENS", "2000"))
    continue_steps = int(os.getenv("RESEARCH_CONTINUE_STEPS", "3"))

    def _log_raw(label: str, resp: object) -> None:
        try:
            raw = resp.to_json(indent=2, warnings=False)
        except Exception:
            raw = repr(resp)
        _log_trace(trace, stage, kind="debug", title=label, content=raw)
        LOGGER.info("%s:\n%s", label, raw)

    response_id = None
    completed = False
    streamed_text_chunks: list[str] = []

    # --- ストリーミング開始 ---
    try:
        with client.responses.stream(
            model=RESEARCH_MODEL,
            input=[
                {"role": "system", "content": [_input_text_block(SYSTEM_PROMPT)]},
                {"role": "user", "content": [_input_text_block(prompt)]},
            ],
            reasoning={"effort": "medium"},
            max_output_tokens=max_tokens,
            tools=DEEP_RESEARCH_TOOLS,
            tool_choice="auto",
        ) as stream:
            for event in stream:
                et = getattr(event, "type", "")
                # 1) レスポンスIDの確保
                if et == "response.created":
                    try:
                        response_id = event.response.id
                    except Exception:
                        pass

                # 2) 逐次テキストの収集（必要に応じて UI に反映も可）
                elif et == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        streamed_text_chunks.append(delta)

                # 3) ツールや検索ステップのログ
                elif et in (
                    "response.tool_call.delta",
                    "response.tool_call.completed",
                    "response.web_search_call",
                    "response.web_browse_call",
                ):
                    try:
                        _log_trace(
                            trace,
                            stage,
                            kind="debug",
                            title=f"Stream Event: {et}",
                            content=event.model_dump_json(),
                        )
                    except Exception:
                        pass

                # 4) 完了フラグ
                elif et == "response.completed":
                    completed = True

            # すべてのイベント受信後
            if completed:
                response = stream.get_final_response()
                _log_raw("Research Raw Response (final)", response)
            else:
                # completed が来なかった場合のフォールバック
                if response_id:
                    response = client.responses.create(
                        model=RESEARCH_MODEL,
                        previous_response_id=response_id,
                        input=[  # ★ 続きを促す最小のユーザー入力を必ず送る
                            {
                                "role": "user",
                                "content": [
                                    _input_text_block(
                                        "続きの調査を完了し、最終の要約と引用付きの結論を出力してください。"
                                    )
                                ],
                            }
                        ],
                        max_output_tokens=max(800, max_tokens // 2),
                        tools=DEEP_RESEARCH_TOOLS,
                        tool_choice="auto",
                    )
                    _log_raw("Research Raw Response (continue after missing completed)", response)
                else:
                    # streamed_text に本文が入っていればそれを返す
                    if streamed_text_chunks:
                        provisional = "".join(streamed_text_chunks).strip()
                        _log_trace(
                            trace,
                            stage,
                            kind="response",
                            title="Research Result (provisional from stream)",
                            content=provisional,
                        )
                        return provisional
                    raise RuntimeError(
                        "Didn't receive a `response.completed` event and no response_id."
                    )
    except OpenAIError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        snapshot = _snapshot_rate_limits(exc)
        if status_code == 429 and snapshot:
            LOGGER.warning("Rate limit snapshot during research call: %s", snapshot)
        raise

    # ここまで来たら response オブジェクトはある想定。抽出して返す。
    try:
        answer = _extract_answer(response)
        _log_trace(trace, stage, kind="response", title="Research Result", content=answer)
        return answer
    except RuntimeError:
        # さらに “続き” 呼び出しをリトライ（数回）
        current = response
        for i in range(continue_steps):
            current = client.responses.create(
                model=RESEARCH_MODEL,
                previous_response_id=current.id,
                input=[  # ★ 続きを促す最小のユーザー入力を必ず送る
                    {
                        "role": "user",
                        "content": [
                            _input_text_block(
                                "続きの調査を完了し、最終の要約と引用付きの結論を出力してください。"
                            )
                        ],
                    }
                ],
                max_output_tokens=max(800, max_tokens // 2),
                tools=DEEP_RESEARCH_TOOLS,
                tool_choice="auto",
            )
            _log_raw(f"Research Raw Response (retry continue {i + 1})", current)
            try:
                answer = _extract_answer(current)
                _log_trace(
                    trace,
                    stage,
                    kind="response",
                    title=f"Research Result (after continue {i + 1})",
                    content=answer,
                )
                return answer
            except RuntimeError:
                continue

        # 最後の手段：ストリームで取れた断片を返す
        if streamed_text_chunks:
            provisional = "".join(streamed_text_chunks).strip()
            if provisional:
                _log_trace(
                    trace,
                    stage,
                    kind="response",
                    title="Research Result (fallback provisional)",
                    content=provisional,
                )
                return provisional

        # デバッグ情報を残して失敗
        try:
            dumped = current.to_json(warnings=False)
        except Exception:
            dumped = {}
        tool_steps = []
        try:
            for block in dumped.get("output") or []:
                if block.get("type") in {"web_search_call", "web_browse_call", "tool_call"}:
                    tool_steps.append({k: block.get(k) for k in ("type", "status", "action", "id")})
        except Exception:
            pass
        _log_trace(
            trace,
            stage,
            kind="info",
            title="No Final Text – Tool Steps Snapshot",
            content=json.dumps(tool_steps[:5], ensure_ascii=False, indent=2),
        )
        raise RuntimeError("OpenAI response did not include textual output.")


def run_deep_research(messages: Iterable[Mapping[str, str]]) -> dict[str, object]:
    """Execute the four-agent workflow and return the answer plus a debug trace.

    四つのエージェントを順番に実行し、応答とトレースログをまとめて返す。
    """

    normalised = _normalise_messages(messages)
    LOGGER.debug("Running deep research pipeline with %d messages", len(normalised))
    trace: TraceLog = []
    triage = _triage_agent(normalised, trace)
    if triage["decision"] == "clarify":
        clarifier = _clarifier_agent(normalised, triage, trace)
        answer = clarifier["question"]
        _log_trace(
            trace,
            "Clarifier",
            kind="info",
            title="Clarifier Question Sent to User",
            content=answer,
        )
        return {"answer": f"Clarifier Agent: {answer}", "trace": trace}

    brief = _instruction_builder_agent(normalised, triage, trace)
    answer = _research_agent(brief, trace)
    return {"answer": answer, "trace": trace}


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.post("/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            LOGGER.warning("/chat called with non-list messages")
            return jsonify({"error": "messages must be a list"}), 400
        try:
            LOGGER.info("/chat request with %d messages", len(messages))
            result = run_deep_research(messages)
        except ValueError as exc:
            LOGGER.warning("Validation error: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except OpenAIError as exc:  # pragma: no cover - network error path
            LOGGER.exception("OpenAI API error during deep research")
            return jsonify({"error": "OpenAI API error", "detail": str(exc)}), 502
        except RuntimeError as exc:
            LOGGER.exception("Runtime error during deep research pipeline")
            return jsonify({"error": str(exc)}), 500
        return jsonify(result)

    return app


def main() -> None:
    configure_logging()
    app = create_app()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
