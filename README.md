# DeepResearch Sample

Flask ベースの簡易的なチャット UI から OpenAI の `o4-mini-deep-research` モデルを呼び出し、オンライン調査の結果を表示するサンプルアプリケーションです。Deep Research API の基本的な呼び出しフローと、会話履歴を引き渡す方法を確認できます。

## セットアップ

1. Python 3.13 を用意します。
2. [uv](https://docs.astral.sh/uv/) をインストールしていることを確認します。
3. 依存関係を同期します。
   ```bash
   uv sync
   ```
4. OpenAI API Key を環境変数 `OPENAI_API_KEY` に設定するか、プロジェクト直下に `openai_api_key.txt` を配置します。
   深掘りモデルはウェブ検索などのツール指定が必須のため、必要に応じて `DEEP_RESEARCH_TOOLS`（JSON 文字列）を設定できます。デフォルトは `[{"type": "web_search_preview"}]` です。
   （従来の `pip install -e .` を使う場合は適宜仮想環境を作成してください。）

## アプリの起動

```bash
uv run python main.py
```

ブラウザで <http://127.0.0.1:5000/> にアクセスするとチャット画面が表示されます。リクエスト時には Deep Research API が必要に応じてウェブ検索を実行し、結果が返信として返ってきます。

## 使い方

1. フォームに調べたいテーマや質問を日本語で入力します。
2. 「調査する」ボタンを押すと会話履歴ごと API に送信され、`o4-mini-deep-research` がオンライン調査を実行します。
3. モデルから返ってきた要約が画面に表示され、以降のメッセージでは前回までの履歴が引き継がれます。
4. エラーが表示された場合は、API キーが設定されているかと、再試行まで少し時間を置くことを確認してください。

## 設計

- **フロントエンド**: Flask の `render_template_string` で提供する単一ページ。JavaScript で会話履歴を管理し、`/chat` に JSON を POST して応答を描画します。
- **API 層**: `/chat` エンドポイントが JSON を受け取り、会話履歴をサニタイズした上で `run_deep_research` に委譲します。不正な入力は 400 番台のレスポンスで弾きます。
- **OpenAI クライアント**: `load_api_key` が環境変数または `openai_api_key.txt` からキーを読み込み、一度生成した `OpenAI` クライアントを `lru_cache` で再利用します。
- **四段エージェントパイプライン**: `_triage_agent` → `_clarifier_agent` → `_instruction_builder_agent` → `_research_agent` の順で制御します。Triager が不足情報を検出した場合は Clarifier が質問のみ返し、Instruction Builder が構造化ブリーフを生成できた時点で `o4-mini-deep-research` を起動します。
- **Deep Research 呼び出し**: Research Agent の仕上げとして `o4-mini-deep-research` に system プロンプトと Instruction Builder が組み立てた指示を渡し、`web_search_preview` ツール（または `DEEP_RESEARCH_TOOLS` 環境変数で指定したツール）を添えて API 要件を満たします。レスポンスは `_extract_answer` で正規化して UI に返します。
- **テスト**: Pytest で API キーの読み込み、バリデーション、HTTP エンドポイントの動作をカバーします。ネットワークコールは `monkeypatch` で差し替えたフェイクを使用します。

## OpenAI クライアントの使い方

`main.py` では最新の OpenAI Python SDK を利用しており、`OpenAI` クラスをインスタンス化した後に `responses.create` エンドポイントを呼び出します。以下が基本的な流れです。

1. `OPENAI_API_KEY` を環境変数に入れるか、`openai_api_key.txt` に保存します。必要に応じて `OPENAI_MAX_RETRIES` でリトライ回数を整数で指定できます。
2. `get_openai_client()` を呼ぶと、API キーを読み込んだ `OpenAI` クライアントが初期化され、`lru_cache` によりプロセス内で再利用されます。
3. `client.responses.create` に system / user ロールのメッセージを JSON 形式で渡します。Deep Research モデルを利用するときは `tools` に `[{"type": "web_search_preview"}]` などのツール一覧を指定します。

例えば `_complete` ヘルパーが行っている最小の呼び出しは次のとおりです。

```python
from main import get_openai_client

client = get_openai_client()
response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": [{"type": "input_text", "text": "..."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "..."}]},
    ],
)
answer = response.output_text  # `_extract_answer` で複数チャンクを統合
```

`_research_agent` では上記に加え Deep Research 用の system プロンプトと詳細な指示を構築し、`reasoning={"effort": "medium"}` や `max_output_tokens` などの追加パラメータを付与しています。これらは OpenAI SDK にそのまま辞書で渡すだけで適用できます。

## テスト

```bash
uv run pytest -q
```

## 参考資料
- [OpenAI Platform Docs – Deep Research](https://platform.openai.com/docs/guides/deep-research)
- [Model Card – o4-mini-deep-research](https://platform.openai.com/docs/models/o4-mini-deep-research)
- [OpenAI Cookbook – Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents)
