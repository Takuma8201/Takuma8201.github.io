import os
import json
import re
from collections import Counter

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from openai import OpenAI
import httpx
from google import genai  # Gemini 用

# =========================
# 設定（文字数制限）
# =========================
QUESTION_MAX_CHARS = 200    # 入力は200文字まで
ANSWER_MAX_CHARS = 300      # 各回答は300文字までに丸める

# =========================
# .env 読み込み & APIキー
# =========================
load_dotenv()

openai_api_key    = (os.getenv("OPENAI_API_KEY") or "").strip()
anthropic_api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
gemini_api_key    = (os.getenv("GEMINI_API_KEY") or "").strip()

# Mistral だけは、コピー時に付いた " や ' を削ぎ落とす
mistral_api_key_raw = os.getenv("MISTRAL_API_KEY") or ""
mistral_api_key = mistral_api_key_raw.strip().strip('"').strip("'")

# デバッグ用（動作確認が済んだらコメントアウトしてOK）
print("MISTRAL_KEY_LEN  =", len(mistral_api_key))
print("MISTRAL_KEY_HEAD =", repr(mistral_api_key[:8]))

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY が設定されていません。.env を確認してください。")

if not anthropic_api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が設定されていません。.env を確認してください。")

if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY が設定されていません。.env を確認してください。")

if not mistral_api_key:
    raise RuntimeError("MISTRAL_API_KEY が設定されていません。.env を確認してください。")

# =========================
# クライアント & FastAPI本体
# =========================
client = OpenAI(api_key=openai_api_key)
gemini_client = genai.Client(api_key=gemini_api_key)
app = FastAPI()


class AskRequest(BaseModel):
    """
    フロントから送られてくるJSON:
    {
      "question": "……",
      "models": {
        "gpt4o": true/false,
        "gpt41": true/false,
        "claude": true/false,
        "gemini": true/false,
        "mistral": true/false
      },
      "need_summary": true/false,
      "genre": "numeric" | "history" | "opinion" | "trivia"   ← 未指定なら numeric
    }
    """
    question: str
    models: dict | None = None
    need_summary: bool = True
    genre: str | None = "numeric"


# =========================
# 共通：文字数を丸める関数
# =========================
def clamp_text(text: str, max_chars: int = ANSWER_MAX_CHARS) -> str:
    """
    テキストを max_chars 文字で切って「…」を付ける
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


# =========================
# OpenAI に質問する関数
# =========================
def ask_openai(model_name: str, question: str) -> str:
    """
    指定した OpenAI モデルに質問して回答テキストを返す
    """
    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "日本語で分かりやすく、できるだけ短く答えてください。"},
            {"role": "user", "content": question},
        ],
        max_tokens=300,
    )
    text = res.choices[0].message.content
    return clamp_text(text)


# =========================
# Gemini に質問する関数
# =========================
def ask_gemini(question: str) -> str:
    """
    Google Gemini に質問して回答テキストを返す
    """
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=question,
        )
        text = resp.text
    except Exception as e:
        text = f"Gemini でエラーが発生しました: {e}"

    return clamp_text(text)


# =========================
# Mistral に質問する関数（HTTP直叩き）
# =========================
def ask_mistral(question: str) -> str:
    """
    Mistral (mistral-small-latest) に質問して回答テキストを返す
    """
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": question}
        ],
    }

    try:
        with httpx.Client(timeout=20.0) as http_client:
            resp = http_client.post(url, headers=headers, json=payload)
            # デバッグ用（401などのとき中身を確認できるように）
            print("MISTRAL_STATUS =", resp.status_code)
            print("MISTRAL_BODY   =", resp.text[:200])
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        text = f"Mistral でエラーが発生しました: {e}"

    return clamp_text(text)


# =========================
# 解析用にテキストを圧縮（トークン節約）
# =========================
def compress_for_analysis(all_answers: dict, genre: str) -> dict:
    """
    トークン節約用に、GPTに送る文章を削減する。

    - summary 用: 各回答の先頭 120 文字
    - contradictions 用:
        * numeric/history/opinion: 数値（％・回数・年など）を含む文だけ
        * trivia: 基本的に空（矛盾検出は簡略化）
    """
    genre = (genre or "numeric").lower()

    summary_parts: list[str] = []
    contradiction_parts: list[str] = []

    for model_name, text in all_answers.items():
        short = text[:120]

        # 要約用
        summary_parts.append(f"[{model_name}]\n{short}")

        # 矛盾用：ジャンルによって絞り方を変える
        if genre in ("numeric", "history", "opinion"):
            # 数字や％を含む文をピックアップ
            numeric_sentences = re.findall(r"[^。]*[0-9０-９％%][^。]*。", text)
            if numeric_sentences:
                contradiction_parts.append(
                    f"[{model_name}]\n" + "\n".join(numeric_sentences)
                )
        else:
            # trivia など → 基本的に矛盾検出は行わない（空のまま）
            pass

    return {
        "summary_text": "\n\n".join(summary_parts),
        "contradiction_text": "\n\n".join(contradiction_parts),
    }


# =========================
# 複数回答を OpenAI に渡して
# 要約 & 矛盾点 を作ってもらう（節約版）
# =========================
def analyze_answers_with_openai(all_answers: dict, genre: str) -> dict:
    """
    トークン節約版：
    - summary: 各回答の先頭120文字をもとに全体要約
    - contradictions: 数値を含む文だけをもとに矛盾検出
    """
    compressed = compress_for_analysis(all_answers, genre)
    summary_text = compressed["summary_text"]
    contradiction_text = compressed["contradiction_text"]

    # trivia などで矛盾検出テキストが空のときの扱い
    contradiction_note = ""
    if not contradiction_text.strip():
        contradiction_note = (
            "なお、矛盾検出用に渡されたテキストはほとんど空かごく短いため、"
            "数値や事実関係の矛盾があっても検出できない可能性があります。"
            "その場合は「大きな矛盾はありません」と簡潔にまとめてください。"
        )

    user_prompt = f"""
以下は複数AIの「圧縮済み回答」です。

【要約用テキスト】
{summary_text}

【矛盾検出用テキスト（数値や年などを含む文のみ）】
{contradiction_text}

ジャンル: {genre}

これをもとに、次の2つを日本語で作成し、必ず JSON 形式だけで出力してください。

1. summary:
    - 全体としての重要なポイントだけをまとめた要約（2〜3文程度）
    - 冗長な言い換えは避け、短く要点だけを書く

2. contradictions:
    - 数値・割合・頻度・確率・年号などに関する食い違いがあれば、簡潔に指摘する
    - 例: 「gpt4o_mini は '快晴の日は1割以下' と述べているが、gpt4_1_mini は '1%以下' と述べており、桁が異なる」
    - 本当に目立った矛盾が見当たらない場合のみ「大きな矛盾はありません」と書く

{contradiction_note}

出力形式（これ「だけ」を返してください）:
{{
  "summary": "ここに要約",
  "contradictions": "ここに矛盾の説明（短く）"
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "あなたは複数回答の要約と矛盾検出を行うアシスタントです。"
                           "出力は必ず有効なJSONだけにし、文章はできるだけ短くしてください。",
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=350,  # 通常より少なめにして節約
    )

    content = res.choices[0].message.content

    try:
        data = json.loads(content)
        return data
    except Exception:
        return {
            "summary": "要約の解析に失敗しました。",
            "contradictions": "矛盾点の解析に失敗しました。",
        }


# =========================
# 頻出単語を Python 側で集計
# =========================
def extract_frequent_words(all_answers: dict) -> str:
    """
    頻出単語を Python で抽出して
    3回以上の単語のみ「回数ごとにグループ」で表示する。
    （トークン消費なし。ローカル処理のみ）
    """
    text = " ".join(all_answers.values())

    # 単語っぽい部分だけを抽出
    words = re.findall(r"[一-龥ぁ-んァ-ンa-zA-Z0-9]+", text)

    # 不要な語を除外
    stop_words = {
        "は", "が", "の", "に", "を", "で", "て", "と",
        "する", "ます", "です", "ある", "ない", "これ", "それ",
        "ため", "こと", "よう"
    }

    filtered = [w for w in words if w not in stop_words and len(w) > 1]

    counter = Counter(filtered)

    # 3回以上のみ抽出
    picked = [(w, c) for w, c in counter.items() if c >= 3]

    if not picked:
        return "3回以上使われた重要な単語はありませんでした。"

    # 回数ごとに降順でグループ化
    groups = {}
    for word, count in picked:
        groups.setdefault(count, []).append(word)

    output_lines = []
    for count in sorted(groups.keys(), reverse=True):
        word_list = "　".join(groups[count])
        output_lines.append(f"{count}回　{word_list}")

    text_out = "\n".join(output_lines)
    return clamp_text(text_out)  # 念のためここも300文字上限


# =========================
# Claude に質問する関数
# =========================
def ask_claude(question: str) -> str:
    """
    Anthropic Claude に質問して回答テキストを返す
    """
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    json_data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": question}
        ],
    }

    with httpx.Client(timeout=20.0) as http_client:
        resp = http_client.post(url, headers=headers, json=json_data)
        resp.raise_for_status()
        data = resp.json()

    try:
        text = data["content"][0]["text"]
    except Exception:
        text = f"Claude のレスポンス解析でエラーが発生しました: {data}"

    return clamp_text(text)


# =========================
# /ask エンドポイント本体
# =========================
@app.post("/ask")
def ask_multi_models(body: AskRequest):
    """
    複数のモデル(OpenAI 2つ + Claude + Gemini + Mistral)に同じ質問を投げて、
    その結果＋まとめ解析（要約・矛盾点・頻出単語）を返す。
    """
    question = body.question.strip()

    # 入力文字数制限（バックエンド側のセーフティ）
    if len(question) > QUESTION_MAX_CHARS:
        msg = f"質問は{QUESTION_MAX_CHARS}文字以内にしてください（現在: {len(question)}文字）。"
        return {
            "question": question[:QUESTION_MAX_CHARS],
            "answers": {
                "error": msg
            },
        }

    # どのモデルを使うか（フロントから来なかった場合は「OpenAI+ClaudeだけON」をデフォルトに）
    selected = body.models or {
        "gpt4o": True,
        "gpt41": True,
        "claude": True,
        "gemini": False,
        "mistral": False,
    }

    genre = (body.genre or "numeric").lower()

    # まずは「各モデルの素の回答」だけを集める
    model_answers: dict[str, str] = {}

    # --- OpenAI gpt-4o-mini ---
    if selected.get("gpt4o", False):
        try:
            model_answers["gpt4o_mini"] = ask_openai("gpt-4o-mini", question)
        except Exception as e:
            model_answers["gpt4o_mini"] = clamp_text(f"OpenAI(gpt-4o-mini)でエラーが発生しました: {e}")

    # --- OpenAI gpt-4.1-mini ---
    if selected.get("gpt41", False):
        try:
            model_answers["gpt4_1_mini"] = ask_openai("gpt-4.1-mini", question)
        except Exception as e:
            model_answers["gpt4_1_mini"] = clamp_text(f"OpenAI(gpt-4.1-mini)でエラーが発生しました: {e}")

    # --- Claude ---
    if selected.get("claude", False):
        try:
            model_answers["claude_haiku"] = ask_claude(question)
        except Exception as e:
            model_answers["claude_haiku"] = clamp_text(f"Claudeでエラーが発生しました: {e}")

    # --- Gemini ---
    if selected.get("gemini", False):
        try:
            model_answers["gemini_2.5_Flash_Lite"] = ask_gemini(question)
        except Exception as e:
            model_answers["gemini_2.5_Flash_Lite"] = clamp_text(f"Geminiでエラーが発生しました: {e}")

    # --- Mistral ---
    if selected.get("mistral", False):
        try:
            model_answers["mistral_small"] = ask_mistral(question)
        except Exception as e:
            model_answers["mistral_small"] = clamp_text(f"Mistralでエラーが発生しました: {e}")

    # 1つも選択されていない／全部エラーで空になった場合の安全策
    if not model_answers:
        return {
            "question": question,
            "answers": {
                "error": "有効なAIモデルが選択されていません。",
            },
        }

    # --- 要約 & 矛盾点 & 頻出単語 ---
    need_summary = body.need_summary

    if need_summary:
        # ジャンル別の扱い
        if genre == "trivia":
            # ★ 雑学/日常会話 → OpenAI解析は呼ばずに軽量サマリだけ作る
            short_parts = []
            for name, ans in model_answers.items():
                short_parts.append(f"{name}: {ans[:60]}")
            summary = clamp_text(" / ".join(short_parts)) or "要約を簡略化しています。"

            contradictions = "この質問は雑学・日常会話ジャンルとして処理しているため、数値や厳密な事実の矛盾検出は行っていません。"
        else:
            # numeric / history / opinion → OpenAIで圧縮解析
            try:
                analysis = analyze_answers_with_openai(model_answers, genre)
                summary = clamp_text(analysis.get("summary", "要約が取得できませんでした。"))
                contradictions = clamp_text(analysis.get("contradictions", "矛盾点が取得できませんでした。"))
            except Exception as e:
                summary = clamp_text(f"要約生成中にエラーが発生しました: {e}")
                contradictions = clamp_text(f"矛盾点抽出中にエラーが発生しました: {e}")

        # 頻出単語（Pythonで解析：トークン消費はなし）
        try:
            frequent_words_text = extract_frequent_words(model_answers)
        except Exception as e:
            frequent_words_text = clamp_text(f"頻出単語抽出中にエラーが発生しました: {e}")
    else:
        # 解析OFF時 → 追加APIコールなし
        summary = ""
        contradictions = ""
        frequent_words_text = ""

    # 最後に、画面に返す answers をまとめる
    answers = dict(model_answers)  # まずは生の回答をコピー
    answers["summary"] = summary
    answers["contradictions"] = contradictions
    answers["frequent_words"] = frequent_words_text

    return {
        "question": question,
        "answers": answers,
    }


# =========================
# ルートで index.html / 利用規約 を返す
# =========================
@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.get("/kiyaku.html")
def read_terms():
    return FileResponse("kiyaku.html")
