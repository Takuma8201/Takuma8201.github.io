import os
import json
import re
from collections import Counter

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import httpx
from google import genai

# =========================
# 設定
# =========================
QUESTION_MAX_CHARS = 200
ANSWER_MAX_CHARS = 300

# =========================
# .env & APIキー
# =========================
load_dotenv()

openai_api_key    = os.getenv("OPENAI_API_KEY", "").strip()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
gemini_api_key    = os.getenv("GEMINI_API_KEY", "").strip()
mistral_api_key   = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")

# =========================
# FastAPI
# =========================
app = FastAPI()

# 静的ファイル
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# クライアント
# =========================
client = OpenAI(api_key=openai_api_key)
gemini_client = genai.Client(api_key=gemini_api_key)

# =========================
# リクエストモデル
# =========================
class AskRequest(BaseModel):
    question: str
    models: dict | None = None
    need_summary: bool = True
    genre: str | None = "numeric"               # numeric | history | opinion | trivia
    contradiction_strength: str = "medium"      # low | medium | high

# =========================
# 共通
# =========================
def clamp(text: str, n: int = ANSWER_MAX_CHARS):
    if not text:
        return ""
    return text if len(text) <= n else text[:n] + "…"

# =========================
# 各AI
# =========================
def ask_openai(model: str, q: str):
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "日本語で簡潔に答えてください。"},
            {"role": "user", "content": q},
        ],
        max_tokens=300,
    )
    return clamp(r.choices[0].message.content)

def ask_claude(q: str):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": q}],
    }
    with httpx.Client(timeout=20) as c:
        r = c.post(url, headers=headers, json=body)
        r.raise_for_status()
        return clamp(r.json()["content"][0]["text"])

def ask_gemini(q: str):
    r = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=q,
    )
    return clamp(r.text)

def ask_mistral(q: str):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": q}],
    }
    with httpx.Client(timeout=20) as c:
        r = c.post(url, headers=headers, json=body)
        r.raise_for_status()
        return clamp(r.json()["choices"][0]["message"]["content"])

# =========================
# 圧縮（解析用）
# =========================
def compress_for_analysis(all_answers: dict, genre: str):
    summary_parts = []
    contradiction_parts = []

    for name, text in all_answers.items():
        summary_parts.append(f"[{name}]\n{text[:120]}")

        if genre in ("numeric", "history", "opinion"):
            nums = re.findall(r"[^。]*[0-9０-９％%][^。]*。", text)
            if nums:
                contradiction_parts.append(f"[{name}]\n" + "\n".join(nums))

    return {
        "summary_text": "\n\n".join(summary_parts),
        "contradiction_text": "\n\n".join(contradiction_parts),
    }

# =========================
# 矛盾検出ルール生成
# =========================
def build_contradiction_instruction(genre: str, strength: str) -> str:
    genre = (genre or "numeric").lower()
    strength = (strength or "medium").lower()

    strength_rule = {
        "low": "明確に結論が食い違う場合のみ矛盾として指摘する。",
        "medium": "数値・割合・確率・年号など意味が変わる差を矛盾として指摘する。",
        "high": "前提条件・適用範囲・定義・因果関係の違いも含めて比較する。",
    }.get(strength, "重要な差異を中心に比較する。")

    genre_rule = {
        "numeric": "数値・単位・割合・桁の違いを重視する。",
        "history": "年号・出来事の前後関係・時系列を重視する。",
        "opinion": "価値判断ではなく、事実や根拠の衝突のみを矛盾として扱う。",
        "trivia": "雑学として扱い、不確実な点は断定しない。",
    }.get(genre, "重要な事実関係の違いを重視する。")

    return f"【強度】{strength_rule}\n【ジャンル】{genre_rule}"

# =========================
# OpenAIで解析
# =========================
def analyze_answers_with_openai(all_answers: dict, genre: str, strength: str) -> dict:
    compressed = compress_for_analysis(all_answers, genre)
    instruction = build_contradiction_instruction(genre, strength)

    user_prompt = f"""
以下は複数AIの回答です。

【要約用】
{compressed['summary_text']}

【矛盾検出用】
{compressed['contradiction_text']}

ジャンル: {genre}
矛盾検出強度: {strength}

{instruction}

次を日本語で、必ずJSONのみで出力してください。

{{
  "summary": "要点を2〜3文で要約",
  "contradictions": "矛盾点を簡潔に。なければ『明確な矛盾は確認できません』"
}}
"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "要約と矛盾検出を行うアシスタントです。JSONのみ出力してください。"},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=350,
    )

    try:
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {
            "summary": "要約の解析に失敗しました。",
            "contradictions": "矛盾点の解析に失敗しました。",
        }

# =========================
# 頻出単語
# =========================
def frequent_words(ans: dict):
    text = " ".join(ans.values())
    words = re.findall(r"[一-龥ぁ-んァ-ンa-zA-Z0-9]+", text)
    stop = {"は","が","の","に","を","で","て","と","です","ます","ある","ない"}
    cnt = Counter(w for w in words if w not in stop and len(w) > 1)
    picked = [(w,c) for w,c in cnt.items() if c >= 3]
    if not picked:
        return "頻出単語はありませんでした。"
    out = {}
    for w,c in picked:
        out.setdefault(c, []).append(w)
    return "\n".join(f"{c}回 {' '.join(ws)}" for c,ws in sorted(out.items(), reverse=True))

# =========================
# /ask API
# =========================
@app.post("/ask")
def ask_api(body: AskRequest):
    q = body.question.strip()
    if len(q) > QUESTION_MAX_CHARS:
        return {"answers": {"error": "質問が長すぎます"}}

    sel = body.models or {
        "gpt4o": True,
        "gpt41": True,
        "claude": True,
        "gemini": False,
        "mistral": False,
    }

    ans = {}
    if sel.get("gpt4o"): ans["gpt4o_mini"] = ask_openai("gpt-4o-mini", q)
    if sel.get("gpt41"): ans["gpt4_1_mini"] = ask_openai("gpt-4.1-mini", q)
    if sel.get("claude"): ans["claude_haiku"] = ask_claude(q)
    if sel.get("gemini"): ans["gemini_flash"] = ask_gemini(q)
    if sel.get("mistral"): ans["mistral_small"] = ask_mistral(q)

    if body.need_summary:
        analysis = analyze_answers_with_openai(
            ans,
            body.genre,
            body.contradiction_strength
        )
        ans["summary"] = clamp(analysis["summary"])
        ans["contradictions"] = clamp(analysis["contradictions"])
        ans["frequent_words"] = frequent_words(ans)

    return {"answers": ans}

# =========================
# HTML
# =========================
@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/kiyaku.html")
def kiyaku():
    return FileResponse("static/kiyaku.html")

@app.get("/privacy.html")
def privacy():
    return FileResponse("static/privacy.html")
