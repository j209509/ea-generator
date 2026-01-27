from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import re

import firebase_admin
from firebase_admin import credentials, auth

from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://app.ea-generator.com",
        "https://ea-generator.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Admin init:
# - local: GOOGLE_APPLICATION_CREDENTIALS があればサービスアカウントJSONを使う
# - Cloud Run: 無ければ実行中のサービスアカウント(ADC)で初期化する
if not firebase_admin._apps:
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=openai_api_key)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2-chat-latest")

FREE_LIMIT = int(os.environ.get("FREE_LIMIT", "3"))
_usage_by_uid: dict[str, int] = {}


def verify_user(authorization: str) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.replace("Bearer ", "", 1).strip()
    try:
        return auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")


class GenerateReq(BaseModel):
    prompt: str


@app.get("/me")
def me(authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    return {"uid": decoded.get("uid"), "email": decoded.get("email")}


def build_system_prompt() -> str:
    return (
        "You are an expert MetaTrader EA developer.\n"
        "Return ONLY the full EA source code as plain text.\n"
        "No markdown. No explanations. No code fences.\n"
        "Target: MetaTrader 5 (MQL5) unless user explicitly requests MT4.\n"
        "The output must compile (include required includes, inputs, OnInit/OnDeinit/OnTick, etc.).\n"
        "If requirements are ambiguous, make reasonable defaults and keep it simple.\n"
    )


def _sanitize_ea_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "EA"
    return s[:32]


def generate_ea_name(user_prompt: str, ea_code: str) -> str:
    sys_prompt = (
        "You generate short filenames for MetaTrader 5 EA source.\n"
        "Output ONLY ONE LINE: an ASCII name using letters, digits, underscore.\n"
        "No spaces. No extension. Length 8-32.\n"
        "Do not include words like draft, version, test.\n"
    )

    up = (user_prompt or "").strip()
    code_head = (ea_code or "").strip().splitlines()[:60]
    code_head_text = "\n".join(code_head)

    user_msg = (
        "Create a concise EA name.\n"
        "User requirement:\n"
        f"{up}\n\n"
        "EA code header:\n"
        f"{code_head_text}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        raw = ""

    return _sanitize_ea_name(raw)


@app.post("/generate")
def generate(body: GenerateReq, authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    uid = decoded.get("uid") or "unknown"

    used = _usage_by_uid.get(uid, 0)
    remaining = max(FREE_LIMIT - used, 0)
    if remaining <= 0:
        raise HTTPException(status_code=403, detail="free limit reached")

    user_prompt = (body.prompt or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=422, detail="prompt is required")

    sys_prompt = build_system_prompt()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        ea_code = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai_error: {str(e)}")

    if not ea_code:
        raise HTTPException(status_code=500, detail="openai returned empty output")

    ea_name = generate_ea_name(user_prompt, ea_code)

    used = used + 1
    _usage_by_uid[uid] = used
    remaining = max(FREE_LIMIT - used, 0)

    preview = ea_code[:300].replace("\r\n", "\n")

    return {
        "ok": True,
        "uid": uid,
        "used": used,
        "remaining": remaining,
        "model": MODEL,
        "received_prompt_len": len(user_prompt),
        "preview": preview,
        "ea_name": ea_name,
        "ea_code": ea_code,
        "ts": int(time.time()),
    }
