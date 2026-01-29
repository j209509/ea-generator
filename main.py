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
    # IMPORTANT:
    # The frontend expects structured JSON so it can display EA Info separately
    # from the EA source code, and so the downloaded file contains CODE ONLY.
    return (
        "You are an expert MetaTrader EA developer.\n"
        "You MUST output ONLY valid JSON (no markdown, no code fences, no extra text).\n"
        "The JSON MUST contain exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.\n"
        "- ea_name: ASCII only (letters/digits/_/-), no spaces, 8-32 chars.\n"
        "- ea_info: Japanese, <= 400 characters. Describe the EA briefly.\n"
        "- recommended_params: Japanese. Up to 5 lines. Each line format: 'Name: Range (short note)'.\n"
        "- ea_code: EA source code ONLY (MQL4 or MQL5 depending on the user's request).\n"
        "  ASCII only inside ea_code (NO Japanese), and it must compile: includes, inputs, OnInit/OnDeinit/OnTick, helpers, etc.\n"
        "If output becomes long, shorten ea_info/recommended_params, but NEVER omit ea_code.\n"
    )


def _extract_json_obj(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from LLM output."""
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    # 1) Direct JSON
    try:
        obj = __import__("json").loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        try:
            obj = __import__("json").loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) First '{' ... last '}'
    first = t.find("{")
    last = t.rfind("}")
    if first >= 0 and last > first:
        chunk = t[first : last + 1]
        try:
            obj = __import__("json").loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _normalize_recommended_params(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        s = v.strip()
    elif isinstance(v, dict):
        # Dict -> "k: v" lines
        lines = []
        for k, val in v.items():
            vv = val if isinstance(val, str) else str(val)
            lines.append(f"{k}: {vv}".strip())
        s = "\n".join(lines).strip()
    else:
        s = str(v).strip()

    # Enforce up to 5 lines
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    lines = lines[:5]
    return "\n".join(lines)


def _normalize_ea_info(v) -> str:
    s = (v or "").strip() if isinstance(v, str) else str(v or "").strip()
    # Roughly cap to 400 chars
    return s[:400]


def _ensure_ascii_only(s: str) -> str:
    # Keep printable ASCII + newlines/tabs.
    out = []
    for ch in s:
        o = ord(ch)
        if ch in ("\n", "\r", "\t"):
            out.append(ch)
        elif 32 <= o <= 126:
            out.append(ch)
        else:
            # Drop non-ASCII characters
            pass
    return "".join(out)


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
        raw_out = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai_error: {str(e)}")

    if not raw_out:
        raise HTTPException(status_code=500, detail="openai returned empty output")

    obj = _extract_json_obj(raw_out)
    if not obj:
        # As a fallback, treat the entire output as code only.
        # (This keeps the service usable even if the model violates format.)
        ea_code_only = _ensure_ascii_only(raw_out)
        ea_name = generate_ea_name(user_prompt, ea_code_only)
        ea_info = ""
        recommended_params = ""
    else:
        ea_name = _sanitize_ea_name(str(obj.get("ea_name") or ""))
        ea_info = _normalize_ea_info(obj.get("ea_info") or "")
        recommended_params = _normalize_recommended_params(obj.get("recommended_params") or "")

        # Some models accidentally nest JSON into ea_code; if so, unwrap it.
        ea_code_val = obj.get("ea_code")
        if isinstance(ea_code_val, str) and ea_code_val.strip().startswith("{") and '"ea_code"' in ea_code_val:
            inner = _extract_json_obj(ea_code_val)
            if inner and isinstance(inner.get("ea_code"), str):
                ea_code_val = inner.get("ea_code")

        ea_code_only = _ensure_ascii_only(str(ea_code_val or "").strip())

        # If model forgot ea_name, fall back to name generator.
        if not ea_name or ea_name == "EA":
            ea_name = generate_ea_name(user_prompt, ea_code_only)

    if not ea_code_only:
        raise HTTPException(status_code=500, detail="failed to obtain ea_code")

    used = used + 1
    _usage_by_uid[uid] = used
    remaining = max(FREE_LIMIT - used, 0)

    preview = ea_code_only[:300].replace("\r\n", "\n")

    return {
        "ok": True,
        "uid": uid,
        "used": used,
        "remaining": remaining,
        "model": MODEL,
        "received_prompt_len": len(user_prompt),
        "preview": preview,
        "ea_name": ea_name,
        "ea_info": ea_info,
        "recommended_params": recommended_params,
        "ea_code": ea_code_only,
        "ts": int(time.time()),
    }
