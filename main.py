from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import re
import json
import hmac
import hashlib
from typing import Any, Optional, Tuple

import firebase_admin
from firebase_admin import credentials, auth, firestore

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

db = firestore.client()

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=openai_api_key)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2-chat-latest")

FREE_LIMIT = 3  # free users can generate up to 3 times
# Stripe webhook signing secret (Stripe Dashboard -> Webhooks -> endpoint -> Signing secret)
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()

# Firestore collection name for user states
USERS_COL = os.environ.get("USERS_COLLECTION", "users").strip() or "users"


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
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
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
            obj = json.loads(chunk)
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
        lines = []
        for k, val in v.items():
            vv = val if isinstance(val, str) else str(val)
            lines.append(f"{k}: {vv}".strip())
        s = "\n".join(lines).strip()
    else:
        s = str(v).strip()

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    lines = lines[:5]
    return "\n".join(lines)


def _normalize_ea_info(v) -> str:
    s = (v or "").strip() if isinstance(v, str) else str(v or "").strip()
    return s[:400]


def _ensure_ascii_only(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch in ("\n", "\r", "\t"):
            out.append(ch)
        elif 32 <= o <= 126:
            out.append(ch)
        else:
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


def _user_doc_ref(uid: str):
    return db.collection(USERS_COL).document(uid)


def _get_or_create_user_state(uid: str, email: str | None) -> dict:
    ref = _user_doc_ref(uid)
    snap = ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        data.setdefault("is_pro", False)
        data.setdefault("free_generate_count", 0)
        return data

    init = {
        "is_pro": False,
        "free_generate_count": 0,
        "email": email or "",
        "stripe_customer_id": "",
        "stripe_subscription_id": "",
        "updated_at": firestore.SERVER_TIMESTAMP,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    ref.set(init, merge=True)
    init["updated_at"] = int(time.time())
    init["created_at"] = int(time.time())
    return init


def _remaining_from_state(state: dict) -> int:
    if bool(state.get("is_pro")):
        return 999999
    used = int(state.get("free_generate_count") or 0)
    rem = FREE_LIMIT - used
    return rem if rem > 0 else 0


def _increment_free_count(uid: str) -> Tuple[int, int]:
    ref = _user_doc_ref(uid)

    @firestore.transactional
    def txn_op(txn: firestore.Transaction):
        snap = ref.get(transaction=txn)
        data = snap.to_dict() or {}
        is_pro = bool(data.get("is_pro"))
        used = int(data.get("free_generate_count") or 0)
        if is_pro:
            return used, 999999
        used2 = used + 1
        txn.set(
            ref,
            {"free_generate_count": used2, "updated_at": firestore.SERVER_TIMESTAMP},
            merge=True,
        )
        rem = FREE_LIMIT - used2
        rem = rem if rem > 0 else 0
        return used2, rem

    txn = db.transaction()
    return txn_op(txn)


def _stripe_parse_sig_header(sig_header: str) -> dict:
    parts: dict[str, list[str]] = {}
    for item in (sig_header or "").split(","):
        item = item.strip()
        if "=" in item:
            k, v = item.split("=", 1)
            parts.setdefault(k, [])
            parts[k].append(v)
    return parts


def _stripe_verify_signature(payload: bytes, sig_header: str, secret: str, tolerance_sec: int = 300) -> bool:
    if not secret:
        return False
    parts = _stripe_parse_sig_header(sig_header)
    t_vals = parts.get("t") or []
    v1_vals = parts.get("v1") or []
    if not t_vals or not v1_vals:
        return False
    try:
        ts = int(t_vals[0])
    except Exception:
        return False

    now = int(time.time())
    if abs(now - ts) > tolerance_sec:
        return False

    signed = f"{ts}.".encode("utf-8") + payload
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()

    for v in v1_vals:
        if hmac.compare_digest(v, expected):
            return True
    return False


def _set_user_pro_by_uid(uid: str, is_pro: bool, customer_id: str = "", subscription_id: str = "", email: str = "") -> None:
    ref = _user_doc_ref(uid)
    payload: dict[str, Any] = {
        "is_pro": bool(is_pro),
        "updated_at": firestore.SERVER_TIMESTAMP,
    }
    if customer_id:
        payload["stripe_customer_id"] = customer_id
    if subscription_id:
        payload["stripe_subscription_id"] = subscription_id
    if email:
        payload["email"] = email
    ref.set(payload, merge=True)


def _find_user_uid_by_email(email: str) -> Optional[str]:
    if not email:
        return None
    try:
        u = auth.get_user_by_email(email)
        return u.uid
    except Exception:
        return None


def _find_user_uid_by_stripe_ids(customer_id: str, subscription_id: str) -> Optional[str]:
    if subscription_id:
        q = db.collection(USERS_COL).where("stripe_subscription_id", "==", subscription_id).limit(1).stream()
        for doc in q:
            return doc.id
    if customer_id:
        q = db.collection(USERS_COL).where("stripe_customer_id", "==", customer_id).limit(1).stream()
        for doc in q:
            return doc.id
    return None


@app.get("/billing/status")
def billing_status(authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    uid = decoded.get("uid") or "unknown"
    email = decoded.get("email")
    state = _get_or_create_user_state(uid, email)
    return {
        "uid": uid,
        "email": email,
        "is_pro": bool(state.get("is_pro")),
        "free_generate_count": int(state.get("free_generate_count") or 0),
        "free_limit": FREE_LIMIT,
        "remaining": _remaining_from_state(state),
        "ts": int(time.time()),
    }


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(default="")):
    payload = await request.body()

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="stripe webhook secret not configured")

    ok = _stripe_verify_signature(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    if not ok:
        raise HTTPException(status_code=400, detail="invalid stripe signature")

    try:
        event = json.loads(payload.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    etype = event.get("type") or ""
    obj = (((event.get("data") or {}).get("object")) or {}) if isinstance(event, dict) else {}

    def _sub_is_pro(status: str) -> bool:
        s = (status or "").lower()
        return s in ("active", "trialing")

    if etype == "checkout.session.completed":
        email = ""
        cust = ""
        sub_id = ""

        if isinstance(obj, dict):
            cust = str(obj.get("customer") or "")
            sub_id = str(obj.get("subscription") or "")
            cd = obj.get("customer_details")
            if isinstance(cd, dict):
                email = str(cd.get("email") or "")
            if not email:
                email = str(obj.get("customer_email") or "")

        uid = _find_user_uid_by_email(email) if email else None
        if uid:
            _set_user_pro_by_uid(uid, True, customer_id=cust, subscription_id=sub_id, email=email)
        return {"ok": True}

    if etype in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        sub_id = ""
        cust = ""
        status = ""

        if isinstance(obj, dict):
            sub_id = str(obj.get("id") or "")
            cust = str(obj.get("customer") or "")
            status = str(obj.get("status") or "")

        uid = _find_user_uid_by_stripe_ids(cust, sub_id)
        is_pro = False if etype == "customer.subscription.deleted" else _sub_is_pro(status)

        if uid:
            _set_user_pro_by_uid(uid, is_pro, customer_id=cust, subscription_id=sub_id)
        return {"ok": True}

    return {"ok": True}


@app.post("/generate")
def generate(body: GenerateReq, authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    uid = decoded.get("uid") or "unknown"
    email = decoded.get("email")

    state = _get_or_create_user_state(uid, email)
    is_pro = bool(state.get("is_pro"))
    used_free = int(state.get("free_generate_count") or 0)
    remaining = _remaining_from_state(state)

    if (not is_pro) and remaining <= 0:
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
        ea_code_only = _ensure_ascii_only(raw_out)
        ea_name = generate_ea_name(user_prompt, ea_code_only)
        ea_info = ""
        recommended_params = ""
    else:
        ea_name = _sanitize_ea_name(str(obj.get("ea_name") or ""))
        ea_info = _normalize_ea_info(obj.get("ea_info") or "")
        recommended_params = _normalize_recommended_params(obj.get("recommended_params") or "")

        ea_code_val = obj.get("ea_code")
        if isinstance(ea_code_val, str) and ea_code_val.strip().startswith("{") and '"ea_code"' in ea_code_val:
            inner = _extract_json_obj(ea_code_val)
            if inner and isinstance(inner.get("ea_code"), str):
                ea_code_val = inner.get("ea_code")

        ea_code_only = _ensure_ascii_only(str(ea_code_val or "").strip())

        if not ea_name or ea_name == "EA":
            ea_name = generate_ea_name(user_prompt, ea_code_only)

    if not ea_code_only:
        raise HTTPException(status_code=500, detail="failed to obtain ea_code")

    if is_pro:
        used = used_free
        remaining = 999999
    else:
        used, remaining = _increment_free_count(uid)

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
