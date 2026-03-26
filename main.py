from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import re
import json
import hmac
import hashlib
import secrets
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
try:
    MAX_GENERATION_PASSES = max(1, int(os.environ.get("MAX_GENERATION_PASSES", "3")))
except Exception:
    MAX_GENERATION_PASSES = 3

STATIC_ISSUE_LIMIT = 12

FREE_LIMIT = 3  # free users can generate up to 3 times
# Stripe webhook signing secret (Stripe Dashboard -> Webhooks -> endpoint -> Signing secret)
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()

# Firestore collection name for user states
USERS_COL = os.environ.get("USERS_COLLECTION", "users").strip() or "users"
SHARES_COL = os.environ.get("SHARES_COLLECTION", "shared_eas").strip() or "shared_eas"

KNOWN_SYMBOL_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "TRY",
    "ZAR",
    "SEK",
    "NOK",
    "DKK",
    "SGD",
    "HKD",
    "MXN",
    "PLN",
    "HUF",
    "CZK",
    "RUB",
    "CNH",
    "CNY",
    "INR",
    "XAU",
    "XAG",
    "BTC",
    "ETH",
    "LTC",
    "BCH",
    "SOL",
}

KNOWN_NONFX_SYMBOLS = {
    "NAS100",
    "US100",
    "US30",
    "GER40",
    "DE40",
    "JP225",
    "UK100",
    "SPX500",
    "BTCUSD",
    "ETHUSD",
    "XAUUSD",
    "XAGUSD",
}


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


class ImproveReq(BaseModel):
    instruction: str = ""
    existing_code: str = ""
    compiler_errors: str = ""
    platform: str = ""


class ShareCreateReq(BaseModel):
    ea_name: str = ""
    ea_info: str = ""
    recommended_params: Any = ""
    ea_code: str = ""
    filename: str = ""
    platform: str = ""
    source_prompt: str = ""


@app.get("/me")
def me(authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    return {"uid": decoded.get("uid"), "email": decoded.get("email")}


def _detect_platform(user_prompt: str) -> str:
    t = (user_prompt or "").upper()
    m = re.search(r"\bMT\s*:\s*(MT4|MT5)\b", t)
    if m:
        return m.group(1)

    idx4 = t.find("MT4")
    idx5 = t.find("MT5")
    if idx4 >= 0 and idx5 < 0:
        return "MT4"
    if idx5 >= 0 and idx4 < 0:
        return "MT5"
    if idx4 >= 0 and idx5 >= 0:
        return "MT4" if idx4 < idx5 else "MT5"
    return "MT5"


def _platform_prompt_rules(platform: str) -> str:
    if platform == "MT4":
        return (
            "Target platform is MT4 / MQL4 only.\n"
            "Use MT4-safe APIs only.\n"
            "Allowed trading style: OrderSend / OrderClose / OrderSelect / OrdersTotal / MarketInfo / Bid / Ask.\n"
            "Do NOT use MT5-only APIs such as MqlTradeRequest, MqlTradeResult, CTrade, PositionSelect, PositionGet*, trade.Buy, trade.Sell.\n"
            "Use #property strict.\n"
            "Implement OnInit, OnDeinit, and OnTick in one file.\n"
            "Do not leave unused variables, dead code, placeholders, or pseudo code.\n"
        )

    return (
        "Target platform is MT5 / MQL5 only.\n"
        "Use MT5-safe APIs only.\n"
        "If the EA places trades, prefer #include <Trade/Trade.mqh> and a CTrade instance.\n"
        "If indicators are used, create handles in OnInit, release them in OnDeinit, and read values with CopyBuffer.\n"
        "Do NOT use MT4-only APIs such as OP_BUY/OP_SELL, MarketInfo, RefreshRates, OrderSelect by position/mode, OrdersTotal, OrderType, Bid, Ask.\n"
        "Use _Symbol, _Point, _Digits, SymbolInfoDouble, PositionSelect as needed.\n"
        "Use #property strict.\n"
        "Implement OnInit, OnDeinit, and OnTick in one file.\n"
        "Do not leave unused variables, dead code, placeholders, or pseudo code.\n"
    )


def build_system_prompt(platform: str) -> str:
    # IMPORTANT:
    # The frontend expects structured JSON so it can display EA Info separately
    # from the EA source code, and so the downloaded file contains CODE ONLY.
    return (
        "You are an expert MetaTrader EA developer focused on compile-safe output.\n"
        "You MUST output ONLY valid JSON (no markdown, no code fences, no extra text).\n"
        "The JSON MUST contain exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.\n"
        "- ea_name: ASCII only (letters/digits/_/-), no spaces, 8-32 chars.\n"
        "- ea_info: Japanese, <= 400 characters. Describe the EA briefly.\n"
        "- recommended_params: Japanese. Up to 5 lines. Each line format: 'Name: Range (short note)'.\n"
        "- ea_code: EA source code ONLY (MQL4 or MQL5 depending on the user's request).\n"
        "  ASCII only inside ea_code (NO Japanese).\n"
        "Before finalizing, internally perform a strict compile-minded review:\n"
        "- no mixed MT4/MT5 APIs\n"
        "- no undeclared identifiers\n"
        "- no missing includes for used classes\n"
        "- no placeholder comments or pseudo code\n"
        "- no unused variables if they would create warnings\n"
        "- balanced braces, parentheses, and brackets\n"
        "- complete event handlers and helper functions\n"
        "- the EA must still be capable of placing trades under plausible market conditions\n"
        "- prefer one clear, reachable entry setup over too many stacked filters\n"
        "- use consistent bar indexing for signals; closed-bar logic should use the same index for price and indicators\n"
        "- if values are described in pips, implement explicit pip-to-price / pip-to-point conversion for 3/5-digit symbols\n"
        "- every CopyBuffer call must validate its return value before the buffer values are used\n"
        "- if the EA places trades, declare and use a MagicNumber and filter positions/history by magic where needed\n"
        "- if entry is blocked by existing positions, only block positions owned by this EA's MagicNumber, not every position on the symbol\n"
        "- if trailing stop, breakeven, or intrabar position management exists, manage open positions before any new-bar gate\n"
        "- if the strategy mentions swap/carry, include explicit swap-related logic instead of only naming the EA after swap\n"
        "- avoid brittle hard-coded symbol guards that can fail OnInit because of broker suffixes/prefixes or a mistyped pair string\n"
        "- if symbol restriction is needed, prefer the chart symbol or a permissive match based on the requested pair; never invent invalid 6-letter pair codes\n"
        "- loss cooldown logic must start only when a new losing-streak threshold is reached; it must not reset every bar from the same historical losses\n"
        "- do not declare dormant inputs or safety features unless they are actually implemented\n"
        "Prefer a simpler but cleaner EA over a complex EA with compile risk.\n"
        f"{_platform_prompt_rules(platform)}"
        "If output becomes long, shorten ea_info/recommended_params, but NEVER omit ea_code.\n"
    )


def build_repair_system_prompt(platform: str) -> str:
    return (
        "You repair MetaTrader EA source code so it becomes structurally compile-safe.\n"
        "It must also remain capable of opening trades under plausible market conditions.\n"
        "Return ONLY valid JSON with exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.\n"
        "- ea_name: ASCII only (letters/digits/_/-), no spaces, 8-32 chars.\n"
        "- ea_info: Japanese, <= 400 characters.\n"
        "- recommended_params: Japanese. Up to 5 lines. Each line format: 'Name: Range (short note)'.\n"
        "- ea_code: FULL EA source code ONLY, ASCII only inside ea_code.\n"
        "Preserve the original trading intent, but simplify aggressively if needed to remove compile risk.\n"
        "If the logic looks overfiltered or internally inconsistent, simplify it so a realistic signal can actually trigger.\n"
        "Use explicit pip conversion helpers when the code exposes pip-based inputs.\n"
        "Use consistent bar indexing for price and indicator signals.\n"
        "Validate every CopyBuffer call before using indicator values.\n"
        "If the EA trades, declare and use a MagicNumber and filter positions/history by magic.\n"
        "If entry is blocked by existing positions, block only positions owned by this EA's MagicNumber.\n"
        "If trailing stop, breakeven, or intrabar exits exist, manage them before any new-bar gate.\n"
        "If the request mentions swap/carry, add explicit swap-related logic.\n"
        "Avoid brittle symbol checks that hard-fail OnInit because of suffixes, prefixes, or mistyped pair strings.\n"
        "Never invent invalid 6-letter pair codes when restricting the symbol.\n"
        "Loss cooldown logic must not be reset every bar from the same historical loss streak.\n"
        "Remove or implement unused inputs such as time-exit or cooldown settings.\n"
        "If ea_info or recommended_params are in English, rewrite them into natural Japanese before returning.\n"
        "Never output markdown, explanations, or code fences.\n"
        f"{_platform_prompt_rules(platform)}"
    )


def build_improve_system_prompt(platform: str) -> str:
    return (
        "You improve an existing MetaTrader EA source code file.\n"
        "You will receive the current EA code, the user's requested changes, and optionally compiler errors.\n"
        "The improved EA must not only compile; it must also remain realistically tradable.\n"
        "Return ONLY valid JSON with exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.\n"
        "- ea_name: ASCII only (letters/digits/_/-), no spaces, 8-32 chars.\n"
        "- ea_info: Japanese, <= 400 characters.\n"
        "- recommended_params: Japanese. Up to 5 lines. Each line format: 'Name: Range (short note)'.\n"
        "- Preserve the original strategy intent unless the user explicitly asks to change it.\n"
        "- Fix compiler problems first when error logs are provided.\n"
        "- Modify only what is necessary; keep working parts stable when reasonable.\n"
        "- If the current entry logic looks too restrictive to trigger, simplify it instead of preserving broken complexity.\n"
        "- Keep bar references consistent; closed-bar signals should use the same bar index for price and indicators.\n"
        "- Use explicit pip conversion helpers when pip-based inputs are present.\n"
        "- Validate every CopyBuffer call before using indicator values.\n"
        "- If the EA trades, declare and use a MagicNumber and filter positions/history by magic.\n"
        "- If entry is blocked by existing positions, block only positions owned by this EA's MagicNumber.\n"
        "- If trailing stop, breakeven, or intrabar exits exist, manage them before any new-bar gate.\n"
        "- If the request mentions swap/carry, add explicit swap-related logic.\n"
        "- Avoid brittle symbol checks that hard-fail OnInit because of suffixes, prefixes, or mistyped pair strings.\n"
        "- Never invent invalid 6-letter pair codes when restricting the symbol.\n"
        "- Loss cooldown logic must not be reset every bar from the same historical loss streak.\n"
        "- Remove or implement unused inputs and dead safety logic.\n"
        "- ea_code must be the FULL source code for one file.\n"
        "- ea_code must stay ASCII only.\n"
        "- If ea_info or recommended_params are in English, rewrite them into natural Japanese.\n"
        "Never output markdown, explanations, or code fences.\n"
        f"{_platform_prompt_rules(platform)}"
    )


def _call_model(system_prompt: str, user_message: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


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
        "You generate short filenames for MetaTrader EA source.\n"
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
        raw = _call_model(sys_prompt, user_msg)
    except Exception:
        raw = ""

    return _sanitize_ea_name(raw)


def _share_doc_ref(share_id: str):
    return db.collection(SHARES_COL).document(share_id)


def _new_share_id() -> str:
    for _ in range(8):
        share_id = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:10]
        if share_id and not _share_doc_ref(share_id).get().exists:
            return share_id
    raise HTTPException(status_code=500, detail="failed to allocate share id")


def _serialize_value(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return v


def _normalize_platform_value(platform: str, filename: str, code: str, source_prompt: str = "") -> str:
    p = (platform or "").strip().upper()
    if p in ("MT4", "MT5"):
        return p

    name = (filename or "").lower().strip()
    if name.endswith(".mq4"):
        return "MT4"
    if name.endswith(".mq5"):
        return "MT5"

    combined = f"{source_prompt}\n{code}"
    detected = _detect_platform(combined)
    return detected if detected in ("MT4", "MT5") else "MT5"


def _fallback_public_summary(ea_name: str, platform: str, ea_info: str, recommended_params: str) -> str:
    base = (ea_info or "").strip()
    if not base:
        base = f"{ea_name} is a shared {platform} EA idea."
    base = re.sub(r"\s+", " ", base).strip()
    if len(base) > 110:
        base = base[:107].rstrip() + "..."

    if recommended_params:
        first_line = recommended_params.splitlines()[0].strip()
        if first_line:
            base = f"{base} Parameters: {first_line}"
            if len(base) > 140:
                base = base[:137].rstrip() + "..."

    return base


def _generate_public_summary(ea_name: str, platform: str, ea_info: str, recommended_params: str) -> str:
    fallback = _fallback_public_summary(ea_name, platform, ea_info, recommended_params)
    user_msg = (
        f"EA name: {ea_name}\n"
        f"Platform: {platform}\n"
        f"EA info: {ea_info}\n"
        f"Recommended params:\n{recommended_params}\n\n"
        "Write a simple Japanese teaser for an EA share page.\n"
        "Requirements:\n"
        "- 80 to 140 Japanese characters\n"
        "- Explain the trading logic at a high level in plain language\n"
        "- No hype, no guarantees, no markdown\n"
        "- Mention MT4 or MT5 only if helpful\n"
        "- Output only the teaser text\n"
    )

    try:
        raw = _call_model(
            "You write short, neutral Japanese teaser copy for trading EA share pages. Output plain text only.",
            user_msg,
        )
        text = re.sub(r"\s+", " ", (raw or "").strip())
        if not text:
            return fallback
        if len(text) > 150:
            text = text[:147].rstrip() + "..."
        return text
    except Exception:
        return fallback


def _share_public_payload(data: dict, share_id: str) -> dict[str, Any]:
    return {
        "share_id": share_id,
        "ea_name": str(data.get("ea_name") or ""),
        "platform": str(data.get("platform") or ""),
        "public_summary": str(data.get("public_summary") or ""),
        "created_at": _serialize_value(data.get("created_at")),
        "updated_at": _serialize_value(data.get("updated_at")),
    }


def _share_detail_payload(data: dict, share_id: str) -> dict[str, Any]:
    payload = _share_public_payload(data, share_id)
    payload.update(
        {
            "ea_info": str(data.get("ea_info") or ""),
            "recommended_params": str(data.get("recommended_params") or ""),
            "ea_code": str(data.get("ea_code") or ""),
            "filename": str(data.get("filename") or ""),
            "owner_uid": str(data.get("owner_uid") or ""),
        }
    )
    return payload


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


def _has_required_keys(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    needed = {"ea_name", "ea_info", "recommended_params", "ea_code"}
    return needed.issubset(set(d.keys()))


def _unwrap_inner_json_if_needed(obj: dict) -> dict:
    """
    症状:
      outer JSON は取れているが、outer["ea_code"] が JSON文字列になっているケース。
    対応:
      inner JSON を parse し、4キー揃っていれば inner を正として採用する。
    """
    try:
        v = obj.get("ea_code")
    except Exception:
        return obj

    if not isinstance(v, str):
        return obj

    s = v.strip()
    if not s.startswith("{"):
        return obj

    # 速度と誤爆抑制のため、キーが見えるときだけ試す
    if '"ea_code"' not in s or '"recommended_params"' not in s:
        return obj

    inner = _extract_json_obj(s)
    if inner and _has_required_keys(inner):
        return inner

    return obj


def _looks_like_mql_code(text: str) -> bool:
    """
    JSONパースに失敗したときに、生出力をEAコード扱いして良いかどうかの判定。
    ここが甘いと「JSON全文がコード表示」になるので、意図的に厳しめにしています。
    """
    if not text:
        return False
    t = text.strip()
    if not t:
        return False

    # JSONっぽいものは除外
    if t.startswith("{") and ('"ea_code"' in t or '"ea_name"' in t):
        return False

    # 主要なMQLっぽさ
    needles = [
        "#include",
        "OnInit",
        "OnTick",
        "OnDeinit",
        "input ",
        "MqlTradeRequest",
        "CTrade",
        "OrderSend",
        "PositionSelect",
        "SymbolInfoDouble",
    ]
    hit = 0
    for n in needles:
        if n in t:
            hit += 1

    # 2つ以上ヒットで採用
    return hit >= 2


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = (item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_symbol_token(token: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (token or "").upper().strip())


def _looks_like_standard_symbol(token: str) -> bool:
    t = _normalize_symbol_token(token)
    if not t:
        return False
    if t in KNOWN_NONFX_SYMBOLS:
        return True
    if len(t) == 6 and t[:3] in KNOWN_SYMBOL_CODES and t[3:] in KNOWN_SYMBOL_CODES:
        return True
    return False


def _extract_requested_symbols(context_text: str) -> list[str]:
    ctx = str(context_text or "")
    matches = re.findall(r"(?:PAIR|SYMBOL|通貨ペア)\s*[:：]\s*([A-Za-z0-9._+\-]+)", ctx, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b[A-Z][A-Z0-9._+\-]{4,19}\b", ctx.upper())

    out: list[str] = []
    for token in matches:
        norm = _normalize_symbol_token(token)
        if _looks_like_standard_symbol(norm):
            out.append(norm)
    return _dedupe_keep_order(out)


def _extract_symbol_guard_literals(src: str) -> list[str]:
    matches: list[str] = []
    patterns = [
        r"(?:_Symbol|Symbol\s*\(\s*\))[^;\n]{0,120}\"([A-Za-z0-9._+\-]{3,20})\"",
        r"\"([A-Za-z0-9._+\-]{3,20})\"[^;\n]{0,120}(?:_Symbol|Symbol\s*\(\s*\))",
    ]
    for pattern in patterns:
        for token in re.findall(pattern, src):
            norm = _normalize_symbol_token(token)
            if norm:
                matches.append(norm)
    return _dedupe_keep_order(matches)


def _collect_balance_issues(code: str) -> list[str]:
    issues: list[str] = []
    stack: list[tuple[str, int]] = []
    in_line_comment = False
    in_block_comment = False
    in_string = ""
    escaped = False
    i = 0
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = {v: k for k, v in pairs.items()}

    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = ""
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch in ('"', "'"):
            in_string = ch
            i += 1
            continue

        if ch in pairs:
            stack.append((ch, i))
        elif ch in closing:
            if not stack or stack[-1][0] != closing[ch]:
                issues.append(f"Unbalanced delimiter near character {i + 1}: unexpected '{ch}'.")
                break
            stack.pop()

        i += 1

    if in_string:
        issues.append("Unterminated string literal detected.")
    if in_block_comment:
        issues.append("Unterminated block comment detected.")
    if stack:
        opener, pos = stack[-1]
        issues.append(f"Unbalanced delimiter near character {pos + 1}: missing closing pair for '{opener}'.")

    return issues


def _collect_unused_input_issues(src: str) -> list[str]:
    names: list[str] = []
    for m in re.finditer(
        r"^\s*input\s+[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;)",
        src,
        flags=re.MULTILINE,
    ):
        name = m.group(1)
        if name not in names:
            names.append(name)

    unused: list[str] = []
    for name in names:
        if len(re.findall(rf"\b{re.escape(name)}\b", src)) <= 1:
            unused.append(name)

    if not unused:
        return []

    preview = ", ".join(unused[:4])
    if len(unused) > 4:
        preview += ", ..."
    return [f"Remove unused input parameters or implement their logic: {preview}."]


def _collect_trade_viability_issues(src: str, platform: str) -> list[str]:
    issues: list[str] = []

    if platform == "MT5":
        has_trade_entry = bool(
            re.search(
                r"\b(?:trade\.(?:Buy|Sell|BuyLimit|SellLimit|BuyStop|SellStop|PositionOpen)|OrderSend)\s*\(",
                src,
            )
        )
    else:
        has_trade_entry = bool(
            re.search(r"\b(?:OrderSend|trade\.(?:Buy|Sell|PositionOpen))\s*\(", src)
        )

    if not has_trade_entry:
        issues.append(
            "No visible trade entry call was found. The EA should contain at least one reachable order-open path."
        )

    if has_trade_entry and not re.search(r"\bMagic(?:Number)?\b", src):
        issues.append(
            "Trading EA should declare and use a MagicNumber so it only manages its own positions and history."
        )

    if has_trade_entry and "PositionSelect(" in src and "POSITION_MAGIC" not in src:
        issues.append(
            "Position management selects symbol positions but does not filter by POSITION_MAGIC."
        )

    if has_trade_entry and re.search(r"\bif\s*\(\s*PositionSelect\s*\(\s*_Symbol\s*\)\s*\)\s*return\s*;", src):
        issues.append(
            "Entry gating blocks on any open position for the symbol. Only positions owned by this EA's MagicNumber should disable new entries."
        )

    if has_trade_entry and "HistoryDealGet" in src and "DEAL_MAGIC" not in src:
        issues.append(
            "History-based loss tracking uses deal history without filtering by DEAL_MAGIC."
        )

    pip_names = _dedupe_keep_order(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*Pips)\b", src))
    has_pip_helper = bool(
        re.search(r"\b(?:PipSize|PipsToPrice|PipsToPoints|PointsPerPip|PipPoint|PipValue)\b", src)
    )
    if pip_names and "_Point" in src and not has_pip_helper:
        preview = ", ".join(pip_names[:4])
        if len(pip_names) > 4:
            preview += ", ..."
        issues.append(
            f"Pip-style parameters are used without an explicit pip conversion helper for 3/5-digit symbols: {preview}."
        )

    if re.search(r"\bspread[A-Za-z0-9_]*\s*=.*?/_Point", src, flags=re.IGNORECASE | re.DOTALL) and re.search(
        r"\bspread[A-Za-z0-9_]*\b.*?(?:<=|<|>=|>)\s*[A-Za-z_][A-Za-z0-9_]*Pips\b",
        src,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        issues.append(
            "Spread logic appears to compare points against a '*Pips' threshold without pip conversion."
        )

    if re.search(r"CopyBuffer\s*\([^,\n]+,\s*0,\s*0,\s*1\s*,", src) and re.search(
        r"\bi(?:Open|Close|High|Low)\s*\([^,\n]+,\s*[^,\n]+,\s*1\s*\)",
        src,
    ):
        issues.append(
            "Signal logic mixes current-bar indicator values with previous closed-bar candle data. Use a consistent closed-bar index."
        )

    if re.search(r"(?m)^\s*CopyBuffer\s*\(", src):
        issues.append(
            "Check CopyBuffer return values before using indicator buffers; standalone CopyBuffer calls can leave stale or zero data."
        )

    if re.search(
        r"\bOnInit\s*\([^)]*\)\s*\{[\s\S]{0,800}?(?:_Symbol|Symbol\s*\(\s*\))[\s\S]{0,160}?INIT_FAILED",
        src,
        flags=re.DOTALL,
    ):
        issues.append(
            "Hard-coded symbol validation can make OnInit fail on broker suffixes/prefixes or a mistyped pair string."
        )

    new_bar_pos = src.find("if(!NewBar()) return;")
    manage_pos = src.find("ManagePosition();")
    if (
        new_bar_pos >= 0
        and manage_pos > new_bar_pos
        and ("PositionModify(" in src or "trade.PositionModify(" in src or "trade.PositionClose(" in src)
    ):
        issues.append(
            "Open-position management runs only after the new-bar gate. Trailing, breakeven, and intrabar exits should usually run on every tick."
        )

    cooldown_assigns = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[A-Za-z_][A-Za-z0-9_]*AfterLoss[A-Za-z0-9_]*\b",
        src,
    )
    for counter in _dedupe_keep_order(cooldown_assigns):
        if re.search(
            rf"\bif\s*\(\s*{re.escape(counter)}\s*>\s*0\s*\)\s*\{{[^{{}}]*{re.escape(counter)}\s*--[^{{}}]*return\s*;",
            src,
            flags=re.DOTALL,
        ) and re.search(r"\bconsecutiveLoss\b", src):
            issues.append(
                "Loss cooldown logic may reset every bar and prevent trading from ever resuming. Start the cooldown only when the threshold is newly reached."
            )
            break

    return issues


def _collect_context_alignment_issues(src: str, context_text: str) -> list[str]:
    issues: list[str] = []
    ctx = (context_text or "").upper()
    requested_symbols = _extract_requested_symbols(context_text or "")
    guard_literals = _extract_symbol_guard_literals(src)

    for literal in guard_literals:
        if len(literal) == 6 and not _looks_like_standard_symbol(literal):
            issues.append(
                f"Hard-coded symbol literal '{literal}' does not look like a standard tradable pair code."
            )

    if requested_symbols and guard_literals:
        mismatched: list[str] = []
        for literal in guard_literals:
            if not any(expected in literal or literal in expected for expected in requested_symbols):
                mismatched.append(literal)
        if mismatched:
            issues.append(
                "Hard-coded symbol checks do not match the requested pair and may stop the EA from starting or trading."
            )

    if any(token in ctx for token in ["USDTRY", "EURTRY", "GBPTRY", "TRY", "XAU", "GOLD", "XAG", "SILVER", "BTC", "ETH", "NAS100", "US30", "GER40", "JP225"]):
        m = re.search(
            r"^\s*input\s+(?:double|int)\s+MaxSpreadPips\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            src,
            flags=re.MULTILINE,
        )
        if m:
            try:
                spread_limit = float(m.group(1))
                if spread_limit <= 3.0:
                    issues.append(
                        "MaxSpreadPips is extremely tight for an exotic, metal, crypto, or index-style symbol and may block all entries."
                    )
            except Exception:
                pass

    if re.search(r"スワップ|SWAP|CARRY", context_text or "", flags=re.IGNORECASE):
        if not re.search(
            r"\b(?:POSITION_SWAP|SYMBOL_SWAP_(?:LONG|SHORT)|DEAL_SWAP|SWAP)\b",
            src,
            flags=re.IGNORECASE,
        ):
            issues.append(
                "The request mentions swap or carry, but the code contains no explicit swap-related logic."
            )

    if re.search(r"売りのみ|SELL ONLY|SHORT ONLY", context_text or "", flags=re.IGNORECASE):
        if re.search(r"\btrade\.Buy\s*\(", src):
            issues.append("The request is sell-only, but the code still contains a buy entry.")

    if re.search(r"買いのみ|BUY ONLY|LONG ONLY", context_text or "", flags=re.IGNORECASE):
        if re.search(r"\btrade\.Sell\s*\(", src):
            issues.append("The request is buy-only, but the code still contains a sell entry.")

    return issues


def _is_blocking_issue(issue: str) -> bool:
    text = (issue or "").strip()
    if not text:
        return False

    non_blocking_prefixes = [
        "Trading EA should declare and use a MagicNumber",
        "Position management selects symbol positions but does not filter by POSITION_MAGIC.",
        "Entry gating blocks on any open position for the symbol.",
        "History-based loss tracking uses deal history without filtering by DEAL_MAGIC.",
        "Pip-style parameters are used without an explicit pip conversion helper",
        "Spread logic appears to compare points against a '*Pips' threshold without pip conversion.",
        "Signal logic mixes current-bar indicator values with previous closed-bar candle data.",
        "Check CopyBuffer return values before using indicator buffers",
        "Open-position management runs only after the new-bar gate.",
        "Loss cooldown logic may reset every bar and prevent trading from ever resuming.",
        "Hard-coded symbol validation can make OnInit fail on broker suffixes/prefixes or a mistyped pair string.",
        "Hard-coded symbol literal '",
        "Hard-coded symbol checks do not match the requested pair",
        "MaxSpreadPips is extremely tight for an exotic, metal, crypto, or index-style symbol",
        "The request mentions swap or carry, but the code contains no explicit swap-related logic.",
        "The request is sell-only, but the code still contains a buy entry.",
        "The request is buy-only, but the code still contains a sell entry.",
        "Remove unused input parameters or implement their logic:",
    ]

    return not any(text.startswith(prefix) for prefix in non_blocking_prefixes)


def _partition_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    deduped = _dedupe_keep_order(issues)
    blocking = [item for item in deduped if _is_blocking_issue(item)]
    advisory = [item for item in deduped if not _is_blocking_issue(item)]
    return blocking, advisory


def _collect_static_issues(code: str, platform: str) -> list[str]:
    issues: list[str] = []
    src = (code or "").strip()
    if not src:
        return ["ea_code is empty."]

    if "```" in src:
        issues.append("ea_code still contains markdown code fences.")
    if src.startswith("{") and ('\"ea_code\"' in src or '\"ea_name\"' in src):
        issues.append("ea_code still contains JSON instead of source code.")
    if "#property strict" not in src:
        issues.append("Add '#property strict' near the top of the file.")
    if not re.search(r"\b(?:int|void)\s+OnInit\s*\(", src):
        issues.append("Implement OnInit with an explicit return type.")
    if not re.search(r"\bvoid\s+OnTick\s*\(", src):
        issues.append("Implement OnTick with an explicit return type.")
    if not re.search(r"\bvoid\s+OnDeinit\s*\(", src):
        issues.append("Implement OnDeinit with an explicit return type.")
    if re.search(r"\b(TODO|FIXME|your logic here|pseudo code|placeholder)\b", src, flags=re.IGNORECASE):
        issues.append("Remove placeholders, TODOs, and pseudo code.")

    issues.extend(_collect_balance_issues(src))

    if ("CTrade" in src or re.search(r"\btrade\.", src)) and "#include <Trade/Trade.mqh>" not in src:
        issues.append("CTrade or trade.* is used without '#include <Trade/Trade.mqh>'.")
    if re.search(r"\btrade\.", src) and not re.search(r"\bCTrade\s+\w+\s*;", src):
        issues.append("trade.* is used but no CTrade instance is declared.")

    if platform == "MT5":
        if re.search(r"\bOP_(BUY|SELL|BUYLIMIT|SELLLIMIT|BUYSTOP|SELLSTOP)\b", src):
            issues.append("MT5 code contains MT4 order constants such as OP_BUY/OP_SELL.")
        if re.search(r"\b(OrderSelect|OrdersTotal|OrderType|OrderTicket|OrderMagicNumber|OrderOpenPrice|OrderClosePrice|OrderSymbol)\b", src):
            issues.append("MT5 code uses MT4 order inspection APIs such as OrderSelect/OrdersTotal.")
        if re.search(r"\b(MarketInfo|RefreshRates)\s*\(", src):
            issues.append("MT5 code uses MT4-only helpers such as MarketInfo/RefreshRates.")
        if re.search(r"(?<![_A-Za-z0-9])(Bid|Ask)(?![_A-Za-z0-9])", src):
            issues.append("MT5 code uses global Bid/Ask instead of SymbolInfoDouble or current tick data.")
        if "OrderSend(" in src and "MqlTradeRequest" not in src and "MqlTradeResult" not in src and "CTrade" not in src:
            issues.append("MT5 code appears to use an MT4-style OrderSend call.")
        if "CopyBuffer(" in src and "INVALID_HANDLE" not in src:
            issues.append("MT5 code uses CopyBuffer but does not visibly guard indicator handles with INVALID_HANDLE checks.")
    else:
        if re.search(r"\b(MqlTradeRequest|MqlTradeResult|MqlTick|CTrade|PositionSelect|PositionGet|HistoryDealGet|HistoryOrderGet)\b", src):
            issues.append("MT4 code contains MT5-only trade APIs such as MqlTradeRequest/CTrade/PositionSelect.")
        if "#include <Trade/Trade.mqh>" in src:
            issues.append("MT4 code includes the MT5 Trade.mqh header.")
        if re.search(r"\btrade\.(Buy|Sell|PositionOpen|PositionClose)\b", src):
            issues.append("MT4 code uses MT5 CTrade helper methods.")
        if "CopyBuffer(" in src:
            issues.append("MT4 code uses CopyBuffer, which is a common source of MT5-style mixed code.")

    issues.extend(_collect_trade_viability_issues(src, platform))
    issues.extend(_collect_unused_input_issues(src))

    return _dedupe_keep_order(issues)[:STATIC_ISSUE_LIMIT]


def _normalize_model_output(raw_out: str, user_prompt: str) -> dict[str, str]:
    obj = _extract_json_obj(raw_out)

    if obj and isinstance(obj, dict):
        obj2 = _unwrap_inner_json_if_needed(obj)

        if not _has_required_keys(obj2):
            raise ValueError("invalid model json shape")

        ea_name = _sanitize_ea_name(str(obj2.get("ea_name") or ""))
        ea_info = _normalize_ea_info(obj2.get("ea_info") or "")
        recommended_params = _normalize_recommended_params(obj2.get("recommended_params") or "")

        ea_code_val = obj2.get("ea_code")
        ea_code_only = _ensure_ascii_only(str(ea_code_val or "").strip())
        s = ea_code_only.lstrip()
        if s.startswith("{") and ('\"ea_code\"' in s or '\"ea_name\"' in s):
            raise ValueError("ea_code contains json (refused)")

        if not ea_name or ea_name == "EA":
            ea_name = generate_ea_name(user_prompt, ea_code_only)

        return {
            "ea_name": ea_name,
            "ea_info": ea_info,
            "recommended_params": recommended_params,
            "ea_code": ea_code_only,
        }

    if _looks_like_mql_code(raw_out):
        ea_code_only = _ensure_ascii_only(raw_out)
        return {
            "ea_name": generate_ea_name(user_prompt, ea_code_only),
            "ea_info": "",
            "recommended_params": "",
            "ea_code": ea_code_only,
        }

    raise ValueError("failed to parse model output as json")


def _build_repair_user_prompt(
    user_prompt: str,
    platform: str,
    raw_out: str,
    issues: list[str],
    parsed: Optional[dict[str, str]] = None,
) -> str:
    current = json.dumps(parsed, ensure_ascii=True) if parsed else raw_out
    bullet_issues = "\n".join(f"- {item}" for item in issues) if issues else "- Output is not valid yet."
    return (
        f"Original user requirement:\n{user_prompt}\n\n"
        f"Target platform: {platform}\n\n"
        "Current candidate output:\n"
        f"{current}\n\n"
        "Fix all of these problems:\n"
        f"{bullet_issues}\n\n"
        "Return a FULL corrected JSON object with exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.\n"
        "The result must be safer to compile than the current candidate.\n"
        "Preserve the strategy intent, but delete or simplify risky code if needed.\n"
        "ea_info and recommended_params must be natural Japanese.\n"
        "No markdown. No explanation.\n"
    )


def _build_improve_user_prompt(
    instruction: str,
    platform: str,
    existing_code: str,
    compiler_errors: str,
) -> str:
    request_text = (instruction or "").strip() or "Fix compile issues and improve the existing EA while preserving its behavior."
    out = [
        f"Target platform: {platform}",
        "",
        "Improvement request:",
        request_text,
        "",
        "Existing EA source:",
        existing_code,
    ]
    if (compiler_errors or "").strip():
        out.extend(
            [
                "",
                "Compiler error log:",
                compiler_errors.strip(),
            ]
        )
    out.extend(
        [
            "",
            "Return a FULL corrected JSON object with exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.",
            "Keep the behavior close to the original EA unless the request says otherwise.",
            "ea_info and recommended_params must be natural Japanese.",
            "No markdown. No explanation.",
        ]
    )
    return "\n".join(out)


def _build_improve_repair_user_prompt(
    instruction: str,
    platform: str,
    existing_code: str,
    compiler_errors: str,
    raw_out: str,
    issues: list[str],
    parsed: Optional[dict[str, str]] = None,
) -> str:
    current = json.dumps(parsed, ensure_ascii=True) if parsed else raw_out
    bullet_issues = "\n".join(f"- {item}" for item in issues) if issues else "- Output is not valid yet."
    request_text = (instruction or "").strip() or "Fix compile issues and improve the existing EA while preserving its behavior."
    parts = [
        f"Improvement request:\n{request_text}",
        "",
        f"Target platform: {platform}",
        "",
        "Original EA source:",
        existing_code,
    ]
    if (compiler_errors or "").strip():
        parts.extend(["", "Compiler error log:", compiler_errors.strip()])
    parts.extend(
        [
            "",
            "Current candidate output:",
            current,
            "",
            "Fix all of these problems:",
            bullet_issues,
            "",
            "Return a FULL corrected JSON object with exactly these 4 keys: ea_name, ea_info, recommended_params, ea_code.",
            "Keep the behavior close to the original EA unless the request says otherwise.",
            "No markdown. No explanation.",
        ]
    )
    return "\n".join(parts)


def _generate_validated_ea(user_prompt: str) -> dict[str, str]:
    platform = _detect_platform(user_prompt)
    raw_out = ""
    parsed: Optional[dict[str, str]] = None
    issues: list[str] = []
    blocking_issues: list[str] = []

    for attempt in range(MAX_GENERATION_PASSES):
        if attempt == 0:
            raw_out = _call_model(build_system_prompt(platform), user_prompt)
        else:
            raw_out = _call_model(
                build_repair_system_prompt(platform),
                _build_repair_user_prompt(user_prompt, platform, raw_out, issues, parsed),
            )

        if not raw_out:
            parsed = None
            issues = ["openai returned empty output"]
            continue

        try:
            parsed = _normalize_model_output(raw_out, user_prompt)
        except ValueError as e:
            parsed = None
            issues = [str(e)]
            blocking_issues = list(issues)
            continue

        issues = _collect_static_issues(parsed.get("ea_code") or "", platform)
        issues.extend(_collect_context_alignment_issues(parsed.get("ea_code") or "", user_prompt))
        blocking_issues, advisory_issues = _partition_issues(issues)
        issues = (blocking_issues + advisory_issues)[:STATIC_ISSUE_LIMIT]
        if not blocking_issues:
            return parsed

    detail = "; ".join(blocking_issues or issues) if (blocking_issues or issues) else "validation failed"
    raise HTTPException(status_code=500, detail=f"generation_validation_failed: {detail}")


def _generate_validated_improved_ea(
    instruction: str,
    existing_code: str,
    compiler_errors: str,
    platform_hint: str,
) -> dict[str, str]:
    source = str(existing_code or "").strip()
    if not source:
        raise HTTPException(status_code=422, detail="existing_code is required")

    platform = _normalize_platform_value(str(platform_hint or ""), "", source, str(instruction or ""))
    raw_out = ""
    parsed: Optional[dict[str, str]] = None
    issues: list[str] = []
    blocking_issues: list[str] = []

    for attempt in range(MAX_GENERATION_PASSES):
        if attempt == 0:
            raw_out = _call_model(
                build_improve_system_prompt(platform),
                _build_improve_user_prompt(instruction, platform, source, compiler_errors),
            )
        else:
            raw_out = _call_model(
                build_repair_system_prompt(platform),
                _build_improve_repair_user_prompt(instruction, platform, source, compiler_errors, raw_out, issues, parsed),
            )

        if not raw_out:
            parsed = None
            issues = ["openai returned empty output"]
            continue

        try:
            parsed = _normalize_model_output(raw_out, instruction or source[:500])
        except ValueError as e:
            parsed = None
            issues = [str(e)]
            blocking_issues = list(issues)
            continue

        issues = _collect_static_issues(parsed.get("ea_code") or "", platform)
        issues.extend(_collect_context_alignment_issues(parsed.get("ea_code") or "", instruction))
        blocking_issues, advisory_issues = _partition_issues(issues)
        issues = (blocking_issues + advisory_issues)[:STATIC_ISSUE_LIMIT]
        if not blocking_issues:
            return parsed

    detail = "; ".join(blocking_issues or issues) if (blocking_issues or issues) else "validation failed"
    raise HTTPException(status_code=500, detail=f"improve_validation_failed: {detail}")


def _build_generation_success_response(candidate: dict[str, str], user_prompt: str, uid: str, is_pro: bool, used_free: int) -> dict:
    ea_name = _sanitize_ea_name(str(candidate.get("ea_name") or ""))
    ea_info = _normalize_ea_info(candidate.get("ea_info") or "")
    recommended_params = _normalize_recommended_params(candidate.get("recommended_params") or "")
    ea_code_only = _ensure_ascii_only(str(candidate.get("ea_code") or "").strip())

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


@app.post("/shares")
def create_share(body: ShareCreateReq, authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    uid = decoded.get("uid") or "unknown"
    email = decoded.get("email") or ""

    ea_code = _ensure_ascii_only(str(body.ea_code or "").strip())
    if not ea_code:
        raise HTTPException(status_code=422, detail="ea_code is required")

    ea_name = _sanitize_ea_name(str(body.ea_name or ""))
    if not ea_name or ea_name == "EA":
        ea_name = generate_ea_name(str(body.source_prompt or ""), ea_code)

    ea_info = _normalize_ea_info(body.ea_info or "")
    recommended_params = _normalize_recommended_params(body.recommended_params or "")
    filename = str(body.filename or "").strip()
    platform = _normalize_platform_value(str(body.platform or ""), filename, ea_code, str(body.source_prompt or ""))
    public_summary = _generate_public_summary(ea_name, platform, ea_info, recommended_params)

    share_id = _new_share_id()
    now = firestore.SERVER_TIMESTAMP
    payload = {
        "owner_uid": uid,
        "owner_email": email,
        "ea_name": ea_name,
        "ea_info": ea_info,
        "recommended_params": recommended_params,
        "ea_code": ea_code,
        "filename": filename,
        "platform": platform,
        "public_summary": public_summary,
        "created_at": now,
        "updated_at": now,
    }
    _share_doc_ref(share_id).set(payload, merge=True)

    return {
        "ok": True,
        "share_id": share_id,
        "public": _share_public_payload({**payload, "created_at": int(time.time()), "updated_at": int(time.time())}, share_id),
    }


@app.get("/shares/{share_id}/public")
def get_share_public(share_id: str):
    sid = (share_id or "").strip()
    if not sid:
        raise HTTPException(status_code=404, detail="share not found")

    snap = _share_doc_ref(sid).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="share not found")

    data = snap.to_dict() or {}
    return {"ok": True, "share": _share_public_payload(data, sid)}


@app.get("/shares/{share_id}")
def get_share_detail(share_id: str, authorization: str = Header(default="")):
    verify_user(authorization)
    sid = (share_id or "").strip()
    if not sid:
        raise HTTPException(status_code=404, detail="share not found")

    snap = _share_doc_ref(sid).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="share not found")

    data = snap.to_dict() or {}
    return {"ok": True, "share": _share_detail_payload(data, sid)}


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

    try:
        candidate = _generate_validated_ea(user_prompt)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"openai_error: {str(e)}")

    return _build_generation_success_response(candidate, user_prompt, uid, is_pro, used_free)


@app.post("/improve")
def improve(body: ImproveReq, authorization: str = Header(default="")):
    decoded = verify_user(authorization)
    uid = decoded.get("uid") or "unknown"
    email = decoded.get("email")

    state = _get_or_create_user_state(uid, email)
    is_pro = bool(state.get("is_pro"))
    used_free = int(state.get("free_generate_count") or 0)
    remaining = _remaining_from_state(state)

    if (not is_pro) and remaining <= 0:
        raise HTTPException(status_code=403, detail="free limit reached")

    instruction = (body.instruction or "").strip()
    existing_code = str(body.existing_code or "").strip()
    compiler_errors = str(body.compiler_errors or "").strip()
    platform = str(body.platform or "").strip()

    if not existing_code:
        raise HTTPException(status_code=422, detail="existing_code is required")

    try:
        candidate = _generate_validated_improved_ea(instruction, existing_code, compiler_errors, platform)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"openai_error: {str(e)}")

    success_prompt = instruction or compiler_errors or existing_code[:500]
    return _build_generation_success_response(candidate, success_prompt, uid, is_pro, used_free)
