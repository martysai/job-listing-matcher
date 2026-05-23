import base64
import hashlib
import hmac
import os
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter()

_USERNAME = os.getenv("AUTH_USERNAME", "saradev")
_PASSWORD = os.getenv("AUTH_PASSWORD", "vibe-code")
COOKIE_NAME = "session"
# HMAC signing key for the session cookie. Pulled from AUTH_SECRET in any
# real deployment; if unset we fall back to a fresh per-process random secret
# so a leaked literal can never sign a valid cookie. Sessions are then
# invalidated on every process restart, which is the safe failure mode.
_SECRET = os.getenv("AUTH_SECRET") or secrets.token_urlsafe(64)
_MAX_AGE = 86400  # 24 hours
_SECURE_COOKIE = os.getenv("HTTPS", "true").lower() != "false"


def _sign(payload: str) -> str:
    return hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_cookie(username: str) -> str:
    ts = int(time.time())
    payload = f"{username}.{ts}"
    return f"{payload}.{_sign(payload)}"


def verify_cookie(value: str) -> bool:
    try:
        *parts, sig = value.split(".")
        payload = ".".join(parts)
        return hmac.compare_digest(_sign(payload), sig)
    except Exception:
        return False


@router.post("/auth/login")
async def login(request: Request, response: Response):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        raise HTTPException(status_code=401, detail="Missing credentials")
    try:
        user, pwd = base64.b64decode(auth[6:]).decode().split(":", 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid header")
    if user != _USERNAME or pwd != _PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    response.set_cookie(
        key=COOKIE_NAME,
        value=make_cookie(user),
        httponly=True,
        samesite="lax",
        max_age=_MAX_AGE,
        secure=_SECURE_COOKIE,
    )
    return {"ok": True}


@router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.get("/auth/me")
async def me(request: Request):
    if not verify_cookie(request.cookies.get(COOKIE_NAME, "")):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"ok": True}
