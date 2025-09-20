from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException
from starlette.requests import Request
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import hashlib
import json
import httpx
from backend.storage.redis_client import RedisClient
from backend.observability.metrics import AI_TOOL_CALLS
from backend.common.auth import require_scopes
from backend.common.config import load_models
from backend.common.secure_env import get_secret
from backend.ai_router.router import execute_trade, ExecuteTradeArgs


app = FastAPI(title="AI Gateway (Secure)")
_redis = RedisClient()


SYSTEM_PROMPT = (
    "Você é a IA orquestradora do WOWCAPITAL. "
    "Retorne SOMENTE chamadas de função válidas quando pedir execução. "
    "Siga as regras de risco e não exponha segredos."
)

EXECUTE_TRADE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_trade",
        "description": "Solicita execução de ordem. Sempre passar idempotency_token.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "exchange": {"type": "string"},
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "size": {"type": "number"},
                "order_type": {"type": "string", "enum": ["MARKET", "LIMIT"]},
                "price": {"type": ["number", "null"]},
                "reason": {"type": "string"},
                "idempotency_token": {"type": "string"}
            },
            "required": ["account_id", "exchange", "symbol", "side", "size", "order_type", "idempotency_token"]
        }
    }
}


class LLMOrchestrateIn(BaseModel):
    prompt: str
    symbol: Optional[str] = None
    execute: bool = False


@app.post("/ai/orchestrate")
async def ai_orchestrate(body: LLMOrchestrateIn, request: Request, _: None = Depends(require_scopes(["read"]))):
    cfg = load_models()
    model = ((cfg.get("models") or {}).get("gpt41mini_primary") or {}).get("model", "gpt-4.1-mini")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body.prompt}]
    cache_key = None
    if _redis.enabled():
        cache_key = f"ai:orch:{model}:{hashlib.sha256(body.prompt.encode()).hexdigest()}"
        cached = await _redis.get_json(cache_key)
        if cached and not body.execute:
            return cached
    try:
        api_key = get_secret("OPENAI_API_KEY")
    except Exception:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "tools": [EXECUTE_TRADE_TOOL],
        "tool_choice": "auto",
        "max_tokens": 600,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=payload)
            data = r.json()
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=str(data))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    out: Dict[str, Any] = {"model": model, "tool_calls": [], "executed": []}
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        name = (tc.get("function") or {}).get("name")
        args_s = (tc.get("function") or {}).get("arguments") or "{}"
        try:
            args = json.loads(args_s)
        except Exception:
            args = {}
        out["tool_calls"].append({"name": name, "arguments": args})
        try:
            AI_TOOL_CALLS.labels(name or "unknown", "suggested").inc()
        except Exception:
            pass
    if _redis.enabled() and cache_key and not body.execute:
        await _redis.set_json(cache_key, out, ex=30)
    if body.execute and out["tool_calls"]:
        # Enforce trade scope manually when executing
        if os.getenv("WOW_AUTH_BYPASS", "0") not in ("1", "true", "TRUE"):
            from backend.common.auth import _keys_map  # type: ignore
            keys = _keys_map()
            api_key = request.headers.get("X-API-Key")
            scopes = keys.get(api_key or "", [])
            if "*" not in scopes and "trade" not in scopes:
                raise HTTPException(status_code=403, detail="insufficient scope")
        results = []
        for call in out["tool_calls"]:
            if call.get("name") != "execute_trade":
                continue
            try:
                et = ExecuteTradeArgs(**(call.get("arguments") or {}))
            except Exception as e:
                results.append({"status": "error", "error": "validation", "detail": str(e)})
                try:
                    AI_TOOL_CALLS.labels(call.get("name") or "unknown", "validation_error").inc()
                except Exception:
                    pass
                continue
            try:
                ack = await execute_trade(et)
                results.append({"status": "ok", "ack": ack})
                try:
                    AI_TOOL_CALLS.labels(call.get("name") or "unknown", "executed").inc()
                except Exception:
                    pass
            except HTTPException:
                raise
            except Exception as e:
                results.append({"status": "error", "error": "execution", "detail": str(e)})
                try:
                    AI_TOOL_CALLS.labels(call.get("name") or "unknown", "execution_error").inc()
                except Exception:
                    pass
        out["executed"] = results
    return out
