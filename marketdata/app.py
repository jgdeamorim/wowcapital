from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
import httpx
from typing import Dict
from backend.common.config import load_yaml
import os
from backend.storage.redis_client import RedisClient
from backend.common.pairs import PairsConfig
import asyncio
import time


md_app = APIRouter()
_redis = RedisClient()


async def _stash_prev_and_set(key: str, value: Dict, ex: int = 1) -> None:
    if not _redis.enabled():
        return
    try:
        prev = await _redis.get_json(key)
        if prev is not None:
            # keep a short-lived previous snapshot for synthetic indicators
            await _redis.set_json(key.replace("md:quote:", "md:last:"), prev, ex=2)
        await _redis.set_json(key, value, ex=ex)
    except Exception:
        # best-effort cache
        pass


def _venue_symbol(symbol: str, venue: str) -> str:
    ins_path = os.getenv("INSTRUMENTS_FILE", "backend/config/instruments.yaml")
    ins = load_yaml(ins_path)
    for it in ins.get("symbols", []):
        if it.get("symbol") == symbol:
            vmap = (it.get("venues") or {})
            return vmap.get(venue, symbol)
    return symbol


async def _fetch_quote_binance(symbol: str) -> Dict:
    vsym = _venue_symbol(symbol, "binance")
    # cache lookup
    if _redis.enabled():
        key = f"md:quote:binance:{symbol}"
        cached = await _redis.get_json(key)
        if cached:
            return cached
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://api.binance.com/api/v3/ticker/bookTicker", params={"symbol": vsym})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"binance error: {r.text}")
        data = r.json()
        bid = float(data.get("bidPrice", 0))
        ask = float(data.get("askPrice", 0))
        mid = (bid + ask) / 2.0 if bid and ask else 0.0
        quote = {"venue": "binance", "symbol": symbol, "bid": bid, "ask": ask, "mid": mid, "spread": max(0.0, ask - bid)}
        if _redis.enabled():
            await _stash_prev_and_set(key, quote, ex=1)
        return quote


async def _fetch_quote_bybit(symbol: str) -> Dict:
    vsym = _venue_symbol(symbol, "bybit")
    if _redis.enabled():
        key = f"md:quote:bybit:{symbol}"
        cached = await _redis.get_json(key)
        if cached:
            return cached
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://api.bybit.com/v5/market/tickers", params={"category": "spot", "symbol": vsym})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"bybit error: {r.text}")
        data = r.json()
        res = (data.get("result") or {}).get("list", [])
        if not res:
            raise HTTPException(status_code=404, detail="bybit symbol not found")
        it = res[0]
        bid = float(it.get("bid1Price", 0))
        ask = float(it.get("ask1Price", 0))
        mid = (bid + ask) / 2.0 if bid and ask else 0.0
        quote = {"venue": "bybit", "symbol": symbol, "bid": bid, "ask": ask, "mid": mid, "spread": max(0.0, ask - bid)}
        if _redis.enabled():
            await _stash_prev_and_set(key, quote, ex=1)
        return quote


async def _fetch_quote_kraken(symbol: str) -> Dict:
    vsym = _venue_symbol(symbol, "kraken")
    if _redis.enabled():
        key = f"md:quote:kraken:{symbol}"
        cached = await _redis.get_json(key)
        if cached:
            return cached
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://api.kraken.com/0/public/Ticker", params={"pair": vsym})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"kraken error: {r.text}")
        data = r.json()
        res = (data.get("result") or {})
        if not res:
            raise HTTPException(status_code=404, detail="kraken symbol not found")
        # take first key
        k = next(iter(res.keys()))
        it = res[k]
        bid = float((it.get("b") or [0])[0])
        ask = float((it.get("a") or [0])[0])
        mid = (bid + ask) / 2.0 if bid and ask else 0.0
        quote = {"venue": "kraken", "symbol": symbol, "bid": bid, "ask": ask, "mid": mid, "spread": max(0.0, ask - bid)}
        if _redis.enabled():
            await _stash_prev_and_set(key, quote, ex=1)
        return quote


@md_app.get("/quote")
async def quote(symbol: str, venue: str):
    if venue == "binance":
        return await _fetch_quote_binance(symbol)
    if venue == "bybit":
        return await _fetch_quote_bybit(symbol)
    if venue == "kraken":
        return await _fetch_quote_kraken(symbol)
    raise HTTPException(status_code=400, detail="unknown venue")


@md_app.get("/quote/auto")
async def quote_auto(symbol: str):
    pairs = PairsConfig()
    errors: Dict[str, str] = {}
    for v in pairs.allowed_venues(symbol):
        try:
            if v == "binance":
                return await _fetch_quote_binance(symbol)
            if v == "bybit":
                return await _fetch_quote_bybit(symbol)
            if v == "kraken":
                return await _fetch_quote_kraken(symbol)
        except HTTPException as e:
            errors[v] = str(e.detail)
            continue
    raise HTTPException(status_code=502, detail={"error": "no venue available", "details": errors})


@md_app.get("/micro")
async def micro(symbol: str):
    venues = ["binance", "bybit", "kraken"]
    data: Dict[str, Dict] = {}
    weights = {}
    for v in venues:
        it = await _redis.get_json(f"md:micro:{v}:{symbol}") if _redis.enabled() else None
        if it:
            data[v] = it
            weights[v] = float(it.get("vol_qty", 0.0))
    if not data:
        return {"symbol": symbol, "venues": {}, "consolidated": {}}
    # weighted consolidation by vol_qty
    wsum = sum(weights.values()) or 1.0
    cons = {
        "spread_bps": sum((it.get("spread_bps", 0.0) or 0.0) * (weights.get(v, 0.0)/wsum) for v, it in data.items()),
        "buy_ratio": sum((it.get("buy_ratio", 0.0) or 0.0) * (weights.get(v, 0.0)/wsum) for v, it in data.items()),
        "vwap": sum((it.get("vwap", 0.0) or 0.0) * (weights.get(v, 0.0)/wsum) for v, it in data.items()),
        "vol_qty": wsum,
        "imbalance_l1": sum(((it.get("imbalance_l1") or 0.0) * (weights.get(v, 0.0)/wsum)) for v, it in data.items()),
        "ofi_rolling": sum(((it.get("ofi_rolling") or 0.0) * (weights.get(v, 0.0)/wsum)) for v, it in data.items()),
    }
    return {"symbol": symbol, "venues": data, "consolidated": cons}


@md_app.get("/depth")
async def depth(symbol: str):
    venues = ["binance", "bybit", "kraken"]
    out: Dict[str, Dict] = {}
    for v in venues:
        it = await _redis.get_json(f"md:depth:{v}:{symbol}") if _redis.enabled() else None
        if it:
            out[v] = it
    # consolidated imbalance (weighted by top5 volume)
    wsum = sum((it.get("top5_bid_qty", 0.0) + it.get("top5_ask_qty", 0.0)) for it in out.values()) or 1.0
    cons = {
        "imbalance_l5": sum(((it.get("imbalance_l5") or 0.0) * ((it.get("top5_bid_qty", 0.0) + it.get("top5_ask_qty", 0.0)) / wsum)) for it in out.values())
    }
    return {"symbol": symbol, "venues": out, "consolidated": cons}


@md_app.get("/pairs")
async def pairs(limit: int = 20, venues: str | None = None, symbols: str | None = None, with_scores: bool = False):
    """Return a quick ranking snapshot of pairs by current micro (vol_qty - spread_bps)."""
    ins = load_yaml(os.getenv("INSTRUMENTS_FILE", "backend/config/instruments.yaml"))
    syms_all = [it.get("symbol") for it in (ins.get("symbols") or []) if it.get("symbol")]
    if symbols:
        syms = [s.strip() for s in symbols.split(",") if s.strip() and s.strip() in syms_all]
    else:
        syms = syms_all
    vlist = [v.strip() for v in (venues.split(",") if venues else ["binance","bybit","kraken"]) if v.strip()]
    scores: Dict[str, Dict] = {}
    for s in syms:
        per_v = []
        for v in vlist:
            m = await _redis.get_json(f"md:micro:{v}:{s}") if _redis.enabled() else None
            if not m:
                continue
            sp = float(m.get("spread_bps", 0.0) or 0.0)
            vq = float(m.get("vol_qty", 0.0) or 0.0)
            per_v.append({"venue": v, "spread_bps": sp, "vol_qty": vq, "score": (vq - sp)})
        if per_v:
            per_v.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            scores[s] = {"symbol": s, "venues": per_v, "best": per_v[0]}
    # order by best score
    ordered = sorted(scores.values(), key=lambda x: x.get("best", {}).get("score", 0.0), reverse=True)
    top = ordered[: max(1, min(limit, len(ordered)))]
    if not with_scores:
        # return concise view (symbol + best)
        compact = [{"symbol": t["symbol"], "best": t.get("best", {})} for t in top]
        return {"count": len(ordered), "top": compact}
    return {"count": len(ordered), "top": top}


@md_app.get("/micro/timeline")
async def micro_timeline(symbol: str, venue: str = "binance", samples: int = 60, interval: float = 1.0, fields: str | None = None):
    if not _redis.enabled():
        return {"symbol": symbol, "venue": venue, "timeline": []}
    allow: set[str] | None = None
    if fields:
        allow = {f.strip() for f in fields.split(",") if f.strip()}
    tl = []
    for _ in range(max(1, samples)):
        it = await _redis.get_json(f"md:micro:{venue}:{symbol}") or {}
        if it:
            if allow:
                it = {k: v for k, v in it.items() if k in allow or k in ("ts_ms","venue","symbol")}
            else:
                # ensure a timestamp exists
                it.setdefault("ts_ms", int(time.time() * 1000))
            tl.append(it)
        await asyncio.sleep(max(0.0, float(interval)))
    return {"symbol": symbol, "venue": venue, "timeline": tl}


@md_app.get("/micro/history")
async def micro_history(symbol: str, venue: str = "binance", minutes: int = 10, limit: int = 600):
    from backend.storage.mongo import MongoStore
    m = MongoStore()
    if not m.enabled():
        return {"symbol": symbol, "venue": venue, "timeline": []}
    try:
        from datetime import datetime, timezone, timedelta
        since = datetime.now(timezone.utc) - timedelta(minutes=max(1, minutes))
        cur = m.db.micro.find({"venue": venue, "symbol": symbol, "ts_dt": {"$gte": since}}).sort("ts_dt", 1).limit(min(int(limit), 2000))
        out = []
        async for doc in cur:
            doc.pop("_id", None)
            out.append(doc)
        return {"symbol": symbol, "venue": venue, "timeline": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@md_app.get("/micro/export")
async def micro_export(symbol: str, venue: str = "binance", minutes: int = 10, limit: int = 600, fmt: str = "ndjson", fields: str | None = None):
    from backend.storage.mongo import MongoStore
    m = MongoStore()
    if not m.enabled():
        raise HTTPException(status_code=500, detail="mongo disabled")
    try:
        from datetime import datetime, timezone, timedelta
        since = datetime.now(timezone.utc) - timedelta(minutes=max(1, minutes))
        cur = m.db.micro.find({"venue": venue, "symbol": symbol, "ts_dt": {"$gte": since}}).sort("ts_dt", 1).limit(min(int(limit), 10000))
        allow: set[str] | None = None
        if fields:
            allow = {f.strip() for f in fields.split(",") if f.strip()}
        rows = []
        async for doc in cur:
            doc.pop("_id", None)
            if allow:
                doc = {k: v for k, v in doc.items() if k in allow}
            rows.append(doc)
        if fmt.lower() == "csv":
            if not rows:
                return PlainTextResponse("")
            # construct header from keys of first row
            cols = list(rows[0].keys())
            lines = [",".join(cols)]
            for r in rows:
                lines.append(",".join(str(r.get(c, "")) for c in cols))
            return PlainTextResponse("\n".join(lines), media_type="text/csv")
        # ndjson default
        import json
        lines = [json.dumps(r, separators=(",", ":")) for r in rows]
        return PlainTextResponse("\n".join(lines), media_type="application/x-ndjson")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@md_app.get("/micro/timeline_compact")
async def micro_timeline_compact(symbol: str, venue: str = "binance", samples: int = 60, interval: float = 1.0):
    # Force compact fields for backtesting efficiency
    return await micro_timeline(symbol=symbol, venue=venue, samples=samples, interval=interval, fields="spread_bps,buy_ratio,imbalance_l5")


@md_app.get("/candles/binance")
async def candles_binance(symbol: str, interval: str = "1m", limit: int = 200):
    vsym = _venue_symbol(symbol, "binance")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://api.binance.com/api/v3/klines", params={"symbol": vsym, "interval": interval, "limit": min(max(1, limit), 1000)})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=r.text)
        data = r.json()
        # Binance kline format: [openTime, open, high, low, close, volume, closeTime, ...]
        out = []
        for it in data:
            out.append({"ts": int(it[0]), "o": float(it[1]), "h": float(it[2]), "l": float(it[3]), "c": float(it[4])})
        return {"symbol": symbol, "venue": "binance", "interval": interval, "candles": out}


@md_app.get("/candles/bybit")
async def candles_bybit(symbol: str, interval: str = "1", limit: int = 200):
    # Bybit v5: interval in minutes as strings: 1,3,5,15,30,60,120,240,360,720,D,W,M
    vsym = _venue_symbol(symbol, "bybit")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://api.bybit.com/v5/market/kline", params={"category": "spot", "symbol": vsym, "interval": interval, "limit": min(max(1, limit), 1000)})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=r.text)
        data = r.json()
        lst = ((data.get("result") or {}).get("list") or [])
        out = []
        for it in lst:
            # [ start, open, high, low, close, volume, turnover ]
            out.append({"ts": int(it[0]), "o": float(it[1]), "h": float(it[2]), "l": float(it[3]), "c": float(it[4])})
        out.sort(key=lambda x: x["ts"])  # ensure ascending
        return {"symbol": symbol, "venue": "bybit", "interval": interval, "candles": out}
