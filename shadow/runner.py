from __future__ import annotations
import asyncio
import json
from typing import List, Dict, Any, Optional
from backend.storage.redis_client import RedisClient
from backend.audit.worm import AsyncWormLog
from backend.observability.perf import PerfAggregator


class ShadowRunner:
    """Consumes strategy.decisions and simulates fills for configured SHADOW accounts.
    Redis config keys:
      shadow:accounts -> JSON list of {venue, account_id, mode}
    Subscribes channel: strategy.decisions
    """

    def __init__(self, redis: Optional[RedisClient] = None, worm: Optional[AsyncWormLog] = None, perf: Optional[PerfAggregator] = None):
        self._redis = redis or RedisClient()
        self._worm = worm or AsyncWormLog("backend/var/audit/trading.ndjson")
        self._perf = perf or PerfAggregator(self._redis, None)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self):
        if not self._redis.enabled() or self._task:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._stop.set()
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None

    async def _load_accounts(self) -> List[Dict[str, Any]]:
        try:
            v = await self._redis.client.get("shadow:accounts")  # type: ignore
            if v:
                return json.loads(v)
        except Exception:
            return []
        return []

    async def _loop(self):
        ps = None
        try:
            ps = self._redis.client.pubsub()  # type: ignore
            await ps.subscribe("strategy.decisions")
            while not self._stop.is_set():
                msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    data = json.loads(msg.get("data") or "{}")
                except Exception:
                    continue
                # Process decision for all shadow accounts configured
                accounts = await self._load_accounts()
                if not accounts:
                    continue
                await self._simulate_for_accounts(accounts, data)
        except asyncio.CancelledError:
            pass
        except Exception:
            await asyncio.sleep(0.2)
        finally:
            try:
                if ps:
                    await ps.unsubscribe("strategy.decisions")
                    await ps.close()
            except Exception:
                pass

    async def _simulate_for_accounts(self, accounts: List[Dict[str, Any]], decision: Dict[str, Any]):
        symbol = decision.get("symbol") or (decision.get("order") or {}).get("symbol") or "BTCUSDT"
        side = (decision.get("side") or (decision.get("order") or {}).get("side") or "BUY").upper()
        venue = (decision.get("venue") or (decision.get("order") or {}).get("venue") or "binance").lower()
        qty = float(decision.get("qty") or (decision.get("order") or {}).get("qty") or 0.0)
        idem = decision.get("idempotency_key") or (decision.get("order") or {}).get("idempotency_key")
        # fetch md snapshot
        mid = 0.0
        try:
            q = await self._redis.get_json(f"md:quote:{venue}:{symbol}")
            if q:
                bid = float(q.get("bid", 0))
                ask = float(q.get("ask", 0))
                mid = (bid + ask)/2.0 if bid and ask else 0.0
        except Exception:
            pass
        micro = await self._redis.get_json(f"md:micro:{venue}:{symbol}") or {}
        spread_bps = float(micro.get("spread_bps", 10.0))
        buy_ratio = float(micro.get("buy_ratio", 0.5))
        # simple slippage model
        base = max(2.0, min(spread_bps, 15.0))
        adj = 0.5 if (side == "BUY" and buy_ratio > 0.55) or (side == "SELL" and buy_ratio < 0.45) else 1.0
        slip_bps = base * adj
        px = mid
        if mid:
            sign = 1.0 if side == "BUY" else -1.0
            px = mid * (1.0 + sign * (slip_bps / 10_000.0))
        # Emit shadow fills per account
        for acc in accounts:
            acc_id = acc.get("account_id") or acc.get("account") or "acc#1"
            await self._worm.append({
                "ts_ns": int(asyncio.get_event_loop().time()*1e9),
                "event": "shadow_fill",
                "payload": {
                    "venue": venue,
                    "account": acc_id,
                    "symbol": symbol,
                    "side": side,
                    "price": px,
                    "qty": qty,
                    "idempotency_key": idem,
                }
            })
            try:
                await self._perf.record_fill(venue=venue, symbol=symbol, side=side, price=float(px or 0.0), qty=float(qty or 0.0), idem=idem)
            except Exception:
                pass
