from __future__ import annotations
import asyncio
import json
import os
from typing import Dict, List
import time
import aiohttp
from backend.common.config import load_yaml
from backend.storage.redis_client import RedisClient
from backend.storage.mongo import MongoStore


class PublicWS:
    def __init__(self, redis: RedisClient | None = None, instruments_path: str | None = None):
        self._redis = redis or RedisClient()
        self._stop = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        # rolling trade buffers per (venue,symbol)
        self._tr_buf: Dict[str, Dict[str, List[tuple[float,float,int,float]]]] = {"binance": {}, "bybit": {}, "kraken": {}, "coinbase": {}}
        self._ins_path = instruments_path or os.getenv("INSTRUMENTS_FILE", "backend/config/instruments.yaml")
        self._mongo = MongoStore()
        self._last_persist: Dict[str, float] = {}
        self._agg_only = (os.getenv("MICRO_AGG_ONLY", "1") in ("1", "true", "TRUE"))
        try:
            self._agg_interval = float(os.getenv("MICRO_AGG_INTERVAL", "1.0"))
        except Exception:
            self._agg_interval = 1.0

    def _append_trade(self, venue: str, symbol: str, price: float, qty: float, is_buy: bool) -> None:
        now = asyncio.get_event_loop().time()
        sign = 1 if is_buy else -1
        buf = self._tr_buf.setdefault(venue, {}).setdefault(symbol, [])
        buf.append((price, qty, sign, now))
        # prune light here as well (safety)
        horizon = 10.0
        if len(buf) > 2048:
            buf[:] = [t for t in buf if now - t[3] <= horizon]

    def _symbols(self) -> List[str]:
        ins = load_yaml(self._ins_path)
        out: List[str] = []
        for it in ins.get("symbols", []):
            s = it.get("symbol")
            if s:
                out.append(s)
        return out

    def _vsym(self, symbol: str, venue: str, ins: Dict) -> str:
        for it in ins.get("symbols", []):
            if it.get("symbol") == symbol:
                return (it.get("venues") or {}).get(venue, symbol)
        return symbol

    async def start(self):
        if not self._redis.enabled():
            return
        ins = load_yaml(self._ins_path)
        syms = self._symbols()
        self._tasks = [
            asyncio.create_task(self._binance(ins, syms)),
            asyncio.create_task(self._binance_trades(ins, syms)),
            asyncio.create_task(self._binance_depth(ins, syms)),
            asyncio.create_task(self._bybit(ins, syms)),
            asyncio.create_task(self._bybit_trades(ins, syms)),
            asyncio.create_task(self._bybit_depth(ins, syms)),
            asyncio.create_task(self._kraken(ins, syms)),
            asyncio.create_task(self._kraken_trades(ins, syms)),
            asyncio.create_task(self._kraken_depth(ins, syms)),
            asyncio.create_task(self._coinbase_depth(ins, syms)),
            asyncio.create_task(self._coinbase_trades(ins, syms)),
            asyncio.create_task(self._aggregate_micro(ins, syms)),
        ]

    async def stop(self):
        self._stop.set()
        for t in self._tasks:
            t.cancel()

    async def _stash_set(self, key: str, value: Dict, ex: int = 1):
        try:
            prev = await self._redis.get_json(key)
            if prev is not None:
                await self._redis.set_json(key.replace("md:quote:", "md:last:"), prev, ex=2)
            await self._redis.set_json(key, value, ex=ex)
        except Exception:
            pass

    async def _update_depth(self, venue: str, symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]):
        """Compute top-5 aggregates and imbalance and store in Redis at md:depth:{venue}:{symbol}."""
        if not self._redis.enabled():
            return
        try:
            tb = sum(q for _, q in bids[:5])
            ta = sum(q for _, q in asks[:5])
            imb_l5 = ((tb - ta) / (tb + ta)) if (tb + ta) > 0 else 0.0
            out = {"venue": venue, "symbol": symbol, "top5_bid_qty": tb, "top5_ask_qty": ta, "imbalance_l5": imb_l5}
            await self._redis.set_json(f"md:depth:{venue}:{symbol}", out, ex=2)
        except Exception:
            pass

    async def _update_micro(self, venue: str, symbol: str, *, bid: float, ask: float, bid_sz: float | None = None, ask_sz: float | None = None):
        if not self._redis.enabled():
            return
        try:
            # compute imbalance from sizes if provided
            imb = None
            if bid_sz is not None and ask_sz is not None and (bid_sz + ask_sz) > 0:
                imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
            # compute trade-derived metrics from buffer (last ~10s)
            now = asyncio.get_event_loop().time()
            buf = self._tr_buf.setdefault(venue, {}).setdefault(symbol, [])
            # prune
            horizon = 10.0
            buf[:] = [t for t in buf if now - t[3] <= horizon]
            sum_q = sum(t[1] for t in buf)
            sum_pq = sum(t[0] * t[1] for t in buf)
            buy_q = sum(t[1] for t in buf if t[2] > 0)
            ofi = sum((1 if t[2] > 0 else -1) * t[1] for t in buf)
            vwap = (sum_pq / sum_q) if sum_q > 0 else 0.0
            buy_ratio = (buy_q / sum_q) if sum_q > 0 else 0.0
            spread = max(0.0, ask - bid)
            mid = (bid + ask) / 2.0 if bid and ask else 0.0
            spread_bps = (spread / mid * 10_000.0) if mid else 0.0
            # convexity proxy: larger when spread widens and buy_ratio flips -> approximate
            conv = 0.0
            try:
                last = await self._redis.get_json(f"md:micro:{venue}:{symbol}") or {}
                prev_spread_bps = float(last.get("spread_bps", 0.0))
                prev_buy_ratio = float(last.get("buy_ratio", 0.0))
                conv = max(-1.0, min(1.0, - (spread_bps - prev_spread_bps) / 10.0))
                if abs(buy_ratio - prev_buy_ratio) > 0.4 and (spread_bps - prev_spread_bps) > 2.0:
                    conv -= 0.3
            except Exception:
                pass
            # Merge L2 depth metrics if present
            imb_l5 = None
            top5_bid_qty = 0.0
            top5_ask_qty = 0.0
            try:
                d = await self._redis.get_json(f"md:depth:{venue}:{symbol}")
                if d:
                    imb_l5 = float(d.get("imbalance_l5", 0.0))
                    top5_bid_qty = float(d.get("top5_bid_qty", 0.0))
                    top5_ask_qty = float(d.get("top5_ask_qty", 0.0))
            except Exception:
                pass
            out = {
                "venue": venue,
                "symbol": symbol,
                "vwap": vwap,
                "vol_qty": sum_q,
                "buy_ratio": buy_ratio,
                "imbalance_l1": imb if imb is not None else None,
                "spread_bps": spread_bps,
                "convexity_proxy": conv,
                "ofi_rolling": ofi,
                "imbalance_l5": imb_l5 if imb_l5 is not None else None,
                "top5_bid_qty": top5_bid_qty,
                "top5_ask_qty": top5_ask_qty,
                "ts_ms": int(time.time()*1000),
            }
            # store
            await self._redis.set_json(f"md:micro:{venue}:{symbol}", out, ex=2)
            # Persist immediately only if not running in aggregator-only mode
            if not self._agg_only:
                try:
                    key = f"{venue}:{symbol}"
                    now = asyncio.get_event_loop().time()
                    if self._mongo.enabled() and (now - float(self._last_persist.get(key, 0.0))) > 0.5:
                        await self._mongo.record_micro(out)
                        self._last_persist[key] = now
                        try:
                            from backend.observability.metrics import MICRO_PERSIST_TOTAL
                            MICRO_PERSIST_TOTAL.labels(venue).inc()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    async def _aggregate_micro(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        venues = ["binance", "bybit", "kraken", "coinbase"]
        while not self._stop.is_set():
            try:
                if self._mongo.enabled():
                    for v in venues:
                        for s in syms:
                            it = await self._redis.get_json(f"md:micro:{v}:{s}") if self._redis.enabled() else None
                            if not it:
                                continue
                            key = f"{v}:{s}"
                            ts_ms = int(it.get("ts_ms", 0) or 0)
                            prev_ts = int(self._last_persist.get(key, 0) or 0)
                            # Avoid duplicate writes for same timestamp
                            if ts_ms and (ts_ms == prev_ts):
                                continue
                            await self._mongo.record_micro(it)
                            self._last_persist[key] = ts_ms or asyncio.get_event_loop().time()
                            try:
                                from backend.observability.metrics import MICRO_PERSIST_TOTAL
                                MICRO_PERSIST_TOTAL.labels(v).inc()
                            except Exception:
                                pass
                await asyncio.sleep(max(0.2, self._agg_interval))
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self._agg_interval)

    async def _binance(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        streams = "/".join([f"{self._vsym(s, 'binance', ins).lower()}@bookTicker" for s in syms])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("binance", "quote").set(1)
                        except Exception:
                            pass
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("binance", "quote").inc()
                                except Exception:
                                    pass
                                data = json.loads(msg.data)
                                it = data.get("data") or {}
                                bid = float(it.get("b", 0))
                                ask = float(it.get("a", 0))
                                vsym = (it.get("s") or "").upper()
                                # reverse map: assume symbol equal across or lookup
                                # We store under generic symbol if found
                                sym = None
                                for x in syms:
                                    if self._vsym(x, 'binance', ins).upper() == vsym:
                                        sym = x
                                        break
                                if sym:
                                    mid = (bid + ask) / 2.0 if bid and ask else 0.0
                                    await self._stash_set(f"md:quote:binance:{sym}", {"venue":"binance","symbol":sym,"bid":bid,"ask":ask,"mid":mid,"spread":max(0.0, ask-bid)}, ex=1)
                                    # sizes (if present)
                                    bsz = float(it.get("B") or 0)
                                    asz = float(it.get("A") or 0)
                                    await self._update_micro("binance", sym, bid=bid, ask=ask, bid_sz=bsz, ask_sz=asz)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("binance", "quote").inc()
                        WS_UP.labels("binance", "quote").set(0)
                    except Exception:
                        pass

    async def _binance_depth(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        streams = "/".join([f"{self._vsym(s, 'binance', ins).lower()}@depth5@100ms" for s in syms])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("binance", "trades").set(1)
                        except Exception:
                            pass
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("binance", "trades").inc()
                                except Exception:
                                    pass
                                data = json.loads(msg.data)
                                it = data.get("data") or {}
                                if isinstance(it, dict) and (it.get("e") in ("depthUpdate", None)):
                                    b = [(float(x[0]), float(x[1])) for x in (it.get("b") or [])]
                                    a = [(float(x[0]), float(x[1])) for x in (it.get("a") or [])]
                                    vsym = it.get("s") or (data.get("stream", "").split("@", 1)[0]).upper()
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'binance', ins).upper() == (vsym or '').upper():
                                            sym = x
                                            break
                                    if sym and (b or a):
                                        await self._update_depth("binance", sym, b, a)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("binance", "trades").inc()
                        WS_UP.labels("binance", "trades").set(0)
                    except Exception:
                        pass

    async def _binance_trades(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        streams = "/".join([f"{self._vsym(s, 'binance', ins).lower()}@aggTrade" for s in syms])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                it = data.get("data") or {}
                                if isinstance(it, dict) and it.get("e") == "aggTrade":
                                    p = float(it.get("p", 0))
                                    q = float(it.get("q", 0))
                                    # m = buyer is maker; if buyer is maker -> sell aggression
                                    is_buy = not bool(it.get("m", False))
                                    vsym = it.get("s", "")
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'binance', ins).upper() == vsym.upper():
                                            sym = x
                                            break
                                    if sym and p and q:
                                        self._append_trade("binance", sym, p, q, is_buy)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

    async def _bybit(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://stream.bybit.com/v5/public/spot"
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("bybit", "quote").set(1)
                        except Exception:
                            pass
                        # subscribe tickers for symbols
                        args = [self._vsym(s, 'bybit', ins) for s in syms]
                        await ws.send_json({"op": "subscribe", "args": [f"tickers.{a}" for a in args]})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("bybit", "quote").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, dict) and data.get("topic", "").startswith("tickers."):
                                    it = (data.get("data") or {})
                                    vsym = it.get("symbol")
                                    bid = float(it.get("bid1Price", 0))
                                    ask = float(it.get("ask1Price", 0))
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'bybit', ins).upper() == (vsym or '').upper():
                                            sym = x
                                            break
                                    if sym:
                                        mid = (bid + ask) / 2.0 if bid and ask else 0.0
                                        await self._stash_set(f"md:quote:bybit:{sym}", {"venue":"bybit","symbol":sym,"bid":bid,"ask":ask,"mid":mid,"spread":max(0.0, ask-bid)}, ex=1)
                                        bsz = float(it.get("bid1Size") or 0)
                                        asz = float(it.get("ask1Size") or 0)
                                        await self._update_micro("bybit", sym, bid=bid, ask=ask, bid_sz=bsz, ask_sz=asz)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("bybit", "quote").inc()
                        WS_UP.labels("bybit", "quote").set(0)
                    except Exception:
                        pass

    async def _bybit_depth(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://stream.bybit.com/v5/public/spot"
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        args = [self._vsym(s, 'bybit', ins) for s in syms]
                        await ws.send_json({"op": "subscribe", "args": [f"orderbook.50.{a}" for a in args]})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = msg.json()
                                if isinstance(data, dict) and data.get("topic", "").startswith("orderbook."):
                                    vsym = data.get("topic", "").split(".", 2)[-1]
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'bybit', ins).upper() == (vsym or '').upper():
                                            sym = x
                                            break
                                    it = (data.get("data") or {})
                                    b = [(float(p), float(q)) for p, q in (it.get("b") or [])]
                                    a = [(float(p), float(q)) for p, q in (it.get("a") or [])]
                                    if sym and (b or a):
                                        await self._update_depth("bybit", sym, b, a)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

    async def _bybit_trades(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://stream.bybit.com/v5/public/spot"
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        args = [self._vsym(s, 'bybit', ins) for s in syms]
                        await ws.send_json({"op": "subscribe", "args": [f"publicTrade.{a}" for a in args]})
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("bybit", "trades").set(1)
                        except Exception:
                            pass
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("bybit", "trades").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, dict) and (data.get("topic", "").startswith("publicTrade.")):
                                    for it in data.get("data", []) or []:
                                        p = float(it.get("p", 0))
                                        q = float(it.get("v", 0))
                                        side = (it.get("S") or it.get("s") or it.get("side") or "").upper()
                                        is_buy = side in ("BUY", "B")
                                        vsym = data.get("topic").split(".", 1)[1]
                                        sym = None
                                        for x in syms:
                                            if self._vsym(x, 'bybit', ins).upper() == vsym.upper():
                                                sym = x
                                                break
                                        if sym and p and q:
                                            self._append_trade("bybit", sym, p, q, is_buy)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("bybit", "trades").inc()
                        WS_UP.labels("bybit", "trades").set(0)
                    except Exception:
                        pass

    async def _coinbase_depth(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://ws-feed.exchange.coinbase.com"
        # L2 book is per-symbol so we manage one connection per symbol
        for s in syms:
            asyncio.create_task(self._coinbase_single_depth(ins, s))

    async def _coinbase_single_depth(self, ins: Dict, sym: str):
        url = "wss://ws-feed.exchange.coinbase.com"
        vsym = self._vsym(sym, 'coinbase', ins)
        backoff = 1.0
        l2_book = {'bids': {}, 'asks': {}}

        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("coinbase", "depth").inc()
                        except Exception:
                            pass
                        await ws.send_json({"type": "subscribe", "product_ids": [vsym], "channels": ["level2"]})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("coinbase", "depth").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, dict):
                                    mtype = data.get("type")
                                    if mtype == 'snapshot':
                                        l2_book['bids'] = {p: q for p, q in data.get('bids', [])}
                                        l2_book['asks'] = {p: q for p, q in data.get('asks', [])}
                                    elif mtype == 'l2update':
                                        for side, price, qty in data.get('changes', []):
                                            book_side = l2_book['bids'] if side == 'buy' else l2_book['asks']
                                            if float(qty) == 0:
                                                book_side.pop(price, None)
                                            else:
                                                book_side[price] = qty
                                    
                                    # Sort and format for downstream
                                    bids = sorted([(float(p), float(q)) for p, q in l2_book['bids'].items()], key=lambda x: x[0], reverse=True)
                                    asks = sorted([(float(p), float(q)) for p, q in l2_book['asks'].items()], key=lambda x: x[0])

                                    if bids and asks:
                                        await self._update_depth("coinbase", sym, bids, asks)
                                        bid, bsz = bids[0]
                                        ask, asz = asks[0]
                                        await self._stash_set(f"md:quote:coinbase:{sym}", {"venue":"coinbase","symbol":sym,"bid":bid,"ask":ask,"mid":(bid+ask)/2.0,"spread":max(0.0, ask-bid)}, ex=1)
                                        await self._update_micro("coinbase", sym, bid=bid, ask=ask, bid_sz=bsz, ask_sz=asz)

                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("coinbase", "depth").inc()
                        WS_UP.labels("coinbase", "depth").dec()
                    except Exception:
                        pass

    async def _coinbase_trades(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://ws-feed.exchange.coinbase.com"
        product_ids = [self._vsym(s, 'coinbase', ins) for s in syms]
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("coinbase", "trades").set(1)
                        except Exception:
                            pass
                        await ws.send_json({"type": "subscribe", "product_ids": product_ids, "channels": ["matches"]})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("coinbase", "trades").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, dict) and data.get("type") == "match":
                                    p = float(data.get("price", 0))
                                    q = float(data.get("size", 0))
                                    is_buy = (data.get("side") == "buy")
                                    vsym = data.get("product_id", "")
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'coinbase', ins).upper() == vsym.upper():
                                            sym = x
                                            break
                                    if sym and p and q:
                                        self._append_trade("coinbase", sym, p, q, is_buy)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("coinbase", "trades").inc()
                        WS_UP.labels("coinbase", "trades").set(0)
                    except Exception:
                        pass

    async def _kraken(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://ws.kraken.com/"
        pairs = [self._vsym(s, 'kraken', ins) for s in syms]
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("kraken", "quote").set(1)
                        except Exception:
                            pass
                        await ws.send_json({"event": "subscribe", "subscription": {"name": "ticker"}, "pair": pairs})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("kraken", "quote").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, list) and len(data) >= 4 and data[2] == "ticker":
                                    it = data[1]
                                    vsym = data[3]
                                    try:
                                        bid = float((it.get("b") or [0])[0])
                                        ask = float((it.get("a") or [0])[0])
                                    except Exception:
                                        bid = ask = 0.0
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'kraken', ins).upper() == (vsym or '').upper():
                                            sym = x
                                            break
                                    if sym:
                                        mid = (bid + ask) / 2.0 if bid and ask else 0.0
                                        await self._stash_set(f"md:quote:kraken:{sym}", {"venue":"kraken","symbol":sym,"bid":bid,"ask":ask,"mid":mid,"spread":max(0.0, ask-bid)}, ex=1)
                                        # Kraken provides lot volumes in b/a arrays
                                        bsz = float((it.get("b") or [0,0,0])[2] or 0)
                                        asz = float((it.get("a") or [0,0,0])[2] or 0)
                                        await self._update_micro("kraken", sym, bid=bid, ask=ask, bid_sz=bsz, ask_sz=asz)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("kraken", "quote").inc()
                        WS_UP.labels("kraken", "quote").set(0)
                    except Exception:
                        pass

    async def _kraken_depth(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://ws.kraken.com/"
        pairs = [self._vsym(s, 'kraken', ins) for s in syms]
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("kraken", "depth").set(1)
                        except Exception:
                            pass
                        await ws.send_json({"event": "subscribe", "subscription": {"name": "book", "depth": 10}, "pair": pairs})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("kraken", "depth").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, list) and len(data) >= 4 and str(data[2]).startswith("book"):
                                    vsym = data[3]
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'kraken', ins).upper() == (vsym or '').upper():
                                            sym = x
                                            break
                                    if not sym:
                                        continue
                                    book = data[1]
                                    # snapshot uses 'as'/'bs'; updates use 'a'/'b'
                                    asks = book.get('as') or book.get('a') or []
                                    bids = book.get('bs') or book.get('b') or []
                                    a = [(float(x[0]), float(x[1])) for x in asks]
                                    b = [(float(x[0]), float(x[1])) for x in bids]
                                    await self._update_depth("kraken", sym, b, a)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("kraken", "depth").inc()
                        WS_UP.labels("kraken", "depth").set(0)
                    except Exception:
                        pass

    async def _kraken_trades(self, ins: Dict, syms: List[str]):
        if not syms:
            return
        url = "wss://ws.kraken.com/"
        pairs = [self._vsym(s, 'kraken', ins) for s in syms]
        backoff = 1.0
        async with aiohttp.ClientSession() as sess:
            while not self._stop.is_set():
                try:
                    async with sess.ws_connect(url, heartbeat=15) as ws:
                        try:
                            from backend.observability.metrics import WS_UP
                            WS_UP.labels("kraken", "trades").set(1)
                        except Exception:
                            pass
                        await ws.send_json({"event": "subscribe", "subscription": {"name": "trade"}, "pair": pairs})
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    from backend.observability.metrics import WS_MESSAGES
                                    WS_MESSAGES.labels("kraken", "trades").inc()
                                except Exception:
                                    pass
                                data = msg.json()
                                if isinstance(data, list) and len(data) >= 4 and data[2] == "trade":
                                    trades = data[1] or []
                                    vsym = data[3]
                                    sym = None
                                    for x in syms:
                                        if self._vsym(x, 'kraken', ins).upper() == (vsym or '').upper():
                                            sym = x
                                            break
                                    for it in trades:
                                        try:
                                            p = float(it[0])
                                            q = float(it[1])
                                            side = (it[3] or "").lower()
                                            is_buy = (side == 'b')
                                            if sym and p and q:
                                                self._append_trade("kraken", sym, p, q, is_buy)
                                        except Exception:
                                            continue
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    try:
                        from backend.observability.metrics import WS_RECONNECTS, WS_UP
                        WS_RECONNECTS.labels("kraken", "trades").inc()
                        WS_UP.labels("kraken", "trades").set(0)
                    except Exception:
                        pass
