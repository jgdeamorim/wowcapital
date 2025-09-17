from __future__ import annotations
"""Update pairs.yaml based on current liquidity/spread snapshots (best-effort).

Reads instruments.yaml for symbols list, fetches recent quotes from Redis (md:quote and md:micro),
and writes backend/config/pairs.yaml with allowed venues and simple scores.

Usage:
  PYTHONPATH=. python3 backend/tools/update_pairs.py

Env:
  REDIS_URL optional
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import asyncio
from backend.storage.redis_client import RedisClient
import argparse
from backend.common.config import load_yaml


async def score_symbol(rc: RedisClient, symbol: str, venues: list[str], *, samples: int = 30, interval_s: float = 2.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"symbol": symbol, "venues": []}
    for v in venues:
        sp_vals = []
        vq_vals = []
        for _ in range(samples):
            q = await rc.get_json(f"md:quote:{v}:{symbol}") or {}
            m = await rc.get_json(f"md:micro:{v}:{symbol}") or {}
            if q or m:
                if m and m.get("spread_bps") is not None:
                    sp = float(m.get("spread_bps", 0.0))
                else:
                    sp = (float(q.get("spread", 0.0)) / max(1e-9, float(q.get("mid", 0.0))) * 10_000.0) if q.get("mid") else 0.0
                sp_vals.append(sp)
                vq_vals.append(float((m or {}).get("vol_qty", 0.0)))
            await asyncio.sleep(interval_s)
        if not sp_vals:
            continue
        spread_bps = sum(sp_vals) / len(sp_vals)
        vol_qty = sum(vq_vals) / max(1, len(vq_vals))
        score = max(0.0, vol_qty) - spread_bps
        out["venues"].append({"venue": v, "spread_bps": spread_bps, "vol_qty": vol_qty, "score": score})
    out["venues"].sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out


async def main():
    parser = argparse.ArgumentParser(description="Update pairs.yaml based on micro/quote snapshots.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--venues", type=str, default="binance,bybit,kraken")
    args = parser.parse_args()
    rc = RedisClient()
    ins = load_yaml("backend/config/instruments.yaml")
    syms = [it.get("symbol") for it in ins.get("symbols", []) if it.get("symbol")]
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    results = []
    for s in syms:
        results.append(await score_symbol(rc, s, venues, samples=int(args.samples), interval_s=float(args.interval)))
    # Pick allowed venues by threshold
    pairs_cfg: Dict[str, Any] = {"pairs": []}
    for r in results:
        allowed = [v["venue"] for v in r.get("venues", []) if v.get("score", 0.0) > 0.0]
        pairs_cfg["pairs"].append({"symbol": r["symbol"], "allowed_venues": allowed, "scores": r.get("venues", [])})
    p = Path("backend/config/pairs.yaml")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(pairs_cfg, sort_keys=False, allow_unicode=True))
    tmp.replace(p)
    print(f"Updated {p}")


if __name__ == "__main__":
    asyncio.run(main())
