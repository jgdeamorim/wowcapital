from __future__ import annotations
import asyncio
import os
from time import time
from backend.orchestrator.plugin_manager import PluginManager
from backend.execution.gateway import OrderRouter
from backend.core.contracts import OrderRequest


async def run_once(manifest_path: str, venue: str, symbol: str, side: str = "BUY", qty: float = 0.001):
    pm = PluginManager()
    m = pm.load_manifest(manifest_path)
    router = OrderRouter()
    req = OrderRequest(
        symbol=symbol,
        side=side,  # BUY/SELL
        qty=qty,
        order_type="MARKET",
        client_id=m.metadata.get("id", "plugin"),
        idempotency_key=f"{m.metadata.get('id','plugin')}-{int(time())}",
        meta={"class": "CRYPTO", "account": os.getenv("ACCOUNT_REF", "acc#1")},
    )
    ack = await router.place(venue, req)
    return ack


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("venue")
    parser.add_argument("symbol")
    parser.add_argument("--side", default="BUY")
    parser.add_argument("--qty", type=float, default=0.001)
    args = parser.parse_args()
    print(asyncio.run(run_once(args.manifest, args.venue, args.symbol, args.side, args.qty)))
