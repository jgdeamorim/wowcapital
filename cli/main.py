from __future__ import annotations
import asyncio
import typer
from typing import Optional
from backend.execution.gateway import OrderRouter
from backend.orchestrator.plugin_manager import PluginManager
from backend.core.contracts import OrderRequest


app = typer.Typer(help="WOWCAPITAL CLI — operações básicas")


@app.command()
def plugins():
    pm = PluginManager()
    # In a real setup we'd load manifests from dir; here list loaded (empty by default)
    for pid in pm.loaded.keys():
        typer.echo(pid)


@app.command()
def balances(venue: str, account: Optional[str] = None):
    async def _run():
        r = OrderRouter()
        bals = await r.balances(venue, account=account)
        for k, v in bals.items():
            typer.echo(f"{k}: free={v.free} used={v.used} total={v.total}")
    asyncio.run(_run())


@app.command()
def positions(venue: str, account: Optional[str] = None):
    async def _run():
        r = OrderRouter()
        pos = await r.positions(venue, account=account)
        for k, v in pos.items():
            typer.echo(f"{k}: qty={v.qty} side={v.side}")
    asyncio.run(_run())


@app.command()
def place(venue: str, symbol: str, side: str = "BUY", qty: float = 0.001, account: Optional[str] = None):
    async def _run():
        r = OrderRouter()
        req = OrderRequest(symbol=symbol, side=side.upper(), qty=qty, order_type="MARKET", client_id="cli", idempotency_key=f"cli-{symbol}", meta={"class": "CRYPTO", "account": account or "acc#1"})
        ack = await r.place(venue, req)
        typer.echo(ack)
    asyncio.run(_run())


if __name__ == "__main__":
    app()

