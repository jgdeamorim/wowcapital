from __future__ import annotations
import curses
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any
import httpx
import yaml


API_URL = Path("backend/config/api.yaml")
DEFAULT_API = "http://127.0.0.1:8080"
_API_CFG: Dict[str, Any] | None = None


def api_cfg() -> Dict[str, Any]:
    global _API_CFG
    if _API_CFG is not None:
        return _API_CFG
    url = DEFAULT_API
    key = None
    if API_URL.exists():
        try:
            cfg = yaml.safe_load(API_URL.read_text()) or {}
            url = cfg.get("url", DEFAULT_API)
            key = cfg.get("api_key")
        except Exception:
            pass
    _API_CFG = {"url": url, "api_key": key}
    return _API_CFG


def api_base() -> str:
    return api_cfg()["url"]


def headers() -> Dict[str, str]:
    cfg = api_cfg()
    h = {}
    if cfg.get("api_key"):
        h["X-API-Key"] = str(cfg.get("api_key"))
    return h


def draw_menu(stdscr, current: int, items: list[str]):
    stdscr.clear()
    stdscr.addstr(0, 2, "WOWCAPITAL TUI — setas ↑↓, Enter seleciona, q para sair")
    for idx, item in enumerate(items):
        if idx == current:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(2 + idx, 4, item)
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(2 + idx, 4, item)
    stdscr.refresh()


def prompt(stdscr, y: int, x: int, label: str, default: str = "") -> str:
    curses.echo()
    stdscr.addstr(y, x, f"{label} [{default}]: ")
    s = stdscr.getstr(y, x + len(label) + 3 + len(default), 60).decode("utf-8")
    curses.noecho()
    return s or default


def show_text(stdscr, title: str, body: str):
    stdscr.clear()
    stdscr.addstr(0, 2, title)
    lines = body.splitlines()
    h, w = stdscr.getmaxyx()
    max_lines = h - 2
    for i, ln in enumerate(lines[:max_lines]):
        stdscr.addstr(2 + i, 2, ln[: w - 4])
    stdscr.addstr(h - 1, 2, "Pressione qualquer tecla para voltar...")
    stdscr.getch()


def balances_view(stdscr):
    base = api_base()
    venue = prompt(stdscr, 2, 2, "Venue (auto/binance/bybit/kraken)", "auto")
    account = prompt(stdscr, 3, 2, "Account", "acc#1")
    url = f"{base}/exec/balances?venue={venue}&account={account}"
    data = httpx.get(url, timeout=10, headers=headers()).json()
    body = json.dumps(data, indent=2, ensure_ascii=False)
    show_text(stdscr, f"Balances — {venue}/{account}", body)


def positions_view(stdscr):
    base = api_base()
    venue = prompt(stdscr, 2, 2, "Venue (auto/binance/bybit/kraken)", "auto")
    account = prompt(stdscr, 3, 2, "Account", "acc#1")
    url = f"{base}/exec/positions?venue={venue}&account={account}"
    data = httpx.get(url, timeout=10, headers=headers()).json()
    body = json.dumps(data, indent=2, ensure_ascii=False)
    show_text(stdscr, f"Positions — {venue}/{account}", body)


def place_order_view(stdscr):
    base = api_base()
    venue = prompt(stdscr, 2, 2, "Venue", "binance")
    account = prompt(stdscr, 3, 2, "Account", "acc#1")
    symbol = prompt(stdscr, 4, 2, "Symbol", "BTCUSDT")
    side = prompt(stdscr, 5, 2, "Side (BUY/SELL)", "BUY").upper()
    qty = prompt(stdscr, 6, 2, "Qty", "0.001")
    idem = f"tui-{uuid.uuid4().hex[:8]}"
    order = {
        "symbol": symbol,
        "side": side,
        "qty": float(qty),
        "order_type": "MARKET",
        "client_id": "tui",
        "idempotency_key": idem,
        "meta": {"class": "CRYPTO", "account": account},
    }
    payload = {"venue": venue, "order": order}
    resp = httpx.post(f"{base}/exec/place", json=payload, timeout=15, headers=headers())
    body = resp.text
    show_text(stdscr, f"Order Ack — {venue}", body)


def cofre_view(stdscr):
    base = api_base()
    data = httpx.get(f"{base}/cofre", timeout=10, headers=headers()).json()
    show_text(stdscr, "Cofre", json.dumps(data, indent=2, ensure_ascii=False))


def cofre_ledger_view(stdscr):
    base = api_base()
    try:
        limit = prompt(stdscr, 2, 2, "Limite de registros (max 1000)", "100")
        url = f"{base}/cofre/ledger?limit={int(limit or '100')}"
    except Exception:
        url = f"{base}/cofre/ledger?limit=100"
    try:
        data = httpx.get(url, timeout=15, headers=headers()).json()
    except Exception as e:
        show_text(stdscr, "Cofre Ledger", f"Erro: {e}")
        return
    show_text(stdscr, "Cofre Ledger", json.dumps(data, indent=2, ensure_ascii=False))


def cofre_sweep_view(stdscr):
    base = api_base()
    venue = prompt(stdscr, 2, 2, "Venue (binance/bybit)", "binance").lower()
    amount = prompt(stdscr, 3, 2, "Amount USDT", "100")
    account = prompt(stdscr, 4, 2, "Account (opcional)", "acc#1")
    reason = prompt(stdscr, 5, 2, "Reason", "manual")
    try:
        payload = {
            "venue": venue,
            "amount_usdt": float(amount),
            "account": account or None,
            "reason": reason or "manual",
        }
    except Exception:
        show_text(stdscr, "Cofre Sweep", "Entrada inválida de amount.")
        return
    try:
        resp = httpx.post(f"{base}/cofre/sweep", json=payload, timeout=20)
        body = resp.text
    except Exception as e:
        body = f"Erro ao chamar /cofre/sweep: {e}"
    show_text(stdscr, f"Cofre Sweep — {venue}", body)


def plugins_view(stdscr):
    base = api_base()
    data = httpx.get(f"{base}/plugins", timeout=10, headers=headers()).json()
    show_text(stdscr, "Plugins", json.dumps(data, indent=2, ensure_ascii=False))


def decide_place_view(stdscr):
    base = api_base()
    plugin_id = prompt(stdscr, 2, 2, "Plugin ID", "jerico.binance.v1")
    venue = prompt(stdscr, 3, 2, "Venue", "binance")
    symbol = prompt(stdscr, 4, 2, "Symbol", "BTCUSDT")
    account = prompt(stdscr, 5, 2, "Account", "acc#1")
    # Minimal snapshot
    now_ns = int(time.time() * 1e9)
    snapshot = {
        "class_": "CRYPTO",
        "symbol": symbol,
        "ts_ns": now_ns,
        "bid": 0.0,
        "ask": 0.0,
        "mid": 0.0,
        "spread": 0.0,
        "features": {"signal": 1.0, "qty": 0.001, "account": account},
    }
    resp = httpx.post(f"{base}/orchestrator/place", json={"plugin_id": plugin_id, "venue": venue, "snapshot": snapshot}, timeout=20, headers=headers())
    show_text(stdscr, "Orchestrator Place", resp.text)


def audit_tail_view(stdscr):
    path = Path("backend/var/audit/trading.ndjson")
    if not path.exists():
        show_text(stdscr, "Audit Tail", "Arquivo de auditoria não encontrado.")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = "\n".join(lines[-200:])
    show_text(stdscr, "Audit Tail (últimas 200 linhas)", tail)


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)
    menu = [
        "Balances",
        "Positions",
        "Place Order",
        "Cofre",
        "Cofre Ledger",
        "Cofre Sweep",
        "Plugins",
        "Decide + Place",
        "Audit Tail",
        "Promotion Status",
        "Promotion Advance",
        "Exec Pause",
        "MD Micro",
        "MD Depth",
        "MD Pairs",
        "Backtest (Micro)",
    ]
    idx = 0
    while True:
        draw_menu(stdscr, idx, menu)
        ch = stdscr.getch()
        if ch in (ord('q'), ord('Q')):
            break
        elif ch == curses.KEY_UP:
            idx = (idx - 1) % len(menu)
        elif ch == curses.KEY_DOWN:
            idx = (idx + 1) % len(menu)
        elif ch in (10, 13):
            if idx == 0:
                balances_view(stdscr)
            elif idx == 1:
                positions_view(stdscr)
            elif idx == 2:
                place_order_view(stdscr)
            elif idx == 3:
                cofre_view(stdscr)
            elif idx == 4:
                cofre_ledger_view(stdscr)
            elif idx == 5:
                cofre_sweep_view(stdscr)
            elif idx == 6:
                plugins_view(stdscr)
            elif idx == 7:
                decide_place_view(stdscr)
            elif idx == 8:
                audit_tail_view(stdscr)
            elif idx == 9:
                promotion_status_view(stdscr)
            elif idx == 10:
                promotion_advance_view(stdscr)
            elif idx == 11:
                exec_pause_view(stdscr)
            elif idx == 12:
                md_micro_view(stdscr)
            elif idx == 13:
                md_depth_view(stdscr)
            elif idx == 14:
                md_pairs_view(stdscr)
            elif idx == 15:
                backtest_micro_view(stdscr)


def promotion_status_view(stdscr):
    base = api_base()
    plugin_id = prompt(stdscr, 2, 2, "Plugin ID", "example_ai_vwap")
    try:
        st = httpx.get(f"{base}/internal/promotion/status", params={"plugin_id": plugin_id}, timeout=10, headers=headers()).json()
        over = httpx.get(f"{base}/runtime/overrides", params={"ns": "risk", "plugin_id": plugin_id}, timeout=10, headers=headers()).json()
    except Exception as e:
        show_text(stdscr, "Promotion Status", f"Erro: {e}")
        return
    body = json.dumps({"status": st, "risk_overrides": over}, indent=2, ensure_ascii=False)
    show_text(stdscr, "Promotion Status", body)


def promotion_advance_view(stdscr):
    base = api_base()
    plugin_id = prompt(stdscr, 2, 2, "Plugin ID", "example_ai_vwap")
    sharpe = prompt(stdscr, 3, 2, "Sharpe", "1.0")
    maxdd = prompt(stdscr, 4, 2, "MaxDD", "0.2")
    trades = prompt(stdscr, 5, 2, "Trades", "50")
    try:
        metrics = {"sharpe": float(sharpe), "maxDD": float(maxdd), "trades": int(trades)}
        resp = httpx.post(f"{base}/internal/promotion/advance", json={"plugin_id": plugin_id, "metrics": metrics}, timeout=15, headers=headers())
        body = resp.text
    except Exception as e:
        body = f"Erro: {e}"
    show_text(stdscr, "Promotion Advance", body)


def exec_pause_view(stdscr):
    base = api_base()
    scope = prompt(stdscr, 2, 2, "Scope (all/venue/account)", "all")
    venue = ""
    account = ""
    if scope in ("venue", "account"):
        venue = prompt(stdscr, 3, 2, "Venue", "binance")
    if scope == "account":
        account = prompt(stdscr, 4, 2, "Account", "acc#1")
    on = prompt(stdscr, 5, 2, "On? (1/0)", "1")
    try:
        payload = {"scope": scope, "venue": venue or None, "account": account or None, "on": (on == "1")}
        resp = httpx.post(f"{base}/internal/exec/pause", json=payload, timeout=10, headers=headers())
        body = resp.text
    except Exception as e:
        body = f"Erro: {e}"
    show_text(stdscr, "Exec Pause", body)


def md_micro_view(stdscr):
    base = api_base()
    symbol = prompt(stdscr, 2, 2, "Symbol", "BTCUSDT")
    try:
        resp = httpx.get(f"{base}/md/micro", params={"symbol": symbol}, timeout=10, headers=headers())
        body = resp.text
    except Exception as e:
        body = f"Erro: {e}"
    show_text(stdscr, "MD Micro", body)


def md_depth_view(stdscr):
    base = api_base()
    symbol = prompt(stdscr, 2, 2, "Symbol", "BTCUSDT")
    try:
        resp = httpx.get(f"{base}/md/depth", params={"symbol": symbol}, timeout=10, headers=headers())
        body = resp.text
    except Exception as e:
        body = f"Erro: {e}"
    show_text(stdscr, "MD Depth", body)


def md_pairs_view(stdscr):
    base = api_base()
    limit = prompt(stdscr, 2, 2, "Limit", "10")
    venues = prompt(stdscr, 3, 2, "Venues (csv)", "binance,bybit,kraken")
    try:
        resp = httpx.get(f"{base}/md/pairs", params={"limit": int(limit or '10'), "venues": venues}, timeout=15, headers=headers())
        body = resp.text
    except Exception as e:
        body = f"Erro: {e}"
    show_text(stdscr, "MD Pairs", body)


def backtest_micro_view(stdscr):
    base = api_base()
    entrypoint = prompt(stdscr, 2, 2, "Entrypoint (module.py:Class)", "backend/plugins/strategies/example_ai_vwap/strategy.py:Strategy")
    symbol = prompt(stdscr, 3, 2, "Symbol", "BTCUSDT")
    venue = prompt(stdscr, 4, 2, "Venue (binance/bybit)", "binance")
    # candles
    if venue == "bybit":
        interval = prompt(stdscr, 5, 2, "Bybit interval (1=1m)", "1")
        limit = prompt(stdscr, 6, 2, "Candles limit", "200")
        c_url = f"{base}/md/candles/bybit?symbol={symbol}&interval={interval}&limit={limit}"
    else:
        interval = prompt(stdscr, 5, 2, "Binance interval (1m)", "1m")
        limit = prompt(stdscr, 6, 2, "Candles limit", "200")
        c_url = f"{base}/md/candles/binance?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        c = httpx.get(c_url, timeout=20, headers=headers()).json()
        candles = c.get("candles", [])
    except Exception as e:
        show_text(stdscr, "Backtest (Micro)", f"Erro ao obter candles: {e}")
        return
    # micro timeline
    samples = prompt(stdscr, 7, 2, "Micro samples", "120")
    sample_int = prompt(stdscr, 8, 2, "Micro interval (s)", "1.0")
    try:
        tl = httpx.get(f"{base}/md/micro/timeline_compact", params={"symbol": symbol, "venue": venue, "samples": int(samples or '120'), "interval": float(sample_int or '1.0')}, timeout=25, headers=headers()).json()
        micro_timeline = tl.get("timeline", [])
    except Exception as e:
        show_text(stdscr, "Backtest (Micro)", f"Erro ao obter micro timeline: {e}")
        return
    # backtest
    body = {
        "entrypoint": entrypoint,
        "symbol": symbol,
        "venue": venue,
        "candles": candles,
        "micro_timeline": micro_timeline,
    }
    try:
        resp = httpx.post(f"{base}/internal/strategy/backtest", json=body, timeout=30, headers=headers())
        out = resp.text
    except Exception as e:
        out = f"Erro no backtest: {e}"
    show_text(stdscr, "Backtest (Micro)", out)


if __name__ == "__main__":
    curses.wrapper(main)
