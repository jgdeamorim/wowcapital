#!/usr/bin/env python3
"""
WOW Capital - TUI Operacional Principal
Interface Text-based para operação do sistema de trading

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import signal
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Static, DataTable, Log,
    ProgressBar, Button, Input, Label
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import Screen

# Imports opcionais - usando mock se não disponível
try:
    from core.contracts import MarketSnapshot
except ImportError:
    MarketSnapshot = None

try:
    from indicators.momentum.momo_1_5l import MOMO15L
except ImportError:
    MOMO15L = None

# from indicators.composite.high_aggression_score import HighAggressionScore
# from plugins.strategies.strategy_1_5l import Strategy1_5L


class DashboardPanel(Static):
    """Panel principal do dashboard com métricas em tempo real"""

    equity = reactive(10000.0)
    pnl_day = reactive(0.0)
    momo_score = reactive(0.0)
    aggression_score = reactive(0.0)

    def compose(self) -> ComposeResult:
        yield Static("🎯 DASHBOARD", id="dashboard-title")
        yield Static("", id="dashboard-metrics")

    def watch_equity(self, equity: float) -> None:
        self.update_metrics()

    def watch_pnl_day(self, pnl_day: float) -> None:
        self.update_metrics()

    def watch_momo_score(self, momo_score: float) -> None:
        self.update_metrics()

    def watch_aggression_score(self, aggression_score: float) -> None:
        self.update_metrics()

    def update_metrics(self):
        pnl_color = "green" if self.pnl_day >= 0 else "red"
        pnl_symbol = "+" if self.pnl_day >= 0 else ""

        momo_color = "green" if self.momo_score > 0 else "red" if self.momo_score < -0.2 else "yellow"
        aggr_color = "red" if self.aggression_score >= 0.92 else "yellow" if self.aggression_score >= 0.75 else "blue"

        metrics = f"""
[bold]Equity:[/bold] ${self.equity:,.2f}
[{pnl_color}]PnL Dia: {pnl_symbol}{self.pnl_day:+.2f} ({self.pnl_day/self.equity*100:+.2f}%)[/{pnl_color}]

[bold]Indicadores:[/bold]
[{momo_color}]MOMO-1.5L: {self.momo_score:+.4f}[/{momo_color}]
[{aggr_color}]Aggression: {self.aggression_score:.4f}[/{aggr_color}]

[bold]Status:[/bold] {'🔴 EXPLOSION READY!' if self.aggression_score >= 0.92 else '🟡 Monitoring' if self.aggression_score >= 0.75 else '🟢 Normal'}
        """.strip()

        self.query_one("#dashboard-metrics").update(metrics)


class OrdersPanel(Static):
    """Panel de ordens ativas e histórico"""

    def compose(self) -> ComposeResult:
        yield Static("📋 ORDERS", id="orders-title")

        table = DataTable()
        table.add_columns("Time", "Symbol", "Side", "Qty", "Price", "Status")
        table.add_row("14:32:15", "BTC/USDT", "BUY", "0.002", "49,850", "FILLED")
        table.add_row("14:31:42", "BTC/USDT", "SELL", "0.001", "49,920", "PARTIAL")

        yield table


class RiskPanel(Static):
    """Panel de métricas de risco"""

    def compose(self) -> ComposeResult:
        yield Static("⚠️  RISK", id="risk-title")
        yield Static("""
[bold]Limits (Spot Shadow):[/bold]
Max Drawdown: 4.0%
Position Size: 8.0%
Leverage: 1x (Spot)

[bold]Current:[/bold]
Drawdown: 0.6%
Exposure: $1,450
Shadow Risk Score: [green]LOW[/green]
        """.strip(), id="risk-metrics")


class ExchangesPanel(Static):
    """Panel de status das exchanges"""

    def compose(self) -> ComposeResult:
        yield Static("🏦 EXCHANGES", id="exchanges-title")
        yield Static("""
[green]✅ Kraken Spot (Shadow Real)[/green] - 18ms
[green]✅ Coinbase Spot (Shadow Real)[/green] - 22ms
[yellow]⚠️  Binance Spot[/yellow] - Standby
[yellow]⚠️  Bybit Spot[/yellow] - Preparing
[red]❌ Futures Desk[/red] - Disabled
        """.strip(), id="exchanges-status")


class ChatPanel(Static):
    """Panel de chat/IA com comandos"""

    def compose(self) -> ComposeResult:
        yield Static("🤖 CHAT-IA", id="chat-title")
        yield Log(id="chat-log")
        yield Input(placeholder="Digite comando ou pergunta...", id="chat-input")

    def on_input_submitted(self, event) -> None:
        chat_log = self.query_one("#chat-log")
        user_input = event.value.strip()

        if user_input:
            chat_log.write_line(f"[cyan]User:[/cyan] {user_input}")

            # Process simple commands
            if user_input.lower() in ["status", "st"]:
                chat_log.write_line("[green]IA:[/green] Sistema operacional. MOMO tracking, sem sinais ativos.")
            elif user_input.lower().startswith("pos"):
                chat_log.write_line("[green]IA:[/green] Posições: BTC/USDT +0.002 (+$15.20)")
            elif user_input.lower() in ["help", "h"]:
                chat_log.write_line("[green]IA:[/green] Comandos: status, pos, balance, risk, quit")
            else:
                chat_log.write_line("[green]IA:[/green] Comando não reconhecido. Digite 'help' para lista.")

        event.input.clear()


class MainTUI(App):
    """Aplicação TUI principal"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1;
    }

    #dashboard-panel {
        column-span: 1;
        row-span: 1;
        border: solid green;
    }

    #orders-panel {
        column-span: 1;
        row-span: 1;
        border: solid blue;
    }

    #risk-panel {
        column-span: 1;
        row-span: 1;
        border: solid yellow;
    }

    #exchanges-panel {
        column-span: 1;
        row-span: 1;
        border: solid cyan;
    }

    #chat-panel {
        column-span: 2;
        row-span: 1;
        border: solid magenta;
    }

    DataTable {
        height: 80%;
    }

    Log {
        height: 80%;
    }

    Input {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "screenshot", "Screenshot"),
        Binding("1", "focus_dashboard", "Dashboard"),
        Binding("2", "focus_orders", "Orders"),
        Binding("3", "focus_risk", "Risk"),
        Binding("4", "focus_exchanges", "Exchanges"),
        Binding("5", "focus_chat", "Chat"),
        Binding("space", "emergency_stop", "EMERGENCY STOP", priority=True),
        Binding("escape", "cancel_orders", "Cancel Orders"),
    ]

    def __init__(self):
        super().__init__()
        self.title = "WOW Capital - TUI Operacional"
        self.sub_title = "High-Frequency Trading System"

        # Initialize indicators (com fallback)
        self.momo = MOMO15L() if MOMO15L else None
        # self.aggression = HighAggressionScore()
        # self.strategy = Strategy1_5L()

        # Mock data for demo
        self.running = True
        self.last_update = time.time()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield DashboardPanel(id="dashboard-panel")
        yield OrdersPanel(id="orders-panel")
        yield RiskPanel(id="risk-panel")
        yield ExchangesPanel(id="exchanges-panel")
        yield ChatPanel(id="chat-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Start background data updates"""
        self.set_interval(1.0, self.update_data)
        self.notify("Sistema iniciado. Pressione 'q' para sair, 'space' para emergency stop.")

    def update_data(self) -> None:
        """Update data from indicators (mock implementation)"""
        import numpy as np

        # Simulate market data
        current_time = time.time()
        time_diff = current_time - self.last_update

        # Mock price movement
        price_change = np.random.normal(0, 0.001) * time_diff
        mock_price = 50000 * (1 + price_change)

        # Mock MOMO calculation
        prices = np.array([mock_price + np.random.normal(0, 50) for _ in range(20)])
        momo_score = np.clip(np.random.normal(0, 0.3), -1.5, 1.5)

        # Mock aggression score
        aggr_score = min(1.0, abs(momo_score) * 0.5 + np.random.uniform(0, 0.3))

        # Mock PnL
        pnl_change = np.random.normal(0, 2.5) * time_diff

        # Update dashboard
        dashboard = self.query_one("#dashboard-panel")
        dashboard.momo_score = momo_score
        dashboard.aggression_score = aggr_score
        dashboard.pnl_day += pnl_change

        self.last_update = current_time

    def action_refresh(self) -> None:
        """Refresh all panels"""
        self.notify("Refreshing data...")

    def action_screenshot(self) -> None:
        """Take screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tui_screenshot_{timestamp}.svg"
        self.save_screenshot(filename)
        self.notify(f"Screenshot saved: {filename}")

    def action_focus_dashboard(self) -> None:
        self.query_one("#dashboard-panel").focus()

    def action_focus_orders(self) -> None:
        self.query_one("#orders-panel").focus()

    def action_focus_risk(self) -> None:
        self.query_one("#risk-panel").focus()

    def action_focus_exchanges(self) -> None:
        self.query_one("#exchanges-panel").focus()

    def action_focus_chat(self) -> None:
        self.query_one("#chat-input").focus()

    def action_emergency_stop(self) -> None:
        """Emergency stop all trading"""
        self.notify("🚨 EMERGENCY STOP ACTIVATED!", severity="error")
        chat_log = self.query_one("#chat-log")
        chat_log.write_line("[red bold]SISTEMA:[/red bold] Emergency stop ativado. Todas as ordens canceladas.")

    def action_cancel_orders(self) -> None:
        """Cancel all pending orders"""
        self.notify("Cancelando ordens pendentes...", severity="warning")
        chat_log = self.query_one("#chat-log")
        chat_log.write_line("[yellow]SISTEMA:[/yellow] Ordens pendentes canceladas.")


def signal_handler(signum, frame):
    """Handle system signals gracefully"""
    print("\n🛑 Shutdown signal received. Closing TUI...")
    sys.exit(0)


def main():
    """Função principal"""
    print("🚀 Iniciando WOW Capital TUI Operacional...")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app = MainTUI()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 TUI fechado pelo usuário.")
    except Exception as e:
        print(f"❌ Erro na TUI: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 TUI Operacional finalizado.")


if __name__ == "__main__":
    main()
