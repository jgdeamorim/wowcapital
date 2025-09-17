#!/usr/bin/env python3
"""
Painel de Live Trading
Painel especializado para execução e monitoramento de trading ao vivo

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

from textual.widgets import Static, DataTable, Button, Input, Label
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.binding import Binding
from textual.app import ComposeResult

from ..components.indicators_widget import (
    MOMOWidget, AggressionWidget, PerformanceWidget
)


class LiveOrdersPanel(Static):
    """Painel de ordens ativas em tempo real"""

    orders = reactive([])
    selected_order = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Live Orders"
        # Mock orders data
        self.mock_orders = [
            {
                'id': 'ORD001',
                'timestamp': datetime.now() - timedelta(minutes=2),
                'symbol': 'BTC/USDT',
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': 0.002,
                'price': 49850.0,
                'filled_qty': 0.002,
                'status': 'FILLED',
                'pnl': 15.20,
                'strategy': '1.5L'
            },
            {
                'id': 'ORD002',
                'timestamp': datetime.now() - timedelta(minutes=1),
                'symbol': 'BTC/USDT',
                'side': 'SELL',
                'type': 'LIMIT',
                'quantity': 0.001,
                'price': 49920.0,
                'filled_qty': 0.0005,
                'status': 'PARTIAL',
                'pnl': 7.80,
                'strategy': '1.5L'
            }
        ]
        self.orders = self.mock_orders

    def compose(self) -> ComposeResult:
        table = DataTable(id="orders-table")
        table.add_columns(
            "Time", "ID", "Symbol", "Side", "Type",
            "Qty", "Price", "Filled", "Status", "PnL", "Strategy"
        )

        for order in self.orders:
            side_color = "green" if order['side'] == 'BUY' else "red"
            status_color = {
                'FILLED': 'green',
                'PARTIAL': 'yellow',
                'PENDING': 'blue',
                'CANCELLED': 'red'
            }.get(order['status'], 'white')

            pnl_color = "green" if order['pnl'] >= 0 else "red"
            pnl_symbol = "+" if order['pnl'] >= 0 else ""

            table.add_row(
                order['timestamp'].strftime("%H:%M:%S"),
                order['id'],
                order['symbol'],
                f"[{side_color}]{order['side']}[/{side_color}]",
                order['type'],
                f"{order['quantity']:.4f}",
                f"${order['price']:,.0f}",
                f"{order['filled_qty']:.4f}",
                f"[{status_color}]{order['status']}[/{status_color}]",
                f"[{pnl_color}]{pnl_symbol}${order['pnl']:.2f}[/{pnl_color}]",
                order['strategy']
            )

        yield table

        # Action buttons
        with Horizontal():
            yield Button("Cancel All", id="cancel-all", variant="error")
            yield Button("Emergency Stop", id="emergency-stop", variant="error")
            yield Button("Refresh", id="refresh-orders", variant="primary")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "cancel-all":
            self.notify("🚫 Cancelling all pending orders...", severity="warning")
        elif event.button.id == "emergency-stop":
            self.notify("🛑 EMERGENCY STOP ACTIVATED!", severity="error")
        elif event.button.id == "refresh-orders":
            self.notify("🔄 Refreshing orders...")


class PositionsPanel(Static):
    """Painel de posições abertas"""

    positions = reactive([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Open Positions"
        # Mock positions
        self.mock_positions = [
            {
                'symbol': 'BTC/USDT',
                'side': 'LONG',
                'size': 0.002,
                'entry_price': 49850.0,
                'current_price': 50020.0,
                'pnl': 0.34,
                'pnl_pct': 0.68,
                'leverage': 15.0,
                'margin': 166.33,
                'liquidation': 46800.0
            }
        ]
        self.positions = self.mock_positions

    def compose(self) -> ComposeResult:
        if not self.positions:
            yield Static("No open positions", classes="empty-state")
            return

        table = DataTable(id="positions-table")
        table.add_columns(
            "Symbol", "Side", "Size", "Entry", "Current",
            "PnL", "PnL%", "Leverage", "Margin", "Liquidation"
        )

        for pos in self.positions:
            side_color = "green" if pos['side'] == 'LONG' else "red"
            pnl_color = "green" if pos['pnl'] >= 0 else "red"
            pnl_symbol = "+" if pos['pnl'] >= 0 else ""
            pnl_pct_symbol = "+" if pos['pnl_pct'] >= 0 else ""

            table.add_row(
                pos['symbol'],
                f"[{side_color}]{pos['side']}[/{side_color}]",
                f"{pos['size']:.4f}",
                f"${pos['entry_price']:,.0f}",
                f"${pos['current_price']:,.0f}",
                f"[{pnl_color}]{pnl_symbol}${pos['pnl']:.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_pct_symbol}{pos['pnl_pct']:.2f}%[/{pnl_color}]",
                f"{pos['leverage']:.1f}x",
                f"${pos['margin']:.2f}",
                f"${pos['liquidation']:,.0f}"
            )

        yield table

        # Position actions
        with Horizontal():
            yield Button("Close All", id="close-all", variant="warning")
            yield Button("Reduce Size", id="reduce-size")
            yield Button("Add to Position", id="add-position", variant="success")


class SignalPanel(Static):
    """Painel de sinais de trading gerados"""

    signals = reactive([])
    auto_trade = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Trading Signals"
        # Mock recent signals
        self.mock_signals = [
            {
                'timestamp': datetime.now() - timedelta(seconds=30),
                'symbol': 'BTC/USDT',
                'signal': 'BUY',
                'strength': 'STRONG',
                'momo_score': 0.7234,
                'aggression': 0.8912,
                'confidence': 0.89,
                'target_price': 50200.0,
                'stop_loss': 49500.0,
                'executed': True
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'symbol': 'ETH/USDT',
                'signal': 'SELL',
                'strength': 'MEDIUM',
                'momo_score': -0.4123,
                'aggression': 0.6234,
                'confidence': 0.72,
                'target_price': 3180.0,
                'stop_loss': 3280.0,
                'executed': False
            }
        ]
        self.signals = self.mock_signals

    def compose(self) -> ComposeResult:
        # Auto-trade toggle
        with Horizontal():
            yield Label("Auto-Trade:")
            yield Button(
                "ON" if self.auto_trade else "OFF",
                id="auto-trade-toggle",
                variant="success" if self.auto_trade else "error"
            )

        # Signals table
        table = DataTable(id="signals-table")
        table.add_columns(
            "Time", "Symbol", "Signal", "Strength", "MOMO",
            "Aggr", "Conf", "Target", "Stop", "Executed"
        )

        for signal in self.signals:
            signal_color = "green" if signal['signal'] == 'BUY' else "red"
            strength_color = {
                'STRONG': 'bright_green',
                'MEDIUM': 'yellow',
                'WEAK': 'dim'
            }.get(signal['strength'], 'white')

            executed_color = "green" if signal['executed'] else "red"
            executed_icon = "✅" if signal['executed'] else "❌"

            table.add_row(
                signal['timestamp'].strftime("%H:%M:%S"),
                signal['symbol'],
                f"[{signal_color}]{signal['signal']}[/{signal_color}]",
                f"[{strength_color}]{signal['strength']}[/{strength_color}]",
                f"{signal['momo_score']:+.4f}",
                f"{signal['aggression']:.4f}",
                f"{signal['confidence']:.2%}",
                f"${signal['target_price']:,.0f}",
                f"${signal['stop_loss']:,.0f}",
                f"[{executed_color}]{executed_icon}[/{executed_color}]"
            )

        yield table

    def on_button_pressed(self, event) -> None:
        if event.button.id == "auto-trade-toggle":
            self.auto_trade = not self.auto_trade
            event.button.label = "ON" if self.auto_trade else "OFF"
            event.button.variant = "success" if self.auto_trade else "error"

            status = "ENABLED" if self.auto_trade else "DISABLED"
            severity = "success" if self.auto_trade else "warning"
            self.notify(f"🤖 Auto-Trade {status}", severity=severity)


class MarketDataPanel(Static):
    """Painel de dados de mercado em tempo real"""

    market_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Market Data"
        # Mock market data
        self.market_data = {
            'BTC/USDT': {
                'price': 50020.45,
                'change_24h': 2.34,
                'change_pct': 4.89,
                'volume_24h': 1234567890,
                'high_24h': 50180.0,
                'low_24h': 48920.0,
                'bid': 50018.20,
                'ask': 50022.70,
                'spread': 0.009
            },
            'ETH/USDT': {
                'price': 3245.67,
                'change_24h': -23.45,
                'change_pct': -0.72,
                'volume_24h': 987654321,
                'high_24h': 3278.90,
                'low_24h': 3201.45,
                'bid': 3244.80,
                'ask': 3246.54,
                'spread': 0.054
            }
        }

    def compose(self) -> ComposeResult:
        table = DataTable(id="market-data-table")
        table.add_columns(
            "Symbol", "Price", "24h Change", "24h %",
            "Volume", "High", "Low", "Bid", "Ask", "Spread"
        )

        for symbol, data in self.market_data.items():
            change_color = "green" if data['change_24h'] >= 0 else "red"
            change_symbol = "+" if data['change_24h'] >= 0 else ""
            pct_symbol = "+" if data['change_pct'] >= 0 else ""

            table.add_row(
                symbol,
                f"${data['price']:,.2f}",
                f"[{change_color}]{change_symbol}{data['change_24h']:.2f}[/{change_color}]",
                f"[{change_color}]{pct_symbol}{data['change_pct']:.2f}%[/{change_color}]",
                f"${data['volume_24h']:,.0f}",
                f"${data['high_24h']:,.2f}",
                f"${data['low_24h']:,.2f}",
                f"${data['bid']:,.2f}",
                f"${data['ask']:,.2f}",
                f"{data['spread']:.3%}"
            )

        yield table


class LiveTradingContainer(Container):
    """Container principal que organiza todos os painéis de live trading"""

    BINDINGS = [
        Binding("f1", "toggle_auto_trade", "Auto-Trade"),
        Binding("f2", "emergency_stop", "Emergency Stop"),
        Binding("f3", "cancel_all", "Cancel All"),
        Binding("f5", "refresh_data", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        # Top row: Indicators and performance
        with Horizontal():
            yield MOMOWidget()
            yield AggressionWidget()
            yield PerformanceWidget()

        # Middle row: Orders and positions
        with Horizontal():
            yield LiveOrdersPanel()
            yield PositionsPanel()

        # Bottom row: Signals and market data
        with Horizontal():
            yield SignalPanel()
            yield MarketDataPanel()

    def action_toggle_auto_trade(self) -> None:
        signal_panel = self.query_one(SignalPanel)
        signal_panel.auto_trade = not signal_panel.auto_trade
        status = "ENABLED" if signal_panel.auto_trade else "DISABLED"
        self.notify(f"🤖 Auto-Trade {status}")

    def action_emergency_stop(self) -> None:
        self.notify("🛑 EMERGENCY STOP ACTIVATED!", severity="error")

    def action_cancel_all(self) -> None:
        self.notify("🚫 Cancelling all orders...", severity="warning")

    def action_refresh_data(self) -> None:
        self.notify("🔄 Refreshing all data...")

    def on_mount(self) -> None:
        """Start real-time updates"""
        self.set_interval(1.0, self.update_real_time_data)

    def update_real_time_data(self) -> None:
        """Update all panels with real-time data (mock implementation)"""
        import random

        # Update market data with small random changes
        market_panel = self.query_one(MarketDataPanel)
        for symbol, data in market_panel.market_data.items():
            # Small price movements
            change_pct = random.uniform(-0.001, 0.001)
            data['price'] *= (1 + change_pct)
            data['bid'] = data['price'] - random.uniform(0.5, 2.0)
            data['ask'] = data['price'] + random.uniform(0.5, 2.0)
            data['spread'] = (data['ask'] - data['bid']) / data['price']

        # Update indicators
        momo_widget = self.query_one(MOMOWidget)
        momo_widget.score = random.uniform(-1.5, 1.5)
        momo_widget.confidence = random.uniform(0.5, 1.0)

        aggr_widget = self.query_one(AggressionWidget)
        aggr_widget.score = random.uniform(0.0, 1.0)
        aggr_widget.components = {
            'momentum': random.uniform(0, 1),
            'volatility': random.uniform(0, 1),
            'volume': random.uniform(0, 1),
            'technical': random.uniform(0, 1),
            'microstructure': random.uniform(0, 1),
            'sentiment': random.uniform(0, 1)
        }

        # Update performance
        perf_widget = self.query_one(PerformanceWidget)
        perf_widget.pnl_day += random.uniform(-1, 3)  # Slight positive bias