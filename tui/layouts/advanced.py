#!/usr/bin/env python3
"""
Layout Avançado para TUI Operacional
Layouts mais sofisticados com múltiplos painéis e widgets especializados

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Header, Footer, Static, DataTable, Log, Button,
    Input, Label, ProgressBar, TabbedContent, TabPane
)
from textual.binding import Binding
from textual.screen import Screen
from textual.reactive import reactive

from ..components.indicators_widget import (
    MOMOWidget, AggressionWidget, RegimeWidget,
    TechnicalSummaryWidget, PerformanceWidget
)


class TradingScreen(Screen):
    """Screen principal de trading com layout grid avançado"""

    CSS = """
    TradingScreen {
        layout: grid;
        grid-size: 4 3;
        grid-gutter: 1;
    }

    #performance-panel {
        column-span: 2;
        row-span: 1;
        border: solid green;
    }

    #technical-panel {
        column-span: 2;
        row-span: 1;
        border: solid blue;
    }

    #momo-widget {
        column-span: 1;
        row-span: 1;
        border: solid yellow;
    }

    #aggression-widget {
        column-span: 1;
        row-span: 1;
        border: solid red;
    }

    #regime-widget {
        column-span: 1;
        row-span: 1;
        border: solid magenta;
    }

    #orders-panel {
        column-span: 1;
        row-span: 1;
        border: solid cyan;
    }

    #chart-panel {
        column-span: 2;
        row-span: 1;
        border: solid white;
    }

    #log-panel {
        column-span: 2;
        row-span: 1;
        border: solid #444444;
    }

    DataTable {
        height: 90%;
    }

    Log {
        height: 85%;
    }

    .chart-container {
        height: 90%;
        background: #000011;
    }
    """

    def compose(self) -> ComposeResult:
        # Performance panel
        yield PerformanceWidget(id="performance-panel")

        # Technical summary
        yield TechnicalSummaryWidget(id="technical-panel")

        # Individual indicator widgets
        yield MOMOWidget(id="momo-widget")
        yield AggressionWidget(id="aggression-widget")
        yield RegimeWidget(id="regime-widget")

        # Orders panel
        yield self.create_orders_panel()

        # Chart panel (placeholder)
        yield self.create_chart_panel()

        # Log panel
        yield self.create_log_panel()

    def create_orders_panel(self) -> Container:
        """Criar painel de ordens com tabela"""
        container = Container(id="orders-panel")

        with container:
            yield Static("📋 ACTIVE ORDERS", classes="panel-title")

            table = DataTable()
            table.add_columns("Time", "Symbol", "Side", "Qty", "Price", "Status", "PnL")

            # Sample orders
            table.add_row("14:32:15", "BTC/USDT", "[green]BUY[/green]", "0.002", "49,850", "[green]FILLED[/green]", "+$15.20")
            table.add_row("14:31:42", "BTC/USDT", "[red]SELL[/red]", "0.001", "49,920", "[yellow]PARTIAL[/yellow]", "+$7.80")
            table.add_row("14:30:05", "ETH/USDT", "[green]BUY[/green]", "0.05", "3,250", "[green]FILLED[/green]", "+$2.30")

            yield table

        return container

    def create_chart_panel(self) -> Container:
        """Criar painel de gráfico (placeholder para implementação futura)"""
        container = Container(id="chart-panel")

        with container:
            yield Static("📊 PRICE CHART", classes="panel-title")
            yield Static("""
[dim]
              Price Action - BTC/USDT (1m)
  50,200 ┤                                   ╭─╮
  50,150 ┤                                 ╭─╯ │
  50,100 ┤                               ╭─╯   │
  50,050 ┤                             ╭─╯     │
  50,000 ┤                           ╭─╯       │
  49,950 ┤                         ╭─╯         │
  49,900 ┤                       ╭─╯           │
  49,850 ┤                     ╭─╯             ╰─╮
  49,800 ┤───────────────────╭─╯                 ╰──
         └─┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─
          09:30 09:45 10:00 10:15 10:30 10:45 11:00 11:15

[/dim]
[green]Volume: 1,234,567 BTC[/green]  [yellow]RSI: 62.3[/yellow]  [blue]MACD: +0.15[/blue]
            """.strip(), classes="chart-container")

        return container

    def create_log_panel(self) -> Container:
        """Criar painel de logs do sistema"""
        container = Container(id="log-panel")

        with container:
            yield Static("📝 SYSTEM LOG", classes="panel-title")

            log = Log(id="system-log")
            log.write_line("[dim]14:32:20[/dim] [green]INFO[/green] Sistema iniciado com sucesso")
            log.write_line("[dim]14:32:21[/dim] [blue]DATA[/blue] Conectado à Binance (latency: 15ms)")
            log.write_line("[dim]14:32:22[/dim] [blue]DATA[/blue] Conectado à Bybit (latency: 23ms)")
            log.write_line("[dim]14:32:25[/dim] [yellow]SIGNAL[/yellow] MOMO-1.5L: +0.3247 (BUY signal)")
            log.write_line("[dim]14:32:30[/dim] [green]ORDER[/green] BUY 0.002 BTC/USDT @ 49,850 - FILLED")
            log.write_line("[dim]14:32:35[/dim] [cyan]PERFORMANCE[/cyan] PnL: +$15.20 (0.15%)")

            yield log

        return container


class AnalyticsScreen(Screen):
    """Screen de analytics com métricas detalhadas"""

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="performance"):
            with TabPane("Performance", id="performance"):
                yield self.create_performance_analytics()

            with TabPane("Indicators", id="indicators"):
                yield self.create_indicators_analytics()

            with TabPane("Risk", id="risk"):
                yield self.create_risk_analytics()

            with TabPane("Backtest", id="backtest"):
                yield self.create_backtest_analytics()

    def create_performance_analytics(self) -> Container:
        """Analytics de performance detalhadas"""
        container = Container()

        with container:
            yield Static("📊 PERFORMANCE ANALYTICS")

            # Performance metrics table
            table = DataTable()
            table.add_columns("Metric", "Value", "Benchmark", "Status")

            table.add_row("Daily Return", "+0.92%", "0.65%", "[green]✅[/green]")
            table.add_row("Sharpe Ratio", "2.34", "1.80", "[green]✅[/green]")
            table.add_row("Max Drawdown", "-1.2%", "-3.8%", "[green]✅[/green]")
            table.add_row("Win Rate", "67.5%", "60.0%", "[green]✅[/green]")
            table.add_row("Profit Factor", "1.85", "1.50", "[green]✅[/green]")
            table.add_row("Trades/Day", "127", "100", "[green]✅[/green]")

            yield table

            # Performance chart (ASCII)
            yield Static("""
Performance Curve (Last 30 days):
   2.5% ┤                                               ╭─
   2.0% ┤                                             ╭─╯
   1.5% ┤                                           ╭─╯
   1.0% ┤                                         ╭─╯
   0.5% ┤                                   ╭───╭─╯
   0.0% ┤─────────────────────────────────╭─╯
  -0.5% ┤
        └┬────┬────┬────┬────┬────┬────┬────┬────┬────┬
         1    5    10   15   20   25   30   35   40   45
            """, classes="chart-container")

        return container

    def create_indicators_analytics(self) -> Container:
        """Analytics de indicadores"""
        container = Container()

        with container:
            yield Static("🔍 INDICATORS ANALYTICS")

            # Grid of indicator widgets
            with Grid():
                yield MOMOWidget()
                yield AggressionWidget()
                yield RegimeWidget()

            # Correlation matrix
            yield Static("""
Correlation Matrix:
             MOMO  Aggr  RSI  MACD  VRP  Regime
MOMO         1.00  0.72 -0.35  0.89  0.45   0.67
Aggression   0.72  1.00 -0.28  0.65  0.78   0.54
RSI         -0.35 -0.28  1.00 -0.42 -0.33  -0.29
MACD         0.89  0.65 -0.42  1.00  0.38   0.71
VRP          0.45  0.78 -0.33  0.38  1.00   0.43
Regime       0.67  0.54 -0.29  0.71  0.43   1.00
            """, classes="chart-container")

        return container

    def create_risk_analytics(self) -> Container:
        """Analytics de risco"""
        container = Container()

        with container:
            yield Static("⚠️ RISK ANALYTICS")

            # Risk metrics
            risk_table = DataTable()
            risk_table.add_columns("Risk Metric", "Current", "Limit", "Utilization")

            risk_table.add_row("Position Size", "$2,500", "$5,000", "50%")
            risk_table.add_row("Leverage", "15x", "25x", "60%")
            risk_table.add_row("Drawdown", "1.2%", "3.8%", "32%")
            risk_table.add_row("VAR (95%)", "$125", "$380", "33%")
            risk_table.add_row("Expected Shortfall", "$195", "$500", "39%")

            yield risk_table

            # Risk distribution
            yield Static("""
Risk Distribution:
  Low Risk   ████████████████████████████████ 65%
  Med Risk   ████████████████ 32%
  High Risk  ██ 3%

Daily VAR Distribution:
    $0-50   ████████████████████████████ 70%
  $50-100   ████████████████ 20%
 $100-150   ████ 8%
 $150+      ██ 2%
            """, classes="chart-container")

        return container

    def create_backtest_analytics(self) -> Container:
        """Analytics de backtest"""
        container = Container()

        with container:
            yield Static("🔄 BACKTEST ANALYTICS")

            # Backtest results
            bt_table = DataTable()
            bt_table.add_columns("Strategy", "Period", "Return", "Sharpe", "MDD", "Trades")

            bt_table.add_row("1.5L Current", "30d", "+27.6%", "2.34", "-1.2%", "3,810")
            bt_table.add_row("1.6 Aggressive", "30d", "+35.4%", "2.18", "-2.1%", "5,127")
            bt_table.add_row("1.6pp-R Refined", "30d", "+24.8%", "2.67", "-0.8%", "2,943")

            yield bt_table

            yield Static("""
Strategy Comparison (Cumulative Returns):
   40% ┤
   35% ┤                    ╭─── 1.6 Aggressive
   30% ┤                  ╭─╯
   25% ┤                ╭─╯ ╭─── 1.5L Current
   20% ┤              ╭─╯  ╭─╯
   15% ┤            ╭─╯   ╭─╯ ╭─── 1.6pp-R Refined
   10% ┤          ╭─╯    ╭─╯ ╭─╯
    5% ┤        ╭─╯     ╭─╯ ╭─╯
    0% ┤──────╭─╯      ╭─╯ ╭─╯
       └┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬
        0     5    10    15    20    25    30
            """, classes="chart-container")

        return container