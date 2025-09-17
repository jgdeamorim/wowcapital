#!/usr/bin/env python3
"""
Componente de Widgets para Indicadores
Widgets especializados para exibição de indicadores técnicos

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from typing import Dict, Any, Optional, List


class MOMOWidget(Static):
    """Widget especializado para MOMO-1.5L"""

    score = reactive(0.0)
    trend = reactive("NEUTRAL")
    confidence = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "MOMO-1.5L"

    def watch_score(self, score: float) -> None:
        self.update_display()

    def watch_trend(self, trend: str) -> None:
        self.update_display()

    def watch_confidence(self, confidence: float) -> None:
        self.update_display()

    def update_display(self):
        # Color coding based on score
        if self.score > 0.6:
            color = "bright_green"
            signal = "🚀 STRONG BUY"
        elif self.score > 0.2:
            color = "green"
            signal = "📈 BUY"
        elif self.score < -0.6:
            color = "bright_red"
            signal = "🔻 STRONG SELL"
        elif self.score < -0.2:
            color = "red"
            signal = "📉 SELL"
        else:
            color = "yellow"
            signal = "⏸️  HOLD"

        # Create progress bar
        normalized_score = (self.score + 1.5) / 3.0  # Normalize -1.5 to 1.5 -> 0 to 1
        bar_char = "█" * int(normalized_score * 20)

        content = f"""
Score: [{color}]{self.score:+.4f}[/{color}]
Signal: {signal}
Trend: {self.trend}
Confidence: {self.confidence:.1%}

[{color}]{bar_char}[/{color}]
        """.strip()

        self.update(content)


class AggressionWidget(Static):
    """Widget especializado para High Aggression Score"""

    score = reactive(0.0)
    components = reactive({})
    explosion_ready = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Aggression Score"

    def watch_score(self, score: float) -> None:
        self.explosion_ready = score >= 0.92
        self.update_display()

    def watch_components(self, components: Dict[str, float]) -> None:
        self.update_display()

    def update_display(self):
        # Color coding for explosion readiness
        if self.explosion_ready:
            color = "bright_red"
            status = "🔴 EXPLOSION READY!"
            border_color = "red"
        elif self.score >= 0.75:
            color = "bright_yellow"
            status = "🟡 High Alert"
            border_color = "yellow"
        else:
            color = "blue"
            status = "🟢 Normal"
            border_color = "blue"

        # Components breakdown
        components_text = ""
        if self.components:
            for name, value in self.components.items():
                bar_width = int(value * 10)
                bar = "█" * bar_width + "░" * (10 - bar_width)
                components_text += f"{name:>12}: [{color}]{bar}[/{color}] {value:.3f}\n"

        content = f"""
[{color} bold]Score: {self.score:.4f}[/{color} bold]
Status: {status}

{components_text.rstrip()}
        """.strip()

        self.styles.border = ("solid", border_color)
        self.update(content)


class RegimeWidget(Static):
    """Widget para exibir regime de mercado"""

    regime = reactive("UNKNOWN")
    confidence = reactive(0.0)
    regime_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Market Regime"

    def watch_regime(self, regime: str) -> None:
        self.update_display()

    def watch_confidence(self, confidence: float) -> None:
        self.update_display()

    def watch_regime_data(self, regime_data: Dict[str, float]) -> None:
        self.update_display()

    def update_display(self):
        # Color coding by regime
        regime_colors = {
            "TRENDING_UP": "bright_green",
            "TRENDING_DOWN": "bright_red",
            "RANGING": "yellow",
            "VOLATILE": "magenta",
            "BREAKOUT": "cyan",
            "UNKNOWN": "white"
        }

        regime_icons = {
            "TRENDING_UP": "📈",
            "TRENDING_DOWN": "📉",
            "RANGING": "↔️",
            "VOLATILE": "⚡",
            "BREAKOUT": "💥",
            "UNKNOWN": "❓"
        }

        color = regime_colors.get(self.regime, "white")
        icon = regime_icons.get(self.regime, "❓")

        # Regime probabilities
        probs_text = ""
        if self.regime_data:
            for regime, prob in sorted(self.regime_data.items(), key=lambda x: x[1], reverse=True):
                bar_width = int(prob * 15)
                bar = "█" * bar_width + "░" * (15 - bar_width)
                prob_color = regime_colors.get(regime, "white")
                probs_text += f"[{prob_color}]{bar}[/{prob_color}] {regime:<12} {prob:.1%}\n"

        content = f"""
[{color} bold]{icon} {self.regime}[/{color} bold]
Confidence: {self.confidence:.1%}

{probs_text.rstrip()}
        """.strip()

        self.update(content)


class TechnicalSummaryWidget(Static):
    """Widget resumo de todos os indicadores técnicos"""

    indicators = reactive({})
    overall_signal = reactive("NEUTRAL")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Technical Summary"

    def watch_indicators(self, indicators: Dict[str, Any]) -> None:
        self.calculate_overall_signal()
        self.update_display()

    def calculate_overall_signal(self):
        """Calcula sinal geral baseado em todos os indicadores"""
        if not self.indicators:
            self.overall_signal = "NEUTRAL"
            return

        signals = []

        # MOMO signal
        momo = self.indicators.get('momo', 0)
        if momo > 0.3:
            signals.append(1)
        elif momo < -0.3:
            signals.append(-1)

        # Aggression signal
        aggr = self.indicators.get('aggression', 0)
        if aggr >= 0.92:
            signals.append(2)  # Strong weight for explosion
        elif aggr >= 0.75:
            signals.append(1)

        # RSI signal
        rsi = self.indicators.get('rsi', 50)
        if rsi > 70:
            signals.append(-1)
        elif rsi < 30:
            signals.append(1)

        # MACD signal
        macd = self.indicators.get('macd_signal', 0)
        if macd > 0:
            signals.append(1)
        elif macd < 0:
            signals.append(-1)

        # Calculate overall
        total_signal = sum(signals)
        if total_signal >= 2:
            self.overall_signal = "STRONG_BUY"
        elif total_signal >= 1:
            self.overall_signal = "BUY"
        elif total_signal <= -2:
            self.overall_signal = "STRONG_SELL"
        elif total_signal <= -1:
            self.overall_signal = "SELL"
        else:
            self.overall_signal = "NEUTRAL"

    def update_display(self):
        # Overall signal color
        signal_colors = {
            "STRONG_BUY": "bright_green",
            "BUY": "green",
            "NEUTRAL": "yellow",
            "SELL": "red",
            "STRONG_SELL": "bright_red"
        }

        signal_icons = {
            "STRONG_BUY": "🚀",
            "BUY": "📈",
            "NEUTRAL": "⏸️",
            "SELL": "📉",
            "STRONG_SELL": "🔻"
        }

        color = signal_colors.get(self.overall_signal, "yellow")
        icon = signal_icons.get(self.overall_signal, "⏸️")

        # Individual indicators
        indicators_text = ""
        if self.indicators:
            for name, value in self.indicators.items():
                if isinstance(value, float):
                    indicators_text += f"{name:>10}: {value:>8.4f}\n"
                else:
                    indicators_text += f"{name:>10}: {str(value):>8}\n"

        content = f"""
[{color} bold]{icon} {self.overall_signal}[/{color} bold]

{indicators_text.rstrip()}
        """.strip()

        self.update(content)


class PerformanceWidget(Static):
    """Widget para métricas de performance"""

    pnl_day = reactive(0.0)
    pnl_total = reactive(0.0)
    win_rate = reactive(0.0)
    sharpe = reactive(0.0)
    max_drawdown = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Performance"

    def watch_pnl_day(self, pnl_day: float) -> None:
        self.update_display()

    def watch_pnl_total(self, pnl_total: float) -> None:
        self.update_display()

    def update_display(self):
        pnl_day_color = "green" if self.pnl_day >= 0 else "red"
        pnl_total_color = "green" if self.pnl_total >= 0 else "red"
        pnl_day_symbol = "+" if self.pnl_day >= 0 else ""
        pnl_total_symbol = "+" if self.pnl_total >= 0 else ""

        content = f"""
[{pnl_day_color}]PnL Today: {pnl_day_symbol}${self.pnl_day:,.2f}[/{pnl_day_color}]
[{pnl_total_color}]PnL Total: {pnl_total_symbol}${self.pnl_total:,.2f}[/{pnl_total_color}]
Win Rate: {self.win_rate:.1%}
Sharpe: {self.sharpe:.2f}
Max DD: {self.max_drawdown:.2%}
        """.strip()

        self.update(content)