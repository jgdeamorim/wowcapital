#!/usr/bin/env python3
"""
WOW Capital - Launcher da Interface TUI
Script principal para iniciar a interface operacional

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import os
import sys
import argparse
import signal
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def check_dependencies():
    """Verifica se todas as dependências estão disponíveis"""
    try:
        import textual
        import rich
        import numpy
        import pandas
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("source tui_env/bin/activate")
        print("pip install textual rich numpy pandas colorama click")
        return False

def launch_basic_tui():
    """Lança a TUI básica"""
    print("🚀 Launching Basic TUI...")
    from tui.main import MainTUI

    app = MainTUI()
    app.run()

def launch_advanced_tui():
    """Lança a TUI avançada com múltiplas telas"""
    print("🚀 Launching Advanced TUI...")

    from textual.app import App
    from tui.layouts.advanced import TradingScreen, AnalyticsScreen
    from textual.binding import Binding

    class AdvancedTradingApp(App):
        TITLE = "WOW Capital - Advanced Trading Interface"
        SUB_TITLE = "High-Performance Trading System"

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("1", "show_trading", "Trading"),
            Binding("2", "show_analytics", "Analytics"),
            Binding("r", "refresh", "Refresh"),
            Binding("h", "help", "Help"),
        ]

        SCREENS = {
            "trading": TradingScreen,
            "analytics": AnalyticsScreen,
        }

        def on_mount(self) -> None:
            self.push_screen("trading")

        def action_show_trading(self) -> None:
            self.switch_screen("trading")

        def action_show_analytics(self) -> None:
            self.switch_screen("analytics")

        def action_refresh(self) -> None:
            self.notify("🔄 Refreshing data...")

        def action_help(self) -> None:
            help_text = """
🎯 WOW Capital TUI - Keyboard Shortcuts:

Navigation:
  1 - Trading Screen
  2 - Analytics Screen
  q - Quit Application
  r - Refresh Data

Trading (F-Keys):
  F1 - Toggle Auto-Trade
  F2 - Emergency Stop
  F3 - Cancel All Orders
  F5 - Refresh Markets

Special:
  Space - Emergency Stop (Global)
  Esc - Cancel Current Operation
  h - Show this Help
            """.strip()
            self.notify(help_text, timeout=10)

    app = AdvancedTradingApp()
    app.run()

def launch_live_trading_tui():
    """Lança a TUI de live trading"""
    print("🚀 Launching Live Trading TUI...")

    from textual.app import App
    from tui.panels.live_trading import LiveTradingContainer
    from textual.widgets import Header, Footer
    from textual.binding import Binding

    class LiveTradingApp(App):
        TITLE = "WOW Capital - Live Trading"
        SUB_TITLE = "Real-Time Trading Interface"

        CSS = """
        Screen {
            background: #001122;
        }

        LiveTradingContainer {
            height: 100%;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit", priority=True),
            Binding("f1", "toggle_auto_trade", "Auto-Trade"),
            Binding("f2", "emergency_stop", "Emergency Stop", priority=True),
            Binding("f3", "cancel_all", "Cancel All"),
            Binding("f5", "refresh_data", "Refresh"),
            Binding("space", "emergency_stop", "EMERGENCY", priority=True),
        ]

        def compose(self):
            yield Header(show_clock=True)
            yield LiveTradingContainer()
            yield Footer()

        def action_toggle_auto_trade(self) -> None:
            container = self.query_one(LiveTradingContainer)
            container.action_toggle_auto_trade()

        def action_emergency_stop(self) -> None:
            container = self.query_one(LiveTradingContainer)
            container.action_emergency_stop()

        def action_cancel_all(self) -> None:
            container = self.query_one(LiveTradingContainer)
            container.action_cancel_all()

        def action_refresh_data(self) -> None:
            container = self.query_one(LiveTradingContainer)
            container.action_refresh_data()

    app = LiveTradingApp()
    app.run()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}. Shutting down TUI...")
    sys.exit(0)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="WOW Capital TUI - Trading Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tui.py                    # Basic TUI
  python run_tui.py --advanced         # Advanced multi-screen TUI
  python run_tui.py --live             # Live trading TUI
  python run_tui.py --check            # Check dependencies only
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["basic", "advanced", "live"],
        default="basic",
        help="TUI mode to launch (default: basic)"
    )

    parser.add_argument(
        "--advanced", "-a",
        action="store_const",
        const="advanced",
        dest="mode",
        help="Launch advanced multi-screen interface"
    )

    parser.add_argument(
        "--live", "-l",
        action="store_const",
        const="live",
        dest="mode",
        help="Launch live trading interface"
    )

    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check dependencies and exit"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🎯 WOW Capital - TUI Launcher")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        if args.check:
            sys.exit(1)
        print("\n⚠️  Continuing with missing dependencies (may cause errors)")

    if args.check:
        print("✅ Dependency check complete")
        sys.exit(0)

    # Set debug mode
    if args.debug:
        os.environ["TEXTUAL_DEBUG"] = "1"
        print("🐛 Debug mode enabled")

    print(f"🚀 Starting TUI in {args.mode} mode...")
    print("💡 Press 'q' to quit, 'h' for help")
    print("⚠️  Press 'Space' for emergency stop")
    print()

    try:
        if args.mode == "basic":
            launch_basic_tui()
        elif args.mode == "advanced":
            launch_advanced_tui()
        elif args.mode == "live":
            launch_live_trading_tui()

    except KeyboardInterrupt:
        print("\n👋 TUI closed by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check dependencies with: python run_tui.py --check")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        print("🏁 TUI launcher finished")

if __name__ == "__main__":
    main()