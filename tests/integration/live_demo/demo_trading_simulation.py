#!/usr/bin/env python3
"""
Demo Trading Simulation - Testes de Operações Reais
Simula operações de trading completas usando dados reais das exchanges

SEGURANÇA: Apenas modo demo/simulação - sem ordens reais!

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import sys
import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tests.integration.exchanges.credentials_manager import CredentialsManager
from tests.integration.exchanges.test_kraken_endpoints import KrakenAPITester
from tests.integration.exchanges.test_bybit_endpoints import BybitAPITester


class DemoTradingSimulator:
    """Simulador de trading demo com dados reais"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Load credentials
        self.cred_manager = CredentialsManager()
        self.credentials = self.cred_manager.load_credentials()

        # Initialize exchange testers
        self.kraken_tester = None
        self.bybit_tester = None

        if self.credentials.kraken:
            self.kraken_tester = KrakenAPITester(self.cred_manager)

        if self.credentials.bybit:
            self.bybit_tester = BybitAPITester(self.cred_manager)

        # Demo trading state
        self.demo_portfolio = {
            'equity': 10000.0,  # $10k demo
            'positions': {},
            'orders': [],
            'pnl_today': 0.0,
            'trades_count': 0,
            'start_time': datetime.now()
        }

        # Market data cache
        self.market_data = {}
        self.price_history = {}

        self.logger.info("Demo Trading Simulator initialized")

    def fetch_market_data(self) -> Dict[str, Any]:
        """Coleta dados de mercado das exchanges"""

        market_data = {
            'timestamp': datetime.now(),
            'exchanges': {},
            'symbols': ['BTCUSD', 'BTCUSDT'],
            'success': False
        }

        try:
            # Get Kraken data
            if self.kraken_tester:
                kraken_btc = self.kraken_tester.test_ticker_data("BTCUSD")
                if kraken_btc.get('success'):
                    market_data['exchanges']['kraken'] = {
                        'symbol': 'BTCUSD',
                        'price': float(kraken_btc.get('last_price', 0)),
                        'bid': float(kraken_btc.get('bid', 0)),
                        'ask': float(kraken_btc.get('ask', 0)),
                        'volume_24h': float(kraken_btc.get('volume_24h', 0)),
                        'high_24h': float(kraken_btc.get('high_24h', 0)),
                        'low_24h': float(kraken_btc.get('low_24h', 0))
                    }

            # Get Bybit data
            if self.bybit_tester:
                bybit_btc = self.bybit_tester.test_ticker_data("BTCUSDT")
                if bybit_btc.get('success'):
                    market_data['exchanges']['bybit'] = {
                        'symbol': 'BTCUSDT',
                        'price': float(bybit_btc.get('last_price', 0)),
                        'bid': float(bybit_btc.get('bid1_price', 0)),
                        'ask': float(bybit_btc.get('ask1_price', 0)),
                        'volume_24h': float(bybit_btc.get('volume24h', 0)),
                        'high_24h': float(bybit_btc.get('high_price24h', 0)),
                        'low_24h': float(bybit_btc.get('low_price24h', 0)),
                        'change_24h': float(bybit_btc.get('price_change24h', 0))
                    }

            market_data['success'] = len(market_data['exchanges']) > 0

            # Update price history
            if market_data['success']:
                timestamp = time.time()
                for exchange, data in market_data['exchanges'].items():
                    symbol_key = f"{exchange}_{data['symbol']}"
                    if symbol_key not in self.price_history:
                        self.price_history[symbol_key] = []

                    self.price_history[symbol_key].append({
                        'timestamp': timestamp,
                        'price': data['price'],
                        'bid': data['bid'],
                        'ask': data['ask']
                    })

                    # Keep only last 100 points
                    if len(self.price_history[symbol_key]) > 100:
                        self.price_history[symbol_key] = self.price_history[symbol_key][-100:]

            self.market_data = market_data
            return market_data

        except Exception as e:
            self.logger.error(f"Erro coletando dados de mercado: {str(e)}")
            market_data['error'] = str(e)
            return market_data

    def calculate_demo_signals(self) -> Dict[str, Any]:
        """Calcula sinais de trading baseado nos dados reais"""

        if not self.market_data.get('success'):
            return {'signals': [], 'error': 'No market data available'}

        signals = []

        try:
            # Simple momentum strategy for demo
            for exchange, data in self.market_data['exchanges'].items():
                symbol_key = f"{exchange}_{data['symbol']}"
                price_hist = self.price_history.get(symbol_key, [])

                if len(price_hist) >= 5:
                    # Calculate simple momentum
                    recent_prices = [p['price'] for p in price_hist[-5:]]
                    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

                    # Generate signal
                    if abs(price_change) > 0.001:  # 0.1% threshold
                        signal = {
                            'exchange': exchange,
                            'symbol': data['symbol'],
                            'side': 'BUY' if price_change > 0 else 'SELL',
                            'strength': min(1.0, abs(price_change) * 100),
                            'price': data['price'],
                            'price_change': price_change,
                            'timestamp': datetime.now(),
                            'confidence': min(0.9, abs(price_change) * 50)
                        }
                        signals.append(signal)

            return {
                'signals': signals,
                'market_data_exchanges': len(self.market_data['exchanges']),
                'total_price_points': sum(len(hist) for hist in self.price_history.values())
            }

        except Exception as e:
            self.logger.error(f"Erro calculando sinais: {str(e)}")
            return {'signals': [], 'error': str(e)}

    def simulate_demo_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Simula execução de uma trade (sem ordem real!)"""

        try:
            # Calculate position size (2% of equity)
            position_value = self.demo_portfolio['equity'] * 0.02
            quantity = position_value / signal['price']

            # Simulate trade execution
            trade = {
                'trade_id': f"demo_{int(time.time())}",
                'exchange': signal['exchange'],
                'symbol': signal['symbol'],
                'side': signal['side'],
                'quantity': quantity,
                'price': signal['price'],
                'value': position_value,
                'timestamp': datetime.now(),
                'status': 'FILLED',  # Always filled in demo
                'commission': position_value * 0.001,  # 0.1% commission
                'signal_confidence': signal['confidence']
            }

            # Update demo portfolio
            position_key = f"{signal['exchange']}_{signal['symbol']}"

            if position_key not in self.demo_portfolio['positions']:
                self.demo_portfolio['positions'][position_key] = {
                    'quantity': 0.0,
                    'avg_price': 0.0,
                    'total_value': 0.0,
                    'unrealized_pnl': 0.0
                }

            position = self.demo_portfolio['positions'][position_key]

            if signal['side'] == 'BUY':
                # Add to position
                new_quantity = position['quantity'] + quantity
                new_total_value = position['total_value'] + position_value

                if new_quantity > 0:
                    position['avg_price'] = new_total_value / new_quantity
                    position['quantity'] = new_quantity
                    position['total_value'] = new_total_value
            else:
                # Reduce position (or go short in demo)
                position['quantity'] -= quantity
                if position['quantity'] < 0:
                    position['avg_price'] = signal['price']  # Update for short position

            # Update portfolio stats
            self.demo_portfolio['trades_count'] += 1
            self.demo_portfolio['equity'] -= trade['commission']

            # Store trade
            self.demo_portfolio['orders'].append(trade)

            self.logger.info(f"Demo trade executed: {signal['side']} {quantity:.6f} {signal['symbol']} @ ${signal['price']}")

            return {
                'success': True,
                'trade': trade,
                'portfolio_equity': self.demo_portfolio['equity'],
                'total_trades': self.demo_portfolio['trades_count']
            }

        except Exception as e:
            self.logger.error(f"Erro simulando trade: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def update_portfolio_pnl(self) -> Dict[str, Any]:
        """Atualiza PnL do portfolio baseado em preços atuais"""

        try:
            total_unrealized_pnl = 0.0

            for position_key, position in self.demo_portfolio['positions'].items():
                if position['quantity'] == 0:
                    continue

                # Get current market price
                exchange, symbol = position_key.split('_', 1)
                current_price = None

                if exchange in self.market_data.get('exchanges', {}):
                    current_price = self.market_data['exchanges'][exchange]['price']

                if current_price:
                    # Calculate unrealized PnL
                    if position['quantity'] > 0:  # Long position
                        pnl = (current_price - position['avg_price']) * position['quantity']
                    else:  # Short position
                        pnl = (position['avg_price'] - current_price) * abs(position['quantity'])

                    position['unrealized_pnl'] = pnl
                    total_unrealized_pnl += pnl

            # Update portfolio
            self.demo_portfolio['pnl_today'] = total_unrealized_pnl
            total_equity = self.demo_portfolio['equity'] + total_unrealized_pnl

            return {
                'success': True,
                'equity_base': self.demo_portfolio['equity'],
                'unrealized_pnl': total_unrealized_pnl,
                'total_equity': total_equity,
                'return_pct': (total_equity - 10000) / 10000 * 100,
                'positions_count': len([p for p in self.demo_portfolio['positions'].values() if p['quantity'] != 0])
            }

        except Exception as e:
            self.logger.error(f"Erro atualizando PnL: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_demo_trading_session(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Executa uma sessão demo de trading"""

        self.logger.info(f"Iniciando sessão demo de {duration_minutes} minutos...")

        session_results = {
            'start_time': datetime.now(),
            'duration_minutes': duration_minutes,
            'market_data_points': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'final_portfolio': {},
            'performance': {},
            'errors': []
        }

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        try:
            iteration = 0
            while datetime.now() < end_time:
                iteration += 1
                self.logger.info(f"Demo iteration {iteration}...")

                # 1. Fetch market data
                market_data = self.fetch_market_data()
                if market_data.get('success'):
                    session_results['market_data_points'] += 1

                    # 2. Calculate signals
                    signals_result = self.calculate_demo_signals()
                    signals = signals_result.get('signals', [])
                    session_results['signals_generated'] += len(signals)

                    # 3. Execute demo trades (only strong signals)
                    for signal in signals:
                        if signal['confidence'] > 0.7:  # Only high confidence signals
                            trade_result = self.simulate_demo_trade(signal)
                            if trade_result.get('success'):
                                session_results['trades_executed'] += 1

                    # 4. Update portfolio PnL
                    pnl_result = self.update_portfolio_pnl()

                    if iteration % 3 == 0:  # Log every 3rd iteration
                        if pnl_result.get('success'):
                            self.logger.info(f"Portfolio: ${pnl_result['total_equity']:.2f} "
                                           f"({pnl_result['return_pct']:+.2f}%), "
                                           f"Trades: {self.demo_portfolio['trades_count']}")
                else:
                    session_results['errors'].append(f"Iteration {iteration}: Market data failed")

                # Wait between iterations
                time.sleep(10)  # 10 seconds between updates

            # Final portfolio status
            final_pnl = self.update_portfolio_pnl()
            session_results['final_portfolio'] = self.demo_portfolio.copy()
            session_results['performance'] = final_pnl

            # Calculate session performance
            session_results['end_time'] = datetime.now()
            session_results['actual_duration'] = (session_results['end_time'] - session_results['start_time']).total_seconds() / 60

            return session_results

        except Exception as e:
            self.logger.error(f"Erro na sessão demo: {str(e)}")
            session_results['errors'].append(f"Session error: {str(e)}")
            return session_results

    def get_session_summary(self, session_results: Dict[str, Any]) -> str:
        """Gera resumo da sessão de trading"""

        summary = f"""
🎯 WOW CAPITAL - DEMO TRADING SESSION SUMMARY
{'='*60}

📊 Session Details:
   Duration: {session_results['actual_duration']:.1f} minutes
   Market Data Points: {session_results['market_data_points']}
   Signals Generated: {session_results['signals_generated']}
   Trades Executed: {session_results['trades_executed']}

💰 Portfolio Performance:
   Start Equity: $10,000.00
   Final Equity: ${session_results['performance'].get('total_equity', 0):.2f}
   Return: {session_results['performance'].get('return_pct', 0):+.2f}%
   Unrealized PnL: ${session_results['performance'].get('unrealized_pnl', 0):+.2f}
   Active Positions: {session_results['performance'].get('positions_count', 0)}

📈 Trading Activity:
   Total Trades: {session_results['final_portfolio']['trades_count']}
   Success Rate: {(session_results['trades_executed'] / max(1, session_results['signals_generated'])) * 100:.1f}%

🔄 Market Data:
   Exchanges Connected: {len(self.market_data.get('exchanges', {}))}
   Price History Points: {sum(len(hist) for hist in self.price_history.values())}

{"⚠️  Errors: " + str(len(session_results['errors'])) if session_results['errors'] else "✅ No Errors"}

{'='*60}
        """.strip()

        return summary


def main():
    """Executa demonstração completa de trading"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🚀 WOW Capital - Demo Trading Simulation")
    print("=" * 60)
    print("IMPORTANTE: Esta é uma simulação com dados reais mas SEM ORDENS REAIS!")
    print("=" * 60)

    try:
        # Create simulator
        simulator = DemoTradingSimulator()

        # Test market data fetch first
        print("\n📊 Testando conectividade com exchanges...")
        market_data = simulator.fetch_market_data()

        if market_data.get('success'):
            print("✅ Dados de mercado coletados com sucesso!")
            for exchange, data in market_data['exchanges'].items():
                print(f"   {exchange.title()}: {data['symbol']} = ${data['price']:,.2f}")
        else:
            print("❌ Erro coletando dados de mercado")
            return False

        # Run demo trading session
        print(f"\n🎯 Iniciando sessão de trading demo...")
        session_results = simulator.run_demo_trading_session(duration_minutes=0.5)  # 30 seconds for quick demo

        # Display results
        summary = simulator.get_session_summary(session_results)
        print(summary)

        # Success criteria
        success = (
            session_results['market_data_points'] > 0 and
            session_results.get('performance', {}).get('success', False) and
            len(session_results['errors']) == 0
        )

        return success

    except Exception as e:
        print(f"❌ Erro geral: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Demo Trading {'SUCCESS' if success else 'NEEDS ATTENTION'}")
    sys.exit(0 if success else 1)