#!/usr/bin/env python3
"""
Credentials Manager - Sistema Seguro de Gerenciamento de Credenciais
Carrega credenciais de forma segura para testes de integração

IMPORTANTE: APENAS PARA TESTES DEMO/TESTNET!

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import os
import sys
import json
from typing import Dict, Optional
from dataclasses import dataclass
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@dataclass
class ExchangeCredentials:
    """Credenciais de uma exchange"""
    api_key: str
    api_secret: str
    environment: str = "testnet"  # testnet | sandbox | live
    account_name: Optional[str] = None


@dataclass
class CoinbaseCredentials:
    """Credenciais Coinbase Advanced Trade"""

    api_key: str
    private_key: Optional[str]
    client_key: Optional[str] = None
    key_id: Optional[str] = None
    base_url: str = "https://api.coinbase.com/api/v3"
    environment: str = "sandbox"
    private_key_path: Optional[str] = None
    key_file: Optional[str] = None


@dataclass
class SystemCredentials:
    """Todas as credenciais do sistema"""

    kraken: Optional[ExchangeCredentials] = None
    bybit: Optional[ExchangeCredentials] = None
    binance: Optional[ExchangeCredentials] = None
    coinbase: Optional[CoinbaseCredentials] = None
    openai_api_key: Optional[str] = None
    trading_mode: str = "demo"
    enable_real_trading: bool = False


class CredentialsManager:
    """Gerenciador seguro de credenciais"""

    def __init__(self, env_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        env_override = env_file or os.getenv("WOWCAPITAL_CREDENTIALS_FILE") or os.getenv("WOWCREDENTIALS_FILE")
        self.env_file = env_override or "config/demo_credentials.env"
        self.credentials: Optional[SystemCredentials] = None

    def load_credentials(self) -> SystemCredentials:
        """Carrega credenciais do arquivo .env"""

        if self.credentials:
            return self.credentials

        try:
            # Load environment file
            env_vars = self._load_env_file()

            # Validate safety mode
            if not self._is_safe_mode(env_vars):
                raise ValueError("ERRO: Modo de operação inseguro para testes de integração.")

            # Create credentials object
            self.credentials = SystemCredentials(
                kraken=self._load_kraken_credentials(env_vars),
                bybit=self._load_bybit_credentials(env_vars),
                binance=self._load_binance_credentials(env_vars),
                coinbase=self._load_coinbase_credentials(env_vars),
                openai_api_key=env_vars.get('OPENAI_API_KEY'),
                trading_mode=env_vars.get('TRADING_MODE', 'demo'),
                enable_real_trading=env_vars.get('ENABLE_REAL_TRADING', 'false').lower() == 'true'
            )

            self.logger.info(f"Credenciais carregadas com sucesso (Modo: {self.credentials.trading_mode.upper()})")
            return self.credentials

        except Exception as e:
            self.logger.error(f"Erro carregando credenciais: {str(e)}")
            raise

    def _load_env_file(self) -> Dict[str, str]:
        """Carrega variáveis do arquivo .env"""

        env_vars = {}
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', self.env_file)

        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Arquivo de credenciais não encontrado: {env_path}")

        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

        return env_vars

    def _is_safe_mode(self, env_vars: Dict[str, str]) -> bool:
        """Verifica se o modo de operação é seguro para testes."""

        trading_mode = env_vars.get('TRADING_MODE', 'demo').lower()
        enable_real = env_vars.get('ENABLE_REAL_TRADING', 'false').lower() == 'true'

        # Real trading is explicitly forbidden in integration tests
        if enable_real:
            self.logger.error("ERRO: ENABLE_REAL_TRADING=true não é permitido em testes de integração.")
            return False

        # Demo mode is always safe
        if trading_mode == 'demo':
            self.logger.info("Modo DEMO verificado.")
            return True

        # Shadow Real mode is also safe for testing purposes
        if trading_mode == 'real' and not enable_real:
            self.logger.info("Modo SHADOW REAL verificado.")
            return True

        # Any other combination is invalid
        self.logger.error(f"Configuração de TRADING_MODE='{trading_mode}' inválida.")
        return False

    def _load_kraken_credentials(self, env_vars: Dict[str, str]) -> Optional[ExchangeCredentials]:
        """Carrega credenciais Kraken"""

        api_key = env_vars.get('KRAKEN_API_KEY')
        private_key = env_vars.get('KRAKEN_PRIVATE_KEY')

        if not api_key or not private_key:
            self.logger.warning("Credenciais Kraken não encontradas")
            return None

        return ExchangeCredentials(
            api_key=api_key,
            api_secret=private_key,
            environment=env_vars.get('KRAKEN_ENVIRONMENT', 'sandbox')
        )

    def _load_bybit_credentials(self, env_vars: Dict[str, str]) -> Optional[ExchangeCredentials]:
        """Carrega credenciais Bybit"""

        api_key = env_vars.get('BYBIT_API_KEY')
        api_secret = env_vars.get('BYBIT_API_SECRET')

        if not api_key or not api_secret:
            self.logger.warning("Credenciais Bybit não encontradas")
            return None

        return ExchangeCredentials(
            api_key=api_key,
            api_secret=api_secret,
            environment=env_vars.get('BYBIT_ENVIRONMENT', 'testnet'),
            account_name=env_vars.get('BYBIT_ACCOUNT_NAME')
        )

    def _load_binance_credentials(self, env_vars: Dict[str, str]) -> Optional[ExchangeCredentials]:
        """Carrega credenciais Binance"""

        api_key = env_vars.get('BINANCE_API_KEY')
        api_secret = env_vars.get('BINANCE_API_SECRET')

        if not api_key or not api_secret:
            self.logger.warning("Credenciais Binance não encontradas")
            return None

        return ExchangeCredentials(
            api_key=api_key,
            api_secret=api_secret,
            environment=env_vars.get('BINANCE_ENVIRONMENT', 'testnet'),
            account_name=env_vars.get('BINANCE_ACCOUNT_NAME')
        )

    def _load_coinbase_credentials(self, env_vars: Dict[str, str]) -> Optional[CoinbaseCredentials]:
        """Carrega credenciais Coinbase Advanced Trade"""

        api_key = env_vars.get('COINBASE_CLIENT_KEY') or env_vars.get('COINBASE_API_KEY') or env_vars.get('COINBASE_KEY_ID')
        private_key_pem = env_vars.get('COINBASE_PRIVATE_KEY_PEM')
        private_key_path = env_vars.get('COINBASE_PRIVATE_KEY_PATH')
        key_file = env_vars.get('COINBASE_KEY_FILE')

        if not api_key:
            self.logger.warning("Credenciais Coinbase não encontradas (API key ausente)")
            return None

        private_key: Optional[str] = None
        if private_key_pem:
            private_key = private_key_pem
        elif private_key_path:
            full_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', private_key_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        private_key = f.read()
                except Exception as exc:
                    self.logger.warning(f"Não foi possível carregar chave privada Coinbase: {exc}")
            else:
                self.logger.warning(f"Arquivo de chave privada Coinbase não encontrado: {full_path}")

        key_id_value = env_vars.get('COINBASE_KEY_ID')
        if not private_key and key_file:
            candidate = os.path.join(os.path.dirname(__file__), '..', '..', '..', key_file)
            if os.path.exists(candidate):
                try:
                    with open(candidate, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    private_key = data.get('privateKey') or data.get('private_key')
                    api_key = api_key or data.get('name') or data.get('apiKey')
                    if not key_id_value:
                        key_id_value = data.get('keyId')
                    if not key_id_value and api_key and '/' in api_key:
                        key_id_value = api_key.split('/')[-1]
                except Exception as exc:
                    self.logger.warning(f"Erro lendo arquivo JSON de credenciais Coinbase: {exc}")
            else:
                self.logger.warning(f"Arquivo JSON de credenciais Coinbase não encontrado: {candidate}")

        base_url = env_vars.get('COINBASE_BASE_URL', 'https://api.coinbase.com/api/v3')

        return CoinbaseCredentials(
            api_key=api_key,
            private_key=private_key,
            client_key=env_vars.get('COINBASE_CLIENT_KEY'),
            key_id=key_id_value,
            base_url=base_url,
            environment=env_vars.get('COINBASE_ENVIRONMENT', 'sandbox'),
            private_key_path=private_key_path,
            key_file=key_file
        )

    def validate_credentials(self) -> Dict[str, bool]:
        """Valida se as credenciais estão corretas, permitindo ambientes 'live' em modo shadow."""

        if not self.credentials:
            raise ValueError("Credenciais não carregadas")

        validation = {
            'kraken': False,
            'bybit': False,
            'binance': False,
            'coinbase': False,
            'openai': False,
            'safety_checks': False
        }

        is_shadow_mode = self.credentials.trading_mode == 'real' and not self.credentials.enable_real_trading
        is_demo_mode = self.credentials.trading_mode == 'demo'

        # Validate Kraken
        if self.credentials.kraken:
            is_demo_env = self.credentials.kraken.environment in ['sandbox', 'testnet']
            is_live_env = self.credentials.kraken.environment not in ['sandbox', 'testnet']
            validation['kraken'] = (
                bool(self.credentials.kraken.api_key) and
                bool(self.credentials.kraken.api_secret) and
                (is_demo_env or (is_live_env and is_shadow_mode))
            )

        # Validate Bybit
        if self.credentials.bybit:
            is_demo_env = self.credentials.bybit.environment in ['testnet', 'demo']
            is_live_env = self.credentials.bybit.environment not in ['testnet', 'demo']
            validation['bybit'] = (
                bool(self.credentials.bybit.api_key) and
                bool(self.credentials.bybit.api_secret) and
                (is_demo_env or (is_live_env and is_shadow_mode))
            )

        # Validate Binance
        if self.credentials.binance:
            is_demo_env = self.credentials.binance.environment in ['testnet', 'sandbox']
            is_live_env = self.credentials.binance.environment not in ['testnet', 'sandbox']
            validation['binance'] = (
                bool(self.credentials.binance.api_key) and
                bool(self.credentials.binance.api_secret) and
                (is_demo_env or (is_live_env and is_shadow_mode))
            )

        # Validate Coinbase
        if self.credentials.coinbase:
            has_key_material = bool(self.credentials.coinbase.private_key)
            is_demo_env = self.credentials.coinbase.environment in ['sandbox', 'testnet', 'demo']
            is_live_env = self.credentials.coinbase.environment not in ['sandbox', 'testnet', 'demo']
            validation['coinbase'] = (
                bool(self.credentials.coinbase.api_key) and has_key_material and
                (is_demo_env or (is_live_env and is_shadow_mode))
            )

        # Validate OpenAI
        validation['openai'] = bool(self.credentials.openai_api_key)

        # Safety checks
        validation['safety_checks'] = is_demo_mode or is_shadow_mode

        return validation

    def get_credentials_summary(self) -> Dict[str, any]:
        """Retorna resumo das credenciais (sem expor secrets)"""

        if not self.credentials:
            return {'status': 'not_loaded'}

        return {
            'trading_mode': self.credentials.trading_mode,
            'enable_real_trading': self.credentials.enable_real_trading,
            'exchanges': {
                'kraken': {
                    'available': self.credentials.kraken is not None,
                    'environment': self.credentials.kraken.environment if self.credentials.kraken else None,
                    'api_key_prefix': self.credentials.kraken.api_key[:8] + '...' if self.credentials.kraken else None
                },
                'bybit': {
                    'available': self.credentials.bybit is not None,
                    'environment': self.credentials.bybit.environment if self.credentials.bybit else None,
                    'account_name': self.credentials.bybit.account_name if self.credentials.bybit else None,
                    'api_key_prefix': self.credentials.bybit.api_key[:8] + '...' if self.credentials.bybit else None
                },
                'binance': {
                    'available': self.credentials.binance is not None,
                    'environment': self.credentials.binance.environment if self.credentials.binance else None,
                    'account_name': self.credentials.binance.account_name if self.credentials.binance else None,
                    'api_key_prefix': self.credentials.binance.api_key[:8] + '...' if self.credentials.binance else None
                },
                'coinbase': {
                    'available': self.credentials.coinbase is not None,
                    'environment': self.credentials.coinbase.environment if self.credentials.coinbase else None,
                    'api_key_prefix': (self.credentials.coinbase.api_key[:8] + '...') if self.credentials.coinbase and self.credentials.coinbase.api_key else None,
                    'base_url': self.credentials.coinbase.base_url if self.credentials.coinbase else None,
                    'key_source': (
                        'inline'
                        if self.credentials.coinbase and self.credentials.coinbase.private_key and not self.credentials.coinbase.private_key_path and not self.credentials.coinbase.key_file
                        else ('file' if self.credentials.coinbase and (self.credentials.coinbase.private_key_path or self.credentials.coinbase.key_file) else None)
                    )
                }
            },
            'openai_available': bool(self.credentials.openai_api_key),
            'safety_mode': True  # Always true for demo
        }


def main():
    """Teste básico do credentials manager"""

    logging.basicConfig(level=logging.INFO)

    print("🔐 WOW Capital - Credentials Manager Test")
    print("=" * 50)

    try:
        # Load credentials
        cred_manager = CredentialsManager()
        credentials = cred_manager.load_credentials()

        print("✅ Credenciais carregadas com sucesso")

        # Validate
        validation = cred_manager.validate_credentials()
        print(f"\n📊 Validação:")
        for service, valid in validation.items():
            status = "✅" if valid else "❌"
            print(f"   {status} {service.title()}: {'Valid' if valid else 'Invalid'}")

        # Summary
        summary = cred_manager.get_credentials_summary()
        print(f"\n📋 Resumo:")
        print(f"   Trading Mode: {summary['trading_mode']}")
        print(f"   Real Trading: {summary['enable_real_trading']}")
        print(f"   Safety Mode: {summary['safety_mode']}")

        print(f"\n🏦 Exchanges:")
        for exchange, info in summary['exchanges'].items():
            if info['available']:
                print(f"   ✅ {exchange.title()}: {info['environment']} ({info.get('api_key_prefix', 'N/A')})")
            else:
                print(f"   ❌ {exchange.title()}: Not configured")

        # Final safety check
        if all(validation.values()):
            print(f"\n🎉 Sistema pronto para testes demo!")
        else:
            print(f"\n⚠️  Algumas credenciais precisam de atenção")

    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
