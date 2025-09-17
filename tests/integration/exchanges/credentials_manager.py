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
class SystemCredentials:
    """Todas as credenciais do sistema"""
    kraken: Optional[ExchangeCredentials] = None
    bybit: Optional[ExchangeCredentials] = None
    binance: Optional[ExchangeCredentials] = None
    openai_api_key: Optional[str] = None
    trading_mode: str = "demo"
    enable_real_trading: bool = False


class CredentialsManager:
    """Gerenciador seguro de credenciais"""

    def __init__(self, env_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.env_file = env_file or "config/demo_credentials.env"
        self.credentials: Optional[SystemCredentials] = None

    def load_credentials(self) -> SystemCredentials:
        """Carrega credenciais do arquivo .env"""

        if self.credentials:
            return self.credentials

        try:
            # Load environment file
            env_vars = self._load_env_file()

            # Validate demo mode
            if not self._is_demo_mode(env_vars):
                raise ValueError("ERRO: Apenas modo demo/testnet permitido!")

            # Create credentials object
            self.credentials = SystemCredentials(
                kraken=self._load_kraken_credentials(env_vars),
                bybit=self._load_bybit_credentials(env_vars),
                binance=self._load_binance_credentials(env_vars),
                openai_api_key=env_vars.get('OPENAI_API_KEY'),
                trading_mode=env_vars.get('TRADING_MODE', 'demo'),
                enable_real_trading=env_vars.get('ENABLE_REAL_TRADING', 'false').lower() == 'true'
            )

            # Final safety check
            if self.credentials.enable_real_trading:
                raise ValueError("ERRO: Real trading desabilitado para testes!")

            self.logger.info("Credenciais carregadas com sucesso (MODO DEMO)")
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

    def _is_demo_mode(self, env_vars: Dict[str, str]) -> bool:
        """Verifica se está em modo demo/testnet"""

        trading_mode = env_vars.get('TRADING_MODE', '').lower()
        demo_mode = env_vars.get('DEMO_MODE', 'false').lower() == 'true'
        test_only = env_vars.get('TEST_ONLY', 'false').lower() == 'true'
        enable_real = env_vars.get('ENABLE_REAL_TRADING', 'false').lower() == 'true'

        kraken_env = env_vars.get('KRAKEN_ENVIRONMENT', '').lower()
        bybit_env = env_vars.get('BYBIT_ENVIRONMENT', '').lower()
        binance_env = env_vars.get('BINANCE_ENVIRONMENT', '').lower()

        # Must be demo/testnet
        if trading_mode != 'demo':
            self.logger.error(f"Trading mode deve ser 'demo', encontrado: {trading_mode}")
            return False

        # Must not enable real trading
        if enable_real:
            self.logger.error("Real trading está habilitado - não permitido para testes")
            return False

        # Exchanges must be in testnet/sandbox
        if kraken_env not in ['sandbox', 'testnet', '']:
            self.logger.error(f"Kraken deve estar em sandbox/testnet, encontrado: {kraken_env}")
            return False

        if bybit_env not in ['testnet', 'sandbox', '']:
            self.logger.error(f"Bybit deve estar em testnet, encontrado: {bybit_env}")
            return False

        if binance_env not in ['testnet', 'sandbox', '']:
            self.logger.error(f"Binance deve estar em testnet, encontrado: {binance_env}")
            return False

        return True

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

    def validate_credentials(self) -> Dict[str, bool]:
        """Valida se as credenciais estão corretas"""

        if not self.credentials:
            raise ValueError("Credenciais não carregadas")

        validation = {
            'kraken': False,
            'bybit': False,
            'binance': False,
            'openai': False,
            'safety_checks': False
        }

        # Validate Kraken
        if self.credentials.kraken:
            validation['kraken'] = (
                bool(self.credentials.kraken.api_key) and
                bool(self.credentials.kraken.api_secret) and
                self.credentials.kraken.environment in ['sandbox', 'testnet']
            )

        # Validate Bybit
        if self.credentials.bybit:
            validation['bybit'] = (
                bool(self.credentials.bybit.api_key) and
                bool(self.credentials.bybit.api_secret) and
                self.credentials.bybit.environment == 'testnet'
            )

        # Validate Binance
        if self.credentials.binance:
            validation['binance'] = (
                bool(self.credentials.binance.api_key) and
                bool(self.credentials.binance.api_secret) and
                self.credentials.binance.environment in ['testnet', 'sandbox']
            )

        # Validate OpenAI
        validation['openai'] = bool(self.credentials.openai_api_key)

        # Safety checks
        validation['safety_checks'] = (
            self.credentials.trading_mode == 'demo' and
            not self.credentials.enable_real_trading
        )

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