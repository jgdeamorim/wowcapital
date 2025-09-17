#!/usr/bin/env python3
"""
Sistema de Sifonamento Avançado - WOW Capital
Implementação completa das regras específicas de sifonamento

Regras implementadas:
- Retenção: min($300, 30% equity, max $500)
- Vault A/B segregação automatizada
- Auto-recomposição: <$50 → +$100
- Cool-down 24-48h pós-recomposição

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

from core.config import settings


class VaultType(Enum):
    """Tipos de vault disponíveis"""
    VAULT_A = "vault_a"  # Vault principal operacional
    VAULT_B = "vault_b"  # Vault secundário de segurança
    EMERGENCY = "emergency"  # Vault de emergência


class TransferStatus(Enum):
    """Status de transferência"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SiphoningConfig:
    """Configuração do sistema de sifonamento"""

    # Retenção rules
    min_retention_fixed: float = 300.0  # $300 mínimo
    max_retention_fixed: float = 500.0  # $500 máximo
    retention_equity_pct: float = 0.30  # 30% do equity

    # Auto-recomposição rules
    recomposition_threshold: float = 50.0  # Trigger quando <$50
    recomposition_amount: float = 100.0   # Adicionar $100
    cooldown_hours_min: int = 24         # 24h mínimo cooldown
    cooldown_hours_max: int = 48         # 48h máximo cooldown

    # Vault segregação
    vault_a_target_pct: float = 0.70    # 70% para Vault A
    vault_b_target_pct: float = 0.30    # 30% para Vault B
    rebalance_threshold_pct: float = 0.15  # 15% desvio para rebalancear

    # Safety limits
    max_single_transfer: float = 10000.0  # $10k max single transfer
    min_transfer_amount: float = 10.0     # $10 min transfer
    max_daily_transfers: int = 50         # Max 50 transfers per day

    # Performance tracking
    profitability_lookback_days: int = 30
    risk_adjustment_factor: float = 1.2


@dataclass
class TransferRecord:
    """Registro de transferência"""
    id: str
    timestamp: datetime
    from_location: str
    to_location: str
    amount: float
    currency: str
    status: TransferStatus
    transfer_type: str  # 'siphoning', 'recomposition', 'rebalancing'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VaultBalance:
    """Saldo de um vault"""
    vault_type: VaultType
    balance_usd: float
    last_updated: datetime
    allocation_pct: float = 0.0


class AdvancedSiphoning:
    """
    Sistema de Sifonamento Avançado

    Funcionalidades:
    1. Cálculo automático de retenção baseado em regras
    2. Segregação automática entre Vault A/B
    3. Auto-recomposição com cooldown
    4. Monitoramento e analytics
    """

    def __init__(self, config: Optional[SiphoningConfig] = None):
        self.config = config or SiphoningConfig()
        self.name = "Advanced-Siphoning"
        self.version = "1.0.0"

        # State tracking
        self.current_equity: float = 0.0
        self.vault_balances: Dict[VaultType, VaultBalance] = {}
        self.last_recomposition: Optional[datetime] = None
        self.transfer_history: List[TransferRecord] = []

        # Analytics
        self.profitability_history: List[Dict[str, Any]] = []
        self.siphoning_analytics: Dict[str, Any] = {}

        # Safety controls
        self.daily_transfer_count: int = 0
        self.daily_transfer_amount: float = 0.0
        self.last_daily_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Cooldown tracking
        self.cooldown_active: bool = False
        self.cooldown_end_time: Optional[datetime] = None

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Advanced Siphoning initialized with retention ${self.config.min_retention_fixed}-${self.config.max_retention_fixed}")

    def calculate_retention_amount(self, current_equity: float) -> float:
        """
        Calcular valor de retenção baseado nas regras:
        min($300, 30% equity, max $500)
        """

        # Calcular 30% do equity
        equity_retention = current_equity * self.config.retention_equity_pct

        # Aplicar min/max limits
        retention_amount = max(
            self.config.min_retention_fixed,
            min(equity_retention, self.config.max_retention_fixed)
        )

        self.logger.debug(f"Retention calculation: equity=${current_equity:.2f}, "
                         f"30%=${equity_retention:.2f}, final=${retention_amount:.2f}")

        return retention_amount

    def calculate_siphoning_amount(self, current_equity: float) -> float:
        """Calcular valor disponível para sifonamento"""

        retention = self.calculate_retention_amount(current_equity)
        available_for_siphoning = max(0.0, current_equity - retention)

        return available_for_siphoning

    def check_recomposition_needed(self, current_equity: float) -> bool:
        """
        Verificar se recomposição é necessária
        Trigger: equity < $50
        """

        return current_equity < self.config.recomposition_threshold

    def is_cooldown_active(self) -> bool:
        """Verificar se cooldown está ativo"""

        if not self.cooldown_active or not self.cooldown_end_time:
            return False

        if datetime.now() >= self.cooldown_end_time:
            self.cooldown_active = False
            self.cooldown_end_time = None
            self.logger.info("Cooldown period ended")
            return False

        return True

    def calculate_vault_allocation(self, total_amount: float) -> Dict[VaultType, float]:
        """
        Calcular alocação entre Vault A e B
        70% Vault A, 30% Vault B
        """

        allocation = {
            VaultType.VAULT_A: total_amount * self.config.vault_a_target_pct,
            VaultType.VAULT_B: total_amount * self.config.vault_b_target_pct
        }

        self.logger.debug(f"Vault allocation: A=${allocation[VaultType.VAULT_A]:.2f}, "
                         f"B=${allocation[VaultType.VAULT_B]:.2f}")

        return allocation

    def check_rebalancing_needed(self) -> bool:
        """
        Verificar se rebalanceamento entre vaults é necessário
        Trigger: desvio > 15% da alocação target
        """

        if not self.vault_balances:
            return False

        total_vault_balance = sum(vault.balance_usd for vault in self.vault_balances.values())
        if total_vault_balance == 0:
            return False

        # Calcular alocações atuais
        vault_a_balance = self.vault_balances.get(VaultType.VAULT_A, VaultBalance(VaultType.VAULT_A, 0.0, datetime.now()))
        vault_b_balance = self.vault_balances.get(VaultType.VAULT_B, VaultBalance(VaultType.VAULT_B, 0.0, datetime.now()))

        current_a_pct = vault_a_balance.balance_usd / total_vault_balance
        current_b_pct = vault_b_balance.balance_usd / total_vault_balance

        # Verificar desvios
        a_deviation = abs(current_a_pct - self.config.vault_a_target_pct)
        b_deviation = abs(current_b_pct - self.config.vault_b_target_pct)

        needs_rebalancing = (a_deviation > self.config.rebalance_threshold_pct or
                           b_deviation > self.config.rebalance_threshold_pct)

        if needs_rebalancing:
            self.logger.info(f"Rebalancing needed: A={current_a_pct:.1%} (target {self.config.vault_a_target_pct:.1%}), "
                           f"B={current_b_pct:.1%} (target {self.config.vault_b_target_pct:.1%})")

        return needs_rebalancing

    async def execute_siphoning(self, current_equity: float, current_pnl: float) -> Dict[str, Any]:
        """
        Executar processo completo de sifonamento
        """

        try:
            self.current_equity = current_equity
            self._reset_daily_limits_if_needed()

            result = {
                'timestamp': datetime.now(),
                'equity': current_equity,
                'pnl': current_pnl,
                'actions_taken': [],
                'transfers': [],
                'errors': []
            }

            # 1. Verificar se recomposição é necessária
            if self.check_recomposition_needed(current_equity):
                if not self.is_cooldown_active():
                    recomp_result = await self._execute_recomposition()
                    result['actions_taken'].append('recomposition')
                    result['transfers'].extend(recomp_result.get('transfers', []))
                else:
                    cooldown_remaining = (self.cooldown_end_time - datetime.now()).total_seconds() / 3600
                    result['errors'].append(f"Recomposition needed but cooldown active ({cooldown_remaining:.1f}h remaining)")

            # 2. Calcular sifonamento disponível
            siphoning_amount = self.calculate_siphoning_amount(current_equity)
            result['siphoning_available'] = siphoning_amount

            if siphoning_amount >= self.config.min_transfer_amount:
                # 3. Executar sifonamento
                siphon_result = await self._execute_siphoning_transfer(siphoning_amount)
                result['actions_taken'].append('siphoning')
                result['transfers'].extend(siphon_result.get('transfers', []))

                # 4. Verificar necessidade de rebalanceamento
                if self.check_rebalancing_needed():
                    rebalance_result = await self._execute_rebalancing()
                    result['actions_taken'].append('rebalancing')
                    result['transfers'].extend(rebalance_result.get('transfers', []))

            # 5. Atualizar analytics
            self._update_analytics(current_equity, current_pnl, result)

            return result

        except Exception as e:
            self.logger.error(f"Error in siphoning execution: {str(e)}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'actions_taken': [],
                'transfers': []
            }

    async def _execute_recomposition(self) -> Dict[str, Any]:
        """
        Executar recomposição: transferir $100 dos vaults para trading
        """

        self.logger.info("Executing recomposition: adding $100 to trading account")

        transfers = []
        recomp_amount = self.config.recomposition_amount

        # Buscar $100 dos vaults (prioridade: Vault B, depois Vault A)
        remaining_needed = recomp_amount

        for vault_type in [VaultType.VAULT_B, VaultType.VAULT_A]:
            if remaining_needed <= 0:
                break

            vault_balance = self.vault_balances.get(vault_type)
            if not vault_balance or vault_balance.balance_usd <= 0:
                continue

            # Calcular quanto retirar deste vault
            withdrawal_amount = min(remaining_needed, vault_balance.balance_usd)

            if withdrawal_amount >= self.config.min_transfer_amount:
                # Simular transferência (na implementação real, seria chamada de API)
                transfer = self._create_transfer_record(
                    from_location=vault_type.value,
                    to_location="trading_account",
                    amount=withdrawal_amount,
                    transfer_type="recomposition"
                )

                transfers.append(transfer)
                self.transfer_history.append(transfer)

                # Atualizar saldo do vault
                vault_balance.balance_usd -= withdrawal_amount
                remaining_needed -= withdrawal_amount

                self.logger.info(f"Recomposition transfer: ${withdrawal_amount:.2f} from {vault_type.value}")

        # Ativar cooldown
        self._activate_cooldown()

        return {
            'transfers': transfers,
            'total_amount': recomp_amount - remaining_needed,
            'remaining_needed': remaining_needed
        }

    async def _execute_siphoning_transfer(self, amount: float) -> Dict[str, Any]:
        """
        Executar transferência de sifonamento para os vaults
        """

        if not self._check_transfer_limits(amount):
            return {'transfers': [], 'error': 'Transfer limits exceeded'}

        # Calcular alocação entre vaults
        vault_allocation = self.calculate_vault_allocation(amount)
        transfers = []

        for vault_type, allocated_amount in vault_allocation.items():
            if allocated_amount >= self.config.min_transfer_amount:
                # Simular transferência
                transfer = self._create_transfer_record(
                    from_location="trading_account",
                    to_location=vault_type.value,
                    amount=allocated_amount,
                    transfer_type="siphoning"
                )

                transfers.append(transfer)
                self.transfer_history.append(transfer)

                # Atualizar saldo do vault
                if vault_type not in self.vault_balances:
                    self.vault_balances[vault_type] = VaultBalance(vault_type, 0.0, datetime.now())

                self.vault_balances[vault_type].balance_usd += allocated_amount
                self.vault_balances[vault_type].last_updated = datetime.now()

                self.logger.info(f"Siphoning transfer: ${allocated_amount:.2f} to {vault_type.value}")

        # Atualizar contadores diários
        self.daily_transfer_count += len(transfers)
        self.daily_transfer_amount += amount

        return {'transfers': transfers, 'total_amount': amount}

    async def _execute_rebalancing(self) -> Dict[str, Any]:
        """
        Executar rebalanceamento entre Vault A e B
        """

        self.logger.info("Executing vault rebalancing")

        vault_a = self.vault_balances.get(VaultType.VAULT_A)
        vault_b = self.vault_balances.get(VaultType.VAULT_B)

        if not vault_a or not vault_b:
            return {'transfers': [], 'error': 'Insufficient vault data for rebalancing'}

        total_balance = vault_a.balance_usd + vault_b.balance_usd
        target_a = total_balance * self.config.vault_a_target_pct
        target_b = total_balance * self.config.vault_b_target_pct

        transfers = []

        # Determinar direção da transferência
        if vault_a.balance_usd > target_a:
            # Transferir de A para B
            transfer_amount = vault_a.balance_usd - target_a
            from_vault, to_vault = VaultType.VAULT_A, VaultType.VAULT_B
        else:
            # Transferir de B para A
            transfer_amount = vault_b.balance_usd - target_b
            from_vault, to_vault = VaultType.VAULT_B, VaultType.VAULT_A

        if transfer_amount >= self.config.min_transfer_amount and self._check_transfer_limits(transfer_amount):
            transfer = self._create_transfer_record(
                from_location=from_vault.value,
                to_location=to_vault.value,
                amount=transfer_amount,
                transfer_type="rebalancing"
            )

            transfers.append(transfer)
            self.transfer_history.append(transfer)

            # Atualizar saldos
            self.vault_balances[from_vault].balance_usd -= transfer_amount
            self.vault_balances[to_vault].balance_usd += transfer_amount

            self.logger.info(f"Rebalancing transfer: ${transfer_amount:.2f} from {from_vault.value} to {to_vault.value}")

        return {'transfers': transfers}

    def _create_transfer_record(
        self,
        from_location: str,
        to_location: str,
        amount: float,
        transfer_type: str
    ) -> TransferRecord:
        """Criar registro de transferência"""

        transfer_id = f"{transfer_type}_{int(time.time())}_{np.random.randint(1000, 9999)}"

        return TransferRecord(
            id=transfer_id,
            timestamp=datetime.now(),
            from_location=from_location,
            to_location=to_location,
            amount=amount,
            currency="USDT",
            status=TransferStatus.COMPLETED,  # Simulado como completo
            transfer_type=transfer_type,
            metadata={
                'equity_at_time': self.current_equity,
                'retention_rule_applied': True
            }
        )

    def _activate_cooldown(self):
        """Ativar período de cooldown após recomposição"""

        # Cooldown aleatório entre 24-48h
        cooldown_hours = np.random.uniform(
            self.config.cooldown_hours_min,
            self.config.cooldown_hours_max
        )

        self.cooldown_active = True
        self.cooldown_end_time = datetime.now() + timedelta(hours=cooldown_hours)
        self.last_recomposition = datetime.now()

        self.logger.info(f"Cooldown activated for {cooldown_hours:.1f} hours until {self.cooldown_end_time}")

    def _check_transfer_limits(self, amount: float) -> bool:
        """Verificar limites de transferência"""

        # Check daily limits
        if self.daily_transfer_count >= self.config.max_daily_transfers:
            self.logger.warning("Daily transfer count limit reached")
            return False

        if self.daily_transfer_amount + amount > self.config.max_single_transfer * 5:  # 5x daily limit
            self.logger.warning("Daily transfer amount limit would be exceeded")
            return False

        # Check single transfer limit
        if amount > self.config.max_single_transfer:
            self.logger.warning(f"Single transfer limit exceeded: ${amount:.2f} > ${self.config.max_single_transfer:.2f}")
            return False

        return True

    def _reset_daily_limits_if_needed(self):
        """Reset daily limits if new day"""

        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        if current_date > self.last_daily_reset:
            self.daily_transfer_count = 0
            self.daily_transfer_amount = 0.0
            self.last_daily_reset = current_date
            self.logger.info("Daily transfer limits reset")

    def _update_analytics(self, current_equity: float, current_pnl: float, result: Dict[str, Any]):
        """Atualizar analytics do sistema"""

        self.profitability_history.append({
            'timestamp': datetime.now(),
            'equity': current_equity,
            'pnl': current_pnl,
            'retention_amount': self.calculate_retention_amount(current_equity),
            'siphoning_amount': self.calculate_siphoning_amount(current_equity),
            'actions': result['actions_taken']
        })

        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.config.profitability_lookback_days)
        self.profitability_history = [
            record for record in self.profitability_history
            if record['timestamp'] >= cutoff_date
        ]

        # Update analytics summary
        self._calculate_analytics_summary()

    def _calculate_analytics_summary(self):
        """Calcular resumo de analytics"""

        if not self.profitability_history:
            return

        recent_data = self.profitability_history[-30:] if len(self.profitability_history) >= 30 else self.profitability_history

        total_siphoned = sum(record['siphoning_amount'] for record in recent_data)
        total_recomposed = len([r for r in self.transfer_history if r.transfer_type == 'recomposition']) * self.config.recomposition_amount

        self.siphoning_analytics = {
            'total_siphoned_30d': total_siphoned,
            'total_recomposed': total_recomposed,
            'net_siphoned': total_siphoned - total_recomposed,
            'siphoning_events': len([r for r in recent_data if r['actions'] and 'siphoning' in r['actions']]),
            'recomposition_events': len([r for r in self.transfer_history if r.transfer_type == 'recomposition']),
            'avg_retention': np.mean([r['retention_amount'] for r in recent_data]),
            'vault_balances': {vt.value: vb.balance_usd for vt, vb in self.vault_balances.items()},
            'cooldown_active': self.cooldown_active,
            'last_recomposition': self.last_recomposition.isoformat() if self.last_recomposition else None
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Obter status completo do sistema"""

        total_vault_balance = sum(vault.balance_usd for vault in self.vault_balances.values())

        return {
            'timestamp': datetime.now(),
            'current_equity': self.current_equity,
            'retention_amount': self.calculate_retention_amount(self.current_equity),
            'siphoning_available': self.calculate_siphoning_amount(self.current_equity),
            'recomposition_needed': self.check_recomposition_needed(self.current_equity),
            'cooldown_active': self.is_cooldown_active(),
            'cooldown_end_time': self.cooldown_end_time.isoformat() if self.cooldown_end_time else None,
            'vault_balances': {vt.value: vb.balance_usd for vt, vb in self.vault_balances.items()},
            'total_vault_balance': total_vault_balance,
            'rebalancing_needed': self.check_rebalancing_needed(),
            'daily_transfers': {
                'count': self.daily_transfer_count,
                'amount': self.daily_transfer_amount,
                'limits': {
                    'max_count': self.config.max_daily_transfers,
                    'max_single': self.config.max_single_transfer
                }
            },
            'analytics': self.siphoning_analytics,
            'recent_transfers': [
                {
                    'id': t.id,
                    'timestamp': t.timestamp.isoformat(),
                    'from': t.from_location,
                    'to': t.to_location,
                    'amount': t.amount,
                    'type': t.transfer_type
                }
                for t in self.transfer_history[-10:]  # Last 10 transfers
            ]
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obter métricas de performance do sistema"""

        if not self.profitability_history:
            return {'status': 'insufficient_data'}

        recent_30d = self.profitability_history[-30:] if len(self.profitability_history) >= 30 else self.profitability_history

        # Calculate key metrics
        total_siphoned = sum(record['siphoning_amount'] for record in recent_30d)
        siphoning_frequency = len([r for r in recent_30d if 'siphoning' in r.get('actions', [])]) / len(recent_30d)

        # Retention efficiency
        avg_retention = np.mean([r['retention_amount'] for r in recent_30d])
        retention_stability = 1.0 - np.std([r['retention_amount'] for r in recent_30d]) / avg_retention if avg_retention > 0 else 0.0

        # Vault balance distribution
        vault_a_balance = self.vault_balances.get(VaultType.VAULT_A, VaultBalance(VaultType.VAULT_A, 0.0, datetime.now())).balance_usd
        vault_b_balance = self.vault_balances.get(VaultType.VAULT_B, VaultBalance(VaultType.VAULT_B, 0.0, datetime.now())).balance_usd
        total_vaults = vault_a_balance + vault_b_balance

        return {
            'period_days': len(recent_30d),
            'total_siphoned': total_siphoned,
            'siphoning_frequency': siphoning_frequency,
            'avg_daily_siphoned': total_siphoned / len(recent_30d) if recent_30d else 0.0,
            'retention_metrics': {
                'average': avg_retention,
                'stability': retention_stability,
                'min_applied': min([r['retention_amount'] for r in recent_30d]),
                'max_applied': max([r['retention_amount'] for r in recent_30d])
            },
            'vault_distribution': {
                'vault_a_pct': vault_a_balance / total_vaults if total_vaults > 0 else 0.0,
                'vault_b_pct': vault_b_balance / total_vaults if total_vaults > 0 else 0.0,
                'total_balance': total_vaults,
                'target_deviation': abs((vault_a_balance / total_vaults) - self.config.vault_a_target_pct) if total_vaults > 0 else 0.0
            },
            'recomposition_stats': {
                'events': len([r for r in self.transfer_history if r.transfer_type == 'recomposition']),
                'total_amount': len([r for r in self.transfer_history if r.transfer_type == 'recomposition']) * self.config.recomposition_amount,
                'last_event': self.last_recomposition.isoformat() if self.last_recomposition else None
            }
        }