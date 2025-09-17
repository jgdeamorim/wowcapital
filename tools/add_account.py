#!/usr/bin/env python3
from __future__ import annotations
"""CLI para adicionar contas em backend/config/accounts.yaml com segurança.

Uso:
  PYTHONPATH=. python3 backend/tools/add_account.py --venue binance --account-id lab1 \
    --mode spot --api-key-env BINANCE_API_KEY --api-secret-env BINANCE_API_SECRET

Ou Bybit:
  PYTHONPATH=. python3 backend/tools/add_account.py --venue bybit --account-id lab1 \
    --category spot --api-key-env BYBIT_API_KEY --api-secret-env BYBIT_API_SECRET
"""
import argparse
from pathlib import Path
import yaml


def load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def save_yaml(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=["binance", "bybit", "kraken"], help="Exchange")
    ap.add_argument("--account-id", required=True, help="ID da conta (label interna)")
    ap.add_argument("--mode", help="Modo binance: spot|futures")
    ap.add_argument("--category", help="Categoria bybit: spot|linear|inverse")
    ap.add_argument("--api-key-env", required=True, help="Nome da env com API KEY")
    ap.add_argument("--api-secret-env", required=True, help="Nome da env com API SECRET")
    ap.add_argument("--file", default="backend/config/accounts.yaml", help="Caminho do YAML de contas")
    args = ap.parse_args()

    p = Path(args.file)
    data = load_yaml(p)
    ex = data.setdefault("exchanges", {}).setdefault(args.venue, {}).setdefault("accounts", {})
    acc = ex.setdefault(args.account_id, {})
    if args.venue == "binance" and args.mode:
        acc["mode"] = args.mode
    if args.venue == "bybit" and args.category:
        acc["category"] = args.category
    # set placeholders referencing env vars
    acc["api_key"] = f"${{{args.api_key_env}:-}}"
    acc["api_secret"] = f"${{{args.api_secret_env}:-}}"
    save_yaml(p, data)
    print(f"✅ Conta {args.venue}:{args.account_id} adicionada em {p}")


if __name__ == "__main__":
    main()

