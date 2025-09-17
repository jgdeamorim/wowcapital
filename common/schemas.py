from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field, root_validator


class AccountEntry(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    mode: Optional[str] = None      # binance
    category: Optional[str] = None  # bybit


class VenueAccounts(BaseModel):
    accounts: Dict[str, AccountEntry] = Field(default_factory=dict)


class AccountsConfig(BaseModel):
    exchanges: Dict[str, VenueAccounts] = Field(default_factory=dict)

    @root_validator(pre=True)
    def normalize(cls, values):
        # Allow raw nested dicts
        ex = values.get('exchanges') or {}
        if isinstance(ex, dict):
            for v, data in ex.items():
                ac = data.get('accounts') or {}
                if not isinstance(ac, dict):
                    raise ValueError('accounts must be dict')
        return values


class ModelEntry(BaseModel):
    provider: str
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ModelsConfig(BaseModel):
    version: int
    default: Dict[str, str]
    models: Dict[str, ModelEntry]

