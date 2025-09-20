from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class Metadata(BaseModel):
    id: str
    name: str
    version: str
    tier: Optional[str] = Field(default="1.6")


class RuntimeCfg(BaseModel):
    entrypoint: Optional[str] = None
    aggressiveness: Optional[str] = None
    indicators: Optional[List[str]] = None
    symbols: Optional[List[str]] = None


class CofrePolicy(BaseModel):
    explosion_threshold_pct: Optional[float] = Field(default=120)
    sweep_pct: float = Field(default=0.85, ge=0.5, le=0.95)
    account_float_min_usd: int = Field(default=300, ge=0)
    account_float_max_usd: int = Field(default=500, ge=0)

    @validator("account_float_max_usd")
    def _max_ge_min(cls, v, values):
        if "account_float_min_usd" in values and v < values["account_float_min_usd"]:
            raise ValueError("account_float_max_usd must be >= min")
        return v


class RiskControls(BaseModel):
    max_leverage: Optional[float] = None
    min_notional_usd: Optional[float] = None
    max_notional_usd: Optional[float] = None
    slippage_cap_bps: Optional[float] = None
    price_guard_bps: Optional[float] = None
    funding_cap_bps: Optional[float] = None


class ManifestModel(BaseModel):
    metadata: Metadata
    runtime: Optional[RuntimeCfg] = None
    cofre_policy: Optional[CofrePolicy] = None
    risk_controls: Optional[RiskControls] = None
    guards: Optional[Dict[str, Any]] = None


# Extended schema for plug-and-play strategy manifests
class Param(BaseModel):
    name: str
    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None


class Safety(BaseModel):
    max_exposure_usd: float = Field(..., ge=0.0)
    max_leverage: int = Field(..., ge=1)
    allow_live_promotion: bool = False


class StrategyManifest(BaseModel):
    name: str
    version: str
    author: str
    description: str
    entrypoint: str  # module:Class or path.py:Class
    tags: List[str] = []
    safety: Safety
    parameters: List[Param] = Field(default_factory=list)
    expected_metrics: Dict[str, float] = Field(default_factory=dict)

    @validator("entrypoint")
    def _entry_format(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("entrypoint must be 'module:ClassName' or 'path.py:ClassName'")
        return v
