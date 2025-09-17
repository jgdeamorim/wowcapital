from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ClassName = Literal["FX","METALS","CRYPTO","CFD","INDEX"]

class OHLCV(BaseModel):
    ts_ns: int
    o: float; h: float; l: float; c: float; v: float

class Quote(BaseModel):
    ts_ns: int
    bid: float
    ask: float
    mid: float
    spread: float

class BookLevel(BaseModel):
    px: float
    qty: float

class Book(BaseModel):
    ts_ns: int
    bids: List[BookLevel]
    asks: List[BookLevel]

class MarketSnapshot(BaseModel):
    class_: ClassName
    symbol: str
    ts_ns: int
    bid: float
    ask: float
    mid: float
    spread: float
    features: Dict[str, float] = Field(default_factory=dict)

class StrategyDecision(BaseModel):
    side: Literal["BUY","SELL","FLAT"]
    size: float
    confidence: float = 1.0
    ttl_ms: int = 800
    reason: str = ""

TIF = Literal["IOC","FOK","GTC","GTD"]
OrderType = Literal["MARKET","LIMIT","STOP","STOP_LIMIT"]

class OrderRequest(BaseModel):
    symbol: str
    side: Literal["BUY","SELL"]
    qty: float
    order_type: OrderType = "MARKET"
    price: Optional[float] = None
    tif: TIF = "IOC"
    expire_at_ns: Optional[int] = None
    client_id: str
    idempotency_key: str
    meta: Dict[str, str] = Field(default_factory=dict)

class OrderAck(BaseModel):
    client_id: str
    broker_order_id: Optional[str]
    accepted: bool
    message: str = ""
    ts_ns: int

class Fill(BaseModel):
    broker_order_id: str
    symbol: str
    side: Literal["BUY","SELL"]
    px: float
    qty: float
    fee: float
    fee_ccy: str
    ts_ns: int

class Position(BaseModel):
    symbol: str
    qty: float
    avg_price: Optional[float] = None
    side: Optional[Literal["LONG","SHORT"]] = None

class Balance(BaseModel):
    ccy: str
    free: float
    used: float
    total: float
