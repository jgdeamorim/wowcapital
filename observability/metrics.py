from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Selector
SELECTOR_FAILOVERS = Counter("selector_failovers_total", "Selector failovers", ["frm","to"]) 
SELECTOR_ACTIVE = Gauge("selector_active_source", "Active source (info-like)", ["class","symbol","source","tier"])

# Aggressive framework
AGGR_EXPLOSIONS = Counter("aggr_explosions_total", "Explosions count", ["account","exchange"])
AGGR_RESETS = Counter("aggr_resets_total", "Resets count", ["account","reason"])
AGGR_COFRE_BAL = Gauge("aggr_cofre_usdt_balance", "Cofre balance (USDT)")
FEE_TO_EDGE = Gauge("fee_to_edge_ratio", "Fee-to-edge ratio", ["strategy","symbol"])

# Execution
EXEC_LAT = Histogram("exec_order_latency_seconds", "Order latency", ["venue","side","ordertype"],
                    buckets=[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0])

# Acks/Fills and timings
ACKS_TOTAL = Counter("exec_acks_total", "Order acks", ["venue"])
FILLS_TOTAL = Counter("exec_fills_total", "Order fills", ["venue","symbol","side"])
ACK_TO_FILL = Histogram("exec_ack_to_fill_seconds", "Ack to fill latency", ["venue","symbol"],
                       buckets=[0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0])

# Cofre sweep totals
COFRE_SWEEP_USDT = Counter("cofre_sweep_usdt_total", "Total swept to cofre (USDT)")

# Exporter
metrics_app = make_asgi_app()

# HTTP server metrics
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["route", "method", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["route", "method"],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# Circuit breaker
CB_TRIPS = Counter("cb_trips_total", "Circuit breaker trips", ["venue"]) 
CB_BLOCKS = Counter("cb_blocks_total", "Orders blocked by circuit breaker", ["venue"]) 
SIZE_MULTIPLIER = Gauge("plugin_size_multiplier", "Size multiplier per plugin", ["plugin_id"]) 

# Market data WS health
WS_UP = Gauge("ws_up", "WebSocket stream up (1) or down (0)", ["venue", "kind"]) 
WS_MESSAGES = Counter("ws_messages_total", "WS messages processed", ["venue", "kind"]) 
WS_RECONNECTS = Counter("ws_reconnects_total", "WS reconnects", ["venue", "kind"]) 

# Micro persistence
MICRO_PERSIST_TOTAL = Counter("micro_persist_total", "Micro snapshots persisted", ["venue"]) 

# AI orchestration telemetry
AI_DECISIONS = Counter(
    "ai_orchestrator_decisions_total",
    "AI orchestrator decisions emitted",
    ["plugin", "action"],
)
AI_EXECUTIONS = Counter(
    "ai_orchestrator_execution_total",
    "AI orchestrator execution outcomes",
    ["plugin", "status"],
)
AI_DECISION_LATENCY = Histogram(
    "ai_orchestrator_decision_seconds",
    "Latency of orchestrator decisions",
    ["plugin"],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)
AI_TOOL_CALLS = Counter(
    "ai_tool_calls_total",
    "AI tool call invocations",
    ["tool", "status"],
)
