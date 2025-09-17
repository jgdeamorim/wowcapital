from __future__ import annotations
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Optional


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


class AsyncWormLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._prev_hash: str = self._load_tail_hash() or "0" * 64
        self._sig_key = os.getenv("AUDIT_HMAC_KEY")

    def _load_tail_hash(self) -> Optional[str]:
        try:
            if not self.path.exists():
                return None
            with self.path.open("rb") as f:
                try:
                    f.seek(-4096, 2)
                except OSError:
                    f.seek(0)
                tail = f.read().splitlines()
                if not tail:
                    return None
                last = tail[-1]
                rec = json.loads(last)
                return rec.get("hash")
        except Exception:
            return None

    async def append(self, event: dict[str, Any]) -> None:
        async with self._lock:
            # Avoid mutating caller dict order unexpectedly
            ev = dict(event)
            ev["prev_hash"] = self._prev_hash
            # Compute chain hash using canonical JSON (without the final hash/sig fields)
            ev_for_hash = {k: v for k, v in ev.items() if k not in ("hash", "sig")}
            payload = _canonical_json(ev_for_hash)
            curr_hash = hashlib.sha256((self._prev_hash + payload).encode()).hexdigest()
            ev["hash"] = curr_hash
            # Optional HMAC signature
            if self._sig_key:
                try:
                    ev["sig"] = hmac.new(self._sig_key.encode(), curr_hash.encode(), hashlib.sha256).hexdigest()
                except Exception:
                    pass
            line = json.dumps(ev, separators=(",", ":"), ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._prev_hash = curr_hash

    def verify_chain(self) -> dict[str, Any]:
        """Verify the hash-chain and optional HMAC signatures.
        Returns {ok: bool, last_hash: str, count: int, errors: list}
        """
        errors: list[dict[str, Any]] = []
        prev = "0" * 64
        count = 0
        last_hash = prev
        if not self.path.exists():
            return {"ok": True, "last_hash": prev, "count": 0, "errors": []}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception as e:
                        errors.append({"line": idx, "error": f"JSON:{e}"})
                        break
                    # Recompute hash
                    ev_for_hash = {k: v for k, v in rec.items() if k not in ("hash", "sig")}
                    payload = _canonical_json({**ev_for_hash, "prev_hash": prev})
                    expect_hash = hashlib.sha256((prev + payload).encode()).hexdigest()
                    got_hash = rec.get("hash")
                    if expect_hash != got_hash:
                        errors.append({"line": idx, "error": "HASH_MISMATCH", "expected": expect_hash, "got": got_hash})
                        break
                    # Verify optional signature
                    if self._sig_key:
                        try:
                            exp_sig = hmac.new(self._sig_key.encode(), expect_hash.encode(), hashlib.sha256).hexdigest()
                            if rec.get("sig") != exp_sig:
                                errors.append({"line": idx, "error": "HMAC_MISMATCH"})
                                break
                        except Exception as e:
                            errors.append({"line": idx, "error": f"HMAC_ERR:{e}"})
                            break
                    prev = expect_hash
                    last_hash = expect_hash
                    count += 1
        except Exception as e:
            errors.append({"line": 0, "error": f"IO:{e}"})
        ok = len(errors) == 0
        return {"ok": ok, "last_hash": last_hash, "count": count, "errors": errors}
