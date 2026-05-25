"""
llm-batch-coalesce: Single-flight / request coalescing for LLM calls.

Multiple callers asking for the same prompt at the same time share one real call.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _hash_request(messages: list[dict[str, Any]], model: str = "", **extras: Any) -> str:
    payload = {"messages": messages, "model": model, **extras}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class _Flight:
    """An in-progress LLM call shared by multiple waiters."""
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class BatchCoalesce:
    """
    Coalesce concurrent identical LLM requests into a single real call.

    If two threads call ``get_or_call(messages, call_fn)`` with the same
    request at overlapping times, only one real ``call_fn`` fires;
    the second waits and gets the same result.

    Usage::

        coalesce = BatchCoalesce()

        def call_llm(messages, model="claude"):
            return real_api_client.create(messages=messages, model=model)

        result = coalesce.get_or_call(messages, call_llm, model="claude")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}
        self._call_count = 0       # real calls made
        self._coalesced_count = 0  # calls that waited on an in-flight request

    def get_or_call(
        self,
        messages: list[dict[str, Any]],
        call_fn: Callable[..., Any],
        model: str = "",
        **extras: Any,
    ) -> Any:
        """
        Return the result for the given request, calling ``call_fn`` if not
        already in flight. Thread-safe.
        """
        key = _hash_request(messages, model=model, **extras)
        with self._lock:
            if key in self._flights:
                flight = self._flights[key]
                self._coalesced_count += 1
                is_leader = False
            else:
                flight = _Flight()
                self._flights[key] = flight
                is_leader = True

        if is_leader:
            try:
                result = call_fn(messages, model=model, **extras) if model else call_fn(messages, **extras)
                with self._lock:
                    self._call_count += 1
                flight.result = result
            except BaseException as exc:
                flight.error = exc
            finally:
                flight.event.set()
                with self._lock:
                    self._flights.pop(key, None)
        else:
            flight.event.wait()

        if flight.error is not None:
            raise flight.error
        return flight.result

    def wrap(self, model: str = "") -> Callable[..., Any]:
        """Decorator: wrap a ``call_fn(messages, **kwargs)`` with coalescing."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
                return self.get_or_call(messages, fn, model=model, **kwargs)
            return wrapper
        return decorator

    @property
    def call_count(self) -> int:
        """Number of real LLM calls made."""
        return self._call_count

    @property
    def coalesced_count(self) -> int:
        """Number of calls that shared an in-flight request."""
        return self._coalesced_count

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._flights)

    def stats(self) -> dict[str, int]:
        return {
            "call_count": self._call_count,
            "coalesced_count": self._coalesced_count,
            "in_flight": self.in_flight,
        }


__all__ = ["BatchCoalesce"]
