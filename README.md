# llm-batch-coalesce-py

[![CI](https://github.com/MukundaKatta/llm-batch-coalesce-py/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-batch-coalesce-py/actions/workflows/ci.yml)

Single-flight request coalescing for LLM calls.

When several callers ask for the **same** prompt at the **same** time, only one
real LLM request is made. The other callers wait on that in-flight request and
all receive the same result. This is the "single-flight" pattern (a.k.a.
request coalescing or deduplication), applied to expensive LLM calls.

- **Zero dependencies** — pure Python standard library.
- **Thread-safe** — built around a single lock and `threading.Event`.
- **Python 3.10+**.

## Why?

LLM calls are slow and often billed per token. If many parts of your app (or
many concurrent requests) ask for the same completion at the same moment, you
pay for and wait on every one of them. Coalescing collapses those duplicate,
overlapping requests into a single underlying call:

```
caller A ─┐
caller B ─┼─►  one real LLM call  ─►  result shared with A, B, C
caller C ─┘
```

Note: this is **not** a cache. Coalescing only shares a call while it is *still
in flight*. Once it completes, the entry is released and the next request makes
a fresh call. Combine it with your own cache if you want results to persist.

## Install

From source:

```bash
pip install git+https://github.com/MukundaKatta/llm-batch-coalesce-py.git
```

Or clone and install locally:

```bash
git clone https://github.com/MukundaKatta/llm-batch-coalesce-py.git
cd llm-batch-coalesce-py
pip install .
```

## Usage

```python
import threading
import time

from llm_batch_coalesce import BatchCoalesce

coalesce = BatchCoalesce()


def call_llm(messages, model="claude"):
    # Stand-in for a real, slow API call.
    time.sleep(0.1)
    return {"content": "Hello there!"}


messages = [{"role": "user", "content": "Say hi"}]
results = []


def worker():
    results.append(coalesce.get_or_call(messages, call_llm, model="claude"))


# Fire five identical requests at the same time.
threads = [threading.Thread(target=worker) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(len(results))             # 5 callers all got a result
print(coalesce.stats())         # {'call_count': 1, 'coalesced_count': 4, 'in_flight': 0}
```

Only **one** real `call_llm` actually ran; the other four were coalesced onto
it.

### Decorator form

Wrap a call function once and call it normally:

```python
coalesce = BatchCoalesce()


@coalesce.wrap(model="claude")
def chat(messages, **kwargs):
    return real_api_client.create(messages=messages, **kwargs)


# Concurrent identical calls coalesce automatically.
reply = chat([{"role": "user", "content": "Say hi"}])
```

## What counts as "the same" request?

Two requests coalesce only if they produce the same key. The key is a SHA-256
hash of:

- `messages`
- `model`
- any extra keyword arguments you pass (e.g. `temperature`, `max_tokens`)

So `get_or_call(msgs, fn, model="claude", temperature=0.0)` and the same call
with `temperature=0.9` are treated as **different** requests and do not share a
call.

## Error handling

If the single real call raises, the exception is propagated to **every** waiter
that coalesced onto it, and the in-flight entry is released so the next request
starts fresh:

```python
try:
    coalesce.get_or_call(messages, call_llm, model="claude")
except SomeAPIError:
    ...  # all coalesced callers see this, then a retry makes a new real call
```

A failed real call still counts toward `call_count`.

## API

### `BatchCoalesce()`

Create a coalescer. Each instance keeps its own in-flight table and counters,
so you can scope coalescing per client, per tenant, etc.

### `get_or_call(messages, call_fn, model="", **extras) -> Any`

Return the result for the request. If an identical request is already in
flight, wait for it and return its result; otherwise call
`call_fn(messages, model=model, **extras)` (or `call_fn(messages, **extras)`
when `model` is empty) and share its result with any concurrent waiters.
Thread-safe.

### `wrap(model="") -> decorator`

Decorator that wraps a `call_fn(messages, **kwargs)` so every invocation goes
through `get_or_call`. The wrapped function preserves the original's name and
docstring (via `functools.wraps`).

### Introspection

| Member | Description |
| --- | --- |
| `call_count` | Number of real `call_fn` invocations made (including failed ones). |
| `coalesced_count` | Number of calls that waited on an in-flight request instead of making their own. |
| `in_flight` | Number of distinct requests currently executing. |
| `stats()` | A consistent snapshot `{"call_count", "coalesced_count", "in_flight"}`. |
| `reset_stats()` | Reset `call_count` and `coalesced_count` to zero (does not touch in-flight requests). |

## Development

Run the test suite (standard library only — no third-party deps required):

```bash
python -m unittest discover -s tests -v
```

## License

MIT
