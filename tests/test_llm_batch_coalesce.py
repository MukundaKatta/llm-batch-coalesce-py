"""Tests for llm-batch-coalesce-py."""
import threading
import pytest
from llm_batch_coalesce import BatchCoalesce


MSGS = [{"role": "user", "content": "Hello"}]


def make_fake_llm(responses=None, call_log=None):
    if responses is None:
        responses = {"_": "default response"}
    if call_log is None:
        call_log = []
    def fake_llm(messages, model="", **kwargs):
        call_log.append(1)
        return {"content": "answer"}
    return fake_llm, call_log


def test_single_call():
    coalesce = BatchCoalesce()
    fn, log = make_fake_llm()
    result = coalesce.get_or_call(MSGS, fn, model="claude")
    assert result["content"] == "answer"
    assert len(log) == 1


def test_call_count_increments():
    coalesce = BatchCoalesce()
    fn, _ = make_fake_llm()
    coalesce.get_or_call(MSGS, fn, model="claude")
    assert coalesce.call_count == 1


def test_different_messages_separate_calls():
    coalesce = BatchCoalesce()
    fn, log = make_fake_llm()
    coalesce.get_or_call([{"role": "user", "content": "A"}], fn, model="claude")
    coalesce.get_or_call([{"role": "user", "content": "B"}], fn, model="claude")
    # Sequential calls don't coalesce (no overlap)
    assert len(log) == 2


def test_in_flight_initially_zero():
    coalesce = BatchCoalesce()
    assert coalesce.in_flight == 0


def test_stats_keys():
    coalesce = BatchCoalesce()
    s = coalesce.stats()
    assert "call_count" in s
    assert "coalesced_count" in s
    assert "in_flight" in s


def test_wrap_decorator():
    coalesce = BatchCoalesce()
    call_log = []

    @coalesce.wrap(model="claude")
    def call_llm(messages, **kwargs):
        call_log.append(1)
        return {"content": "hi"}

    result = call_llm(MSGS)
    assert result["content"] == "hi"
    assert len(call_log) == 1


def test_exception_propagates():
    coalesce = BatchCoalesce()

    def bad_fn(messages, model="", **kwargs):
        raise ValueError("API error")

    with pytest.raises(ValueError, match="API error"):
        coalesce.get_or_call(MSGS, bad_fn, model="x")


def test_concurrent_same_request_coalesces():
    """Two threads hitting the same key at the same time share one call."""
    import time
    coalesce = BatchCoalesce()
    call_log = []
    barrier = threading.Barrier(2)

    def slow_fn(messages, model="", **kwargs):
        call_log.append(1)
        time.sleep(0.05)
        return {"content": "shared"}

    results = []

    def worker():
        barrier.wait()
        r = coalesce.get_or_call(MSGS, slow_fn, model="claude")
        results.append(r)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert len(results) == 2
    assert all(r["content"] == "shared" for r in results)
    # Only one real call should have been made
    assert len(call_log) == 1
    assert coalesce.coalesced_count == 1


def test_coalesced_count_starts_zero():
    coalesce = BatchCoalesce()
    assert coalesce.coalesced_count == 0
