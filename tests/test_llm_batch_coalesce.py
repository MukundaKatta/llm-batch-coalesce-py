"""Tests for llm-batch-coalesce-py.

These tests use only the Python standard library (``unittest``) so they run
without any third-party dependencies. Run them with::

    python3 -m unittest discover -s tests
"""

import os
import sys
import threading
import time
import unittest

# Make the package importable without an editable install: add ``src`` to the
# path so ``import llm_batch_coalesce`` resolves when tests run from a checkout.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_batch_coalesce import BatchCoalesce  # noqa: E402


MSGS = [{"role": "user", "content": "Hello"}]


def make_fake_llm():
    """Return ``(fake_llm, call_log)``; each call appends 1 to ``call_log``."""
    call_log = []

    def fake_llm(messages, model="", **kwargs):
        call_log.append(1)
        return {"content": "answer"}

    return fake_llm, call_log


class SequentialBehaviorTests(unittest.TestCase):
    def test_single_call(self):
        coalesce = BatchCoalesce()
        fn, log = make_fake_llm()
        result = coalesce.get_or_call(MSGS, fn, model="claude")
        self.assertEqual(result["content"], "answer")
        self.assertEqual(len(log), 1)

    def test_call_count_increments(self):
        coalesce = BatchCoalesce()
        fn, _ = make_fake_llm()
        coalesce.get_or_call(MSGS, fn, model="claude")
        self.assertEqual(coalesce.call_count, 1)

    def test_different_messages_separate_calls(self):
        coalesce = BatchCoalesce()
        fn, log = make_fake_llm()
        coalesce.get_or_call([{"role": "user", "content": "A"}], fn, model="claude")
        coalesce.get_or_call([{"role": "user", "content": "B"}], fn, model="claude")
        # Sequential calls don't coalesce (no overlap).
        self.assertEqual(len(log), 2)

    def test_different_model_separate_calls(self):
        coalesce = BatchCoalesce()
        fn, log = make_fake_llm()
        coalesce.get_or_call(MSGS, fn, model="claude")
        coalesce.get_or_call(MSGS, fn, model="gpt")
        self.assertEqual(len(log), 2)

    def test_different_extras_separate_calls(self):
        coalesce = BatchCoalesce()
        fn, log = make_fake_llm()
        coalesce.get_or_call(MSGS, fn, model="claude", temperature=0.0)
        coalesce.get_or_call(MSGS, fn, model="claude", temperature=0.9)
        self.assertEqual(len(log), 2)

    def test_call_with_no_model(self):
        coalesce = BatchCoalesce()
        seen = {}

        def fn(messages, **kwargs):
            seen["model_passed"] = "model" in kwargs
            return {"content": "ok"}

        result = coalesce.get_or_call(MSGS, fn)
        self.assertEqual(result["content"], "ok")
        # When model is empty, it should not be forwarded as a kwarg.
        self.assertFalse(seen["model_passed"])

    def test_extras_forwarded_to_call_fn(self):
        coalesce = BatchCoalesce()
        captured = {}

        def fn(messages, model="", **kwargs):
            captured.update(kwargs)
            captured["model"] = model
            return {"content": "ok"}

        coalesce.get_or_call(MSGS, fn, model="claude", temperature=0.7, max_tokens=10)
        self.assertEqual(captured["model"], "claude")
        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["max_tokens"], 10)


class StatsTests(unittest.TestCase):
    def test_in_flight_initially_zero(self):
        coalesce = BatchCoalesce()
        self.assertEqual(coalesce.in_flight, 0)

    def test_coalesced_count_starts_zero(self):
        coalesce = BatchCoalesce()
        self.assertEqual(coalesce.coalesced_count, 0)

    def test_stats_keys(self):
        coalesce = BatchCoalesce()
        s = coalesce.stats()
        self.assertIn("call_count", s)
        self.assertIn("coalesced_count", s)
        self.assertIn("in_flight", s)

    def test_stats_snapshot_values(self):
        coalesce = BatchCoalesce()
        fn, _ = make_fake_llm()
        coalesce.get_or_call(MSGS, fn, model="claude")
        s = coalesce.stats()
        self.assertEqual(s, {"call_count": 1, "coalesced_count": 0, "in_flight": 0})

    def test_in_flight_returns_to_zero_after_call(self):
        coalesce = BatchCoalesce()
        fn, _ = make_fake_llm()
        coalesce.get_or_call(MSGS, fn, model="claude")
        self.assertEqual(coalesce.in_flight, 0)

    def test_reset_stats(self):
        coalesce = BatchCoalesce()
        fn, _ = make_fake_llm()
        coalesce.get_or_call(MSGS, fn, model="claude")
        self.assertEqual(coalesce.call_count, 1)
        coalesce.reset_stats()
        self.assertEqual(coalesce.call_count, 0)
        self.assertEqual(coalesce.coalesced_count, 0)
        # A subsequent call counts again from a clean slate.
        coalesce.get_or_call(MSGS, fn, model="claude")
        self.assertEqual(coalesce.call_count, 1)


class ErrorHandlingTests(unittest.TestCase):
    def test_exception_propagates(self):
        coalesce = BatchCoalesce()

        def bad_fn(messages, model="", **kwargs):
            raise ValueError("API error")

        with self.assertRaisesRegex(ValueError, "API error"):
            coalesce.get_or_call(MSGS, bad_fn, model="x")

    def test_call_count_counts_failed_real_call(self):
        """A real call that raises is still a real call and must be counted."""
        coalesce = BatchCoalesce()

        def bad_fn(messages, model="", **kwargs):
            raise ValueError("API error")

        with self.assertRaisesRegex(ValueError, "API error"):
            coalesce.get_or_call(MSGS, bad_fn, model="x")

        self.assertEqual(coalesce.call_count, 1)

    def test_in_flight_cleared_after_failure(self):
        """A failed call must not leave a stale flight registered."""
        coalesce = BatchCoalesce()

        def bad_fn(messages, model="", **kwargs):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            coalesce.get_or_call(MSGS, bad_fn, model="x")
        self.assertEqual(coalesce.in_flight, 0)

    def test_retry_after_failure_makes_new_call(self):
        """After a failure the key is freed, so a retry triggers a fresh call."""
        coalesce = BatchCoalesce()
        attempts = []

        def flaky_fn(messages, model="", **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("transient")
            return {"content": "ok"}

        with self.assertRaises(ValueError):
            coalesce.get_or_call(MSGS, flaky_fn, model="x")
        result = coalesce.get_or_call(MSGS, flaky_fn, model="x")
        self.assertEqual(result["content"], "ok")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(coalesce.call_count, 2)


class WrapDecoratorTests(unittest.TestCase):
    def test_wrap_decorator(self):
        coalesce = BatchCoalesce()
        call_log = []

        @coalesce.wrap(model="claude")
        def call_llm(messages, **kwargs):
            call_log.append(1)
            return {"content": "hi"}

        result = call_llm(MSGS)
        self.assertEqual(result["content"], "hi")
        self.assertEqual(len(call_log), 1)

    def test_wrap_preserves_function_metadata(self):
        coalesce = BatchCoalesce()

        @coalesce.wrap(model="claude")
        def call_llm(messages, **kwargs):
            """Docstring for call_llm."""
            return {"content": "hi"}

        self.assertEqual(call_llm.__name__, "call_llm")
        self.assertEqual(call_llm.__doc__, "Docstring for call_llm.")


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_same_request_coalesces(self):
        """Two threads hitting the same key at the same time share one call."""
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
            results.append(coalesce.get_or_call(MSGS, slow_fn, model="claude"))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["content"] == "shared" for r in results))
        # Only one real call should have been made.
        self.assertEqual(len(call_log), 1)
        self.assertEqual(coalesce.coalesced_count, 1)
        self.assertEqual(coalesce.call_count, 1)

    def test_many_concurrent_callers_one_real_call(self):
        """N concurrent identical requests collapse to a single real call."""
        n = 12
        coalesce = BatchCoalesce()
        call_log = []
        barrier = threading.Barrier(n)

        def slow_fn(messages, model="", **kwargs):
            call_log.append(1)
            time.sleep(0.05)
            return {"content": "shared"}

        results = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            r = coalesce.get_or_call(MSGS, slow_fn, model="claude")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), n)
        self.assertEqual(len(call_log), 1)
        self.assertEqual(coalesce.call_count, 1)
        self.assertEqual(coalesce.coalesced_count, n - 1)

    def test_concurrent_error_propagates_to_all_waiters(self):
        """Waiters coalesced onto a failing leader all observe the exception."""
        coalesce = BatchCoalesce()
        barrier = threading.Barrier(2)

        def slow_bad_fn(messages, model="", **kwargs):
            time.sleep(0.05)
            raise ValueError("boom")

        outcomes = []

        def worker():
            barrier.wait()
            try:
                coalesce.get_or_call(MSGS, slow_bad_fn, model="claude")
                outcomes.append("ok")
            except ValueError:
                outcomes.append("err")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(outcomes.count("err"), 2)
        # Only one real (failed) call should have been made.
        self.assertEqual(coalesce.call_count, 1)
        self.assertEqual(coalesce.coalesced_count, 1)


if __name__ == "__main__":
    unittest.main()
