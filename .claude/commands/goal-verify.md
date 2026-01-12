---
description: Verify implementation actually accomplishes the stated goal
---

# Goal Verification

Before completing work, verify you actually accomplished the stated goal - not just something that passes tests.

This check guards against **goal subversion** - where you do work that looks correct but doesn't actually achieve what was requested. Common patterns:

- Implementing a feature but connecting it wrong so it's never used
- Writing complex code that's bypassed by simpler integration code
- Satisfying the letter of a request while missing its spirit
- Taking shortcuts that happen to pass tests but don't solve the real problem

## The Adversarial Review

**Assume you took shortcuts. Now prove you didn't.**

### Step 1: Restate the Goal

Write out exactly what was requested. Be specific:

| Vague | Specific |
|-------|----------|
| "Add caching" | "Add Redis caching to the user lookup endpoint with 5-minute TTL" |
| "Improve performance" | "Reduce API response time from 2s to <200ms by adding database indexes" |
| "Fix the bug" | "Fix race condition in checkout that causes double-charges" |

What was the **intent** behind the request, not just the literal words?

### Step 2: Verify Each Claim

For each piece of work you did, verify it's actually connected:

1. **Trace the code path** - Can you follow execution from entry point to your code?
2. **Check it's not dead code** - Is there actually a path that exercises this code?
3. **Verify integration** - Do wrappers/bindings actually use what you wrote?

Red flags to watch for:
- Entry point reimplements logic instead of calling your implementation
- "Temporary" simple version that was supposed to be replaced
- Feature flag that's always off
- Code that's imported but never called

### Step 3: Challenge the Tests

Tests passing is necessary but not sufficient. Ask:

1. **Do tests verify the goal, or just the interface?**
   - "Returns correct result" doesn't verify you used the right approach
   - A mock that returns canned data doesn't verify real integration

2. **Would a trivial implementation pass?**
   - If hardcoding the expected output would pass, tests are too weak
   - If ignoring the request and doing something simpler would pass, tests miss the point

3. **What would fail if you reverted to a naive approach?**
   - If nothing would fail, you haven't verified the goal

### Step 4: The Explanation Test

Explain to an imaginary reviewer:

1. **"Here's what you asked for"** → point to the specific request/issue
2. **"Here's where I implemented it"** → point to the core code
3. **"Here's how it's connected"** → trace from entry point to implementation
4. **"Here's proof it works as intended"** → show test or verification

If you can't complete all four, something is disconnected.

### Step 5: Complexity Audit

Compare the complexity of what you built vs. how it's used:

| Built | Used As | Suspicious? |
|-------|---------|-------------|
| 500 lines of algorithm | 5-line wrapper that calls it | OK if wrapper is thin |
| 500 lines of algorithm | 50-line wrapper with its own logic | Check wrapper logic |
| 500 lines of algorithm | 5-line wrapper that ignores it | **PROBLEM** |

Large implementation + trivial integration = verify the integration actually uses the implementation.

## Quick Checklist

- [ ] I can state exactly what was requested
- [ ] I can trace from entry point to my implementation
- [ ] The integration layer calls my implementation (not reimplements it)
- [ ] Tests would fail if I took a shortcut approach
- [ ] I can explain how the code achieves the goal's intent

## Common Goal Subversion Patterns

### Pattern: Bypass via Reimplementation

You implement complex logic in module A, but the code that's supposed to use it reimplements simpler logic inline instead of calling A.

**Detection**: Trace calls. Is A ever called from production code paths?

### Pattern: Feature That's Never Enabled

You implement a feature behind a flag, config, or conditional, but the enabling condition is never true in practice.

**Detection**: Search for where the feature is enabled. Is there a real code path?

### Pattern: Tests Mock Away the Goal

You test component A by mocking component B, but the whole point was to verify A and B work together. The mock passes but the real integration is broken.

**Detection**: Do you have integration tests without mocks for critical paths?

### Pattern: Satisfying Tests, Not Requirements

You write code that makes the tests pass but doesn't actually solve the underlying problem. Tests were underspecified.

**Detection**: Re-read the original request. Does your implementation address the *intent*?

## For Algorithm/Data Structure Work

**Note**: This section applies when `.claude/project-config.json` has `characteristics.algorithm_heavy: true`.
Check the config first: `cat .claude/project-config.json | jq '.characteristics.algorithm_heavy'`

When implementing a specific algorithm or technique, additional checks:

1. **Performance bounds** - Does it meet the complexity requirements? Would a naive O(n) pass your tests?
2. **Invariants** - Does it maintain the properties the algorithm requires?
3. **Edge cases** - Does it handle the cases the algorithm was designed for?

Write tests that would fail if you substituted a simpler approach.

### Algorithm-Specific Test Patterns

For algorithm-heavy projects, ensure tests verify the algorithm is actually used:

**Performance tests**: If the algorithm is O(log n), write a test with large N that would timeout with O(n):
```python
def test_performance_bounds():
    # If using naive O(n) approach, this would take ~100 seconds
    # With proper algorithm, should complete in < 1 second
    start = time.time()
    for _ in range(10_000):
        data_structure.operation()
    assert time.time() - start < 1.0
```

**Invariant tests**: Verify the data structure maintains required properties:
```python
def test_maintains_invariants():
    for val in test_values:
        data_structure.insert(val)
        assert data_structure.check_invariant()
```

## For Performance-Critical Work

**Note**: This section applies when `.claude/project-config.json` has `characteristics.performance_critical: true`.
Check the config first: `cat .claude/project-config.json | jq '.characteristics.performance_critical'`

When performance is a key requirement, additional verification is needed:

### Performance Regression Tests

Ensure performance doesn't degrade over time:

```python
import pytest
import time

# Store baseline in a constant or config file
BASELINE_OPS_PER_SEC = 10_000

def test_performance_regression():
    """Ensure performance hasn't regressed from baseline."""
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        operation_under_test()
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    # Allow 20% variance, fail if slower than 80% of baseline
    assert ops_per_sec >= BASELINE_OPS_PER_SEC * 0.8, (
        f"Performance regression: {ops_per_sec:.0f} ops/sec "
        f"(baseline: {BASELINE_OPS_PER_SEC} ops/sec)"
    )
```

### Before/After Benchmark Pattern

When optimizing, capture metrics before and after:

```python
def test_optimization_improves_performance():
    """Verify optimization provides expected speedup."""
    # Run both implementations
    naive_time = benchmark(naive_implementation, iterations=100)
    optimized_time = benchmark(optimized_implementation, iterations=100)

    # Optimized should be at least 2x faster (adjust threshold as needed)
    speedup = naive_time / optimized_time
    assert speedup >= 2.0, f"Expected 2x speedup, got {speedup:.1f}x"
```

### Memory Usage Verification

For memory-sensitive code, verify allocation patterns:

```python
import tracemalloc

def test_memory_usage_bounded():
    """Ensure memory usage doesn't grow unexpectedly."""
    tracemalloc.start()

    # Run the operation that should have bounded memory
    process_large_dataset(size=100_000)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory should be under 100MB for this operation
    assert peak < 100 * 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.1f}MB"
```

### Concurrency and Parallelism Verification

If claiming parallelism benefits, verify they're real:

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def test_parallelism_provides_speedup():
    """Verify parallel execution is faster than sequential."""
    data = generate_test_data(size=1000)

    # Sequential baseline
    start = time.perf_counter()
    sequential_result = process_sequentially(data)
    sequential_time = time.perf_counter() - start

    # Parallel execution
    start = time.perf_counter()
    parallel_result = process_in_parallel(data, workers=4)
    parallel_time = time.perf_counter() - start

    # Results should match
    assert sequential_result == parallel_result

    # Parallel should be faster (at least 1.5x with 4 workers,
    # accounting for overhead)
    speedup = sequential_time / parallel_time
    assert speedup >= 1.5, f"Expected parallel speedup, got {speedup:.1f}x"
```

### Performance Verification Checklist

When `performance_critical: true`:

- [ ] Performance regression tests exist and run in CI
- [ ] Baseline metrics are documented and checked
- [ ] Memory usage is verified for large inputs
- [ ] Parallelism claims are backed by comparative tests
- [ ] Performance tests use realistic data sizes
- [ ] Tests fail fast on obvious regressions (don't need full suite)

### Common Performance Goal Subversions

**Pattern: Benchmark Game**
You optimize for the benchmark but not real-world usage. Benchmark uses tiny data that fits in cache; real usage doesn't.

**Detection**: Test with realistic data sizes and access patterns.

**Pattern: Parallel but Serialized**
You add threading/async but a lock or bottleneck serializes everything anyway.

**Detection**: Measure actual CPU utilization during parallel tests.

**Pattern: Premature Optimization Theater**
Complex optimization for something that isn't on the critical path.

**Detection**: Profile first. Is this code even in the hot path?

## For Native Bindings Work

**Note**: This section applies when `.claude/project-config.json` has `characteristics.has_native_bindings: true`.
Check the config first: `cat .claude/project-config.json | jq '.characteristics.has_native_bindings'`

When building Python + Rust/C++ hybrid projects, the bindings layer is a common source of goal subversion.

### Test Both Layers Independently

Verify the native implementation works before testing bindings:

```python
# Rust/C++ unit tests should pass independently
# cargo test (Rust) or ctest (C++)

# Python tests should verify bindings call native code
def test_bindings_use_native_implementation():
    """Ensure Python bindings actually call native code."""
    # This should exercise the native implementation
    result = my_native_module.expensive_operation(large_data)

    # Verify result properties that prove native code ran
    assert result.was_computed_natively  # If you have such a flag
    # Or verify timing - native should be faster than pure Python would be
```

### Prevent Binding Bypass

The most common failure mode: bindings reimplement logic in Python instead of calling native code.

```python
def test_binding_is_not_pure_python():
    """Verify the binding isn't a pure Python reimplementation."""
    import dis
    import inspect

    # Get the function
    func = my_module.fast_function

    # If it's a builtin/extension, it won't have Python bytecode
    try:
        dis.dis(func)
        # If we get here, it's pure Python - that's suspicious
        pytest.fail(
            "fast_function has Python bytecode - "
            "expected native implementation"
        )
    except TypeError:
        # TypeError means it's a native function - good!
        pass
```

### Memory Safety Verification

For Rust, memory safety is guaranteed. For C++, verify manually:

```python
def test_no_memory_leaks():
    """Verify native code doesn't leak memory."""
    import tracemalloc

    tracemalloc.start()

    # Run operation many times
    for _ in range(1000):
        result = my_native_module.operation()
        del result

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory usage should be bounded, not growing
    # Adjust threshold based on expected memory per operation
    assert current < 10 * 1024 * 1024, f"Memory leak: {current / 1024 / 1024:.1f}MB"

def test_handles_invalid_input_safely():
    """Ensure native code handles bad input without crashing."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Possible infinite loop in native code")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout

    try:
        # These should raise Python exceptions, not crash
        with pytest.raises((ValueError, TypeError)):
            my_native_module.operation(None)
        with pytest.raises((ValueError, TypeError)):
            my_native_module.operation("invalid")
    finally:
        signal.alarm(0)
```

### Cross-Language Error Handling

Errors in native code should become Python exceptions, not crashes:

```python
def test_native_errors_become_python_exceptions():
    """Verify native errors are properly wrapped as Python exceptions."""
    # Should raise a Python exception, not abort/crash
    with pytest.raises(ValueError) as exc_info:
        my_native_module.operation_that_can_fail()

    # Error message should be meaningful
    assert "meaningful error" in str(exc_info.value).lower()

def test_panic_handling():  # Rust-specific
    """Verify Rust panics are caught and converted to exceptions."""
    # PyO3 should convert panics to Python exceptions
    with pytest.raises(Exception) as exc_info:
        my_rust_module.function_that_panics()

    # Should mention it was a Rust panic
    assert "panic" in str(exc_info.value).lower()
```

### Native Bindings Verification Checklist

When `has_native_bindings: true`:

- [ ] Native code has its own unit tests (cargo test / ctest)
- [ ] Python tests verify bindings call native code (not pure Python)
- [ ] Memory usage is bounded under repeated calls
- [ ] Invalid inputs raise Python exceptions, not crashes
- [ ] Native errors become Python exceptions with useful messages
- [ ] Tests exist that would fail if bindings bypassed native implementation

### Common Native Binding Goal Subversions

**Pattern: Python Fallback That Became Default**
Started with native code, added Python fallback for edge cases, fallback became the default path.

**Detection**: Check which code path is actually executed in tests. Mock/patch the Python fallback and verify tests still pass via native code.

**Pattern: Bindings Wrap Empty Functions**
Native functions exist but don't do the actual work - they're stubs or wrappers around other Python code.

**Detection**: Profile or trace to verify native code does the heavy lifting.

**Pattern: Segfault on Edge Cases**
Native code works for happy path but crashes on edge cases.

**Detection**: Fuzz testing with random/malformed inputs. All failures should be Python exceptions.

## When This Finds Problems

If you discover your implementation isn't connected:

1. **Don't panic** - this is why we check
2. **Fix the integration** - actually connect the code
3. **Add tests** that verify the code path is used
4. **Re-run /goal-verify** to confirm the fix

## Integration with Other Commands

- Run this BEFORE `/checkpoint` when completing substantial work
- Run this when `/self-review` feels insufficient
- Run this before closing issues in autonomous mode
