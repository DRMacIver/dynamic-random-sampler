"""Hypothesis stateful test for SamplerDict dict API conformance.

This test verifies that SamplerDict behaves like a dict[str, float],
with the additional constraint that weights must be non-negative.
"""

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)


class WeightDictModel:
    """Reference model: a plain Python dict with weight validation."""

    def __init__(self) -> None:
        self.data: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> float:
        return self.data[key]

    def __setitem__(self, key: str, weight: float) -> None:
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.data[key] = weight

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def keys(self) -> list[str]:
        return list(self.data.keys())

    def values(self) -> list[float]:
        return list(self.data.values())

    def items(self) -> list[tuple[str, float]]:
        return list(self.data.items())

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.data.get(key, default)

    def pop(self, key: str) -> float:
        return self.data.pop(key)

    def update(self, other: dict[str, float]) -> None:
        for _key, weight in other.items():
            if weight < 0:
                raise ValueError("weight must be non-negative")
        self.data.update(other)

    def clear(self) -> None:
        self.data.clear()

    def setdefault(self, key: str, default: float) -> float:
        if default < 0:
            raise ValueError("weight must be non-negative")
        return self.data.setdefault(key, default)


# Strategy for valid keys (simple ASCII strings)
key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=48),
    min_size=1,
    max_size=10,
)

# Strategy for valid weights (non-negative floats, not too extreme)
weight_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)

# Strategy for positive weights (for testing sampling)
positive_weight_strategy = st.floats(min_value=0.1, max_value=1000.0, allow_nan=False)


class SamplerDictConformance(RuleBasedStateMachine):
    """Stateful test comparing SamplerDict to WeightDictModel."""

    def __init__(self) -> None:
        super().__init__()
        self.model: WeightDictModel | None = None
        self.wd: Any = None

    @initialize()
    def init_dict(self) -> None:
        from dynamic_random_sampler import SamplerDict

        self.model = WeightDictModel()
        self.wd = SamplerDict(seed=42)

    @invariant()
    def lengths_match(self) -> None:
        if self.model is not None:
            assert len(self.wd) == len(self.model), (
                f"Length mismatch: wd={len(self.wd)}, model={len(self.model)}"
            )

    @invariant()
    def keys_match(self) -> None:
        if self.model is not None:
            wd_keys = set(self.wd.keys())
            model_keys = set(self.model.keys())
            assert wd_keys == model_keys, (
                f"Keys mismatch: wd={wd_keys}, model={model_keys}"
            )

    @invariant()
    def values_consistent(self) -> None:
        """Check that all keys have matching weights."""
        if self.model is not None:
            for key in self.model:
                wd_val = self.wd[key]
                model_val = self.model[key]
                assert abs(wd_val - model_val) < 1e-9, (
                    f"Value mismatch for {key!r}: wd={wd_val}, model={model_val}"
                )

    @rule(key=key_strategy, weight=weight_strategy)
    def set_item(self, key: str, weight: float) -> None:
        """Test setting a key's weight."""
        if self.model is None:
            return

        model_error = None
        try:
            self.model[key] = weight
        except ValueError as e:
            model_error = e

        wd_error = None
        try:
            self.wd[key] = weight
        except ValueError as e:
            wd_error = e

        if model_error is not None:
            assert wd_error is not None, "Model raised ValueError but wd succeeded"
        else:
            assert wd_error is None, "wd raised ValueError but model succeeded"

    @rule(key=key_strategy)
    def get_item(self, key: str) -> None:
        """Test getting a key's weight."""
        if self.model is None:
            return

        model_error = None
        model_result = None
        try:
            model_result = self.model[key]
        except KeyError as e:
            model_error = e

        wd_error = None
        wd_result = None
        try:
            wd_result = self.wd[key]
        except KeyError as e:
            wd_error = e

        if model_error is not None:
            assert wd_error is not None, (
                f"Model raised KeyError but wd returned {wd_result}"
            )
        else:
            assert wd_error is None, (
                f"wd raised KeyError but model returned {model_result}"
            )
            assert wd_result is not None and model_result is not None
            assert abs(wd_result - model_result) < 1e-9, (
                f"getitem mismatch: wd={wd_result}, model={model_result}"
            )

    @rule(key=key_strategy)
    def del_item(self, key: str) -> None:
        """Test deleting a key."""
        if self.model is None:
            return

        model_error = None
        try:
            del self.model[key]
        except KeyError as e:
            model_error = e

        wd_error = None
        try:
            del self.wd[key]
        except KeyError as e:
            wd_error = e

        if model_error is not None:
            assert wd_error is not None, "Model raised KeyError but wd succeeded"
        else:
            assert wd_error is None, "wd raised KeyError but model succeeded"

    @rule(key=key_strategy)
    def check_contains(self, key: str) -> None:
        """Test checking if a key exists."""
        if self.model is None:
            return
        assert (key in self.wd) == (key in self.model), f"contains mismatch for {key!r}"

    @rule(key=key_strategy, default=weight_strategy)
    def get_with_default(self, key: str, default: float) -> None:
        """Test get() with default value."""
        if self.model is None:
            return
        wd_result = self.wd.get(key, default)
        model_result = self.model.get(key, default)
        if wd_result is None or model_result is None:
            assert wd_result == model_result
        else:
            assert abs(wd_result - model_result) < 1e-9, (
                f"get mismatch: wd={wd_result}, model={model_result}"
            )

    @rule(key=key_strategy)
    def pop_item(self, key: str) -> None:
        """Test pop() method."""
        if self.model is None:
            return

        model_error = None
        model_result = None
        try:
            model_result = self.model.pop(key)
        except KeyError as e:
            model_error = e

        wd_error = None
        wd_result = None
        try:
            wd_result = self.wd.pop(key)
        except KeyError as e:
            wd_error = e

        if model_error is not None:
            assert wd_error is not None, (
                f"Model raised KeyError but wd returned {wd_result}"
            )
        else:
            assert wd_error is None, (
                f"wd raised KeyError but model returned {model_result}"
            )
            assert wd_result is not None and model_result is not None
            assert abs(wd_result - model_result) < 1e-9, (
                f"pop mismatch: wd={wd_result}, model={model_result}"
            )

    @rule(key=key_strategy, default=positive_weight_strategy)
    def setdefault_item(self, key: str, default: float) -> None:
        """Test setdefault() method."""
        if self.model is None:
            return

        model_error = None
        model_result = None
        try:
            model_result = self.model.setdefault(key, default)
        except ValueError as e:
            model_error = e

        wd_error = None
        wd_result = None
        try:
            wd_result = self.wd.setdefault(key, default)
        except ValueError as e:
            wd_error = e

        if model_error is not None:
            assert wd_error is not None, "Model raised ValueError but wd succeeded"
        else:
            assert wd_error is None, "wd raised ValueError but model succeeded"
            assert wd_result is not None and model_result is not None
            assert abs(wd_result - model_result) < 1e-9, (
                f"setdefault mismatch: wd={wd_result}, model={model_result}"
            )

    @rule()
    def clear_all(self) -> None:
        """Test clear() method."""
        if self.model is None:
            return
        self.model.clear()
        self.wd.clear()

    @rule()
    def check_iteration(self) -> None:
        """Test that iteration yields all keys."""
        if self.model is None:
            return
        wd_keys = list(self.wd)
        model_keys = list(self.model)
        # Order may differ due to swap-remove, so compare as sets
        assert set(wd_keys) == set(model_keys), (
            f"iteration mismatch: wd={set(wd_keys)}, model={set(model_keys)}"
        )

    @rule()
    def check_keys(self) -> None:
        """Test keys() method."""
        if self.model is None:
            return
        wd_keys = set(self.wd.keys())
        model_keys = set(self.model.keys())
        assert wd_keys == model_keys, f"keys mismatch: wd={wd_keys}, model={model_keys}"

    @rule()
    def check_values(self) -> None:
        """Test values() method returns correct weights."""
        if self.model is None:
            return
        # Compare sorted values since order may differ
        wd_values = sorted(self.wd.values())
        model_values = sorted(self.model.values())
        assert len(wd_values) == len(model_values)
        for wd_v, model_v in zip(wd_values, model_values, strict=True):
            assert abs(wd_v - model_v) < 1e-9, (
                f"values mismatch: wd={wd_v}, model={model_v}"
            )

    @rule()
    def check_items(self) -> None:
        """Test items() method."""
        if self.model is None:
            return
        wd_items = dict(self.wd.items())
        model_items = dict(self.model.items())
        assert wd_items.keys() == model_items.keys()
        for key in model_items:
            assert abs(wd_items[key] - model_items[key]) < 1e-9, (
                f"items mismatch for {key!r}: "
                f"wd={wd_items[key]}, model={model_items[key]}"
            )


# Configure and run the test
@pytest.mark.slow
class TestSamplerDictConformance(SamplerDictConformance.TestCase):  # pyright: ignore[reportUntypedBaseClass]
    """Stateful test class - slow due to comprehensive state exploration."""

    settings = settings(max_examples=100, stateful_step_count=50)
