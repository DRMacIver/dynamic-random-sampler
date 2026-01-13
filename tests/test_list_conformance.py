"""Hypothesis stateful test for list API conformance.

This test verifies that DynamicSampler behaves like a list of weights,
with stable indices (no index-shifting operations).
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


class WeightListModel:
    """Reference model: a plain Python list of weights.

    This is just a list with validation that weights are positive.
    Uses stable indices like DynamicSampler - only append/pop are allowed.
    """

    def __init__(self, weights: list[float]) -> None:
        self.weights = list(weights)

    def __len__(self) -> int:
        return len(self.weights)

    def __getitem__(self, index: int) -> float:
        return self.weights[index]

    def __setitem__(self, index: int, weight: float) -> None:
        self.weights[index] = weight

    def __contains__(self, weight: float) -> bool:
        return any(abs(w - weight) < 1e-10 for w in self.weights)

    def __iter__(self):
        return iter(self.weights)

    def append(self, weight: float) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        self.weights.append(weight)

    def extend(self, weights: list[float]) -> None:
        for w in weights:
            self.append(w)

    def pop(self) -> float:
        if not self.weights:
            raise IndexError("pop from empty list")
        return self.weights.pop()

    def clear(self) -> None:
        self.weights.clear()

    def index(self, weight: float) -> int:
        for i, w in enumerate(self.weights):
            if abs(w - weight) < 1e-10:
                return i
        raise ValueError(f"{weight} is not in list")

    def count(self, weight: float) -> int:
        return sum(1 for w in self.weights if abs(w - weight) < 1e-10)



class SamplerListConformance(RuleBasedStateMachine):
    """Stateful test comparing DynamicSampler to WeightListModel."""

    def __init__(self) -> None:
        super().__init__()
        self.model: WeightListModel | None = None
        self.sampler: Any = None

    @initialize(
        weights=st.lists(
            st.floats(min_value=0.1, max_value=100.0), min_size=1, max_size=10
        )
    )
    def init_sampler(self, weights: list[float]) -> None:
        from dynamic_random_sampler import DynamicSampler
        self.model = WeightListModel(weights)
        self.sampler = DynamicSampler(weights)

    @invariant()
    def lengths_match(self) -> None:
        if self.model is not None:
            assert len(self.sampler) == len(self.model), \
                f"Length mismatch: sampler={len(self.sampler)}, model={len(self.model)}"

    @invariant()
    def weights_match(self) -> None:
        if self.model is not None:
            sampler_weights = list(self.sampler)
            model_weights = list(self.model)
            assert len(sampler_weights) == len(model_weights), (
                f"Weight list lengths differ: {len(sampler_weights)} "
                f"vs {len(model_weights)}"
            )
            zipped = zip(sampler_weights, model_weights, strict=True)
            for i, (sw, mw) in enumerate(zipped):
                assert abs(sw - mw) < 1e-9, (
                    f"Weight mismatch at index {i}: sampler={sw}, model={mw}"
                )

    @rule(index=st.integers(min_value=-20, max_value=20))
    def get_item(self, index: int) -> None:
        if self.model is None:
            return

        model_error = None
        model_result = None
        try:
            model_result = self.model[index]
        except IndexError as e:
            model_error = e

        sampler_error = None
        sampler_result = None
        try:
            sampler_result = self.sampler[index]
        except IndexError as e:
            sampler_error = e

        # Both should error or both should succeed
        if model_error is not None:
            assert sampler_error is not None, \
                f"Model raised IndexError but sampler returned {sampler_result}"
        else:
            assert sampler_error is None, (
                f"Sampler raised IndexError but model returned {model_result}"
            )
            assert sampler_result is not None and model_result is not None
            assert abs(sampler_result - model_result) < 1e-9, (
                f"getitem mismatch: sampler={sampler_result}, model={model_result}"
            )

    @rule(index=st.integers(min_value=-20, max_value=20),
          weight=st.floats(min_value=0.0, max_value=100.0))
    def set_item(self, index: int, weight: float) -> None:
        if self.model is None:
            return

        model_error = None
        try:
            self.model[index] = weight
        except IndexError as e:
            model_error = e

        sampler_error = None
        try:
            self.sampler[index] = weight
        except IndexError as e:
            sampler_error = e

        if model_error is not None:
            assert sampler_error is not None, \
                "Model raised IndexError but sampler succeeded"
        else:
            assert sampler_error is None, \
                "Sampler raised IndexError but model succeeded"

    @rule(weight=st.floats(min_value=0.1, max_value=100.0))
    def append_weight(self, weight: float) -> None:
        if self.model is None:
            return
        self.model.append(weight)
        self.sampler.append(weight)

    @rule(
        weights=st.lists(
            st.floats(min_value=0.1, max_value=100.0), min_size=0, max_size=5
        )
    )
    def extend_weights(self, weights: list[float]) -> None:
        if self.model is None:
            return
        self.model.extend(weights)
        self.sampler.extend(weights)

    @rule()
    def pop_weight(self) -> None:
        if self.model is None:
            return

        model_error = None
        model_result = None
        try:
            model_result = self.model.pop()
        except IndexError as e:
            model_error = e

        sampler_error = None
        sampler_result = None
        try:
            sampler_result = self.sampler.pop()
        except IndexError as e:
            sampler_error = e

        if model_error is not None:
            assert sampler_error is not None, (
                f"Model raised IndexError on pop but sampler returned {sampler_result}"
            )
        else:
            assert sampler_error is None, (
                "Sampler raised IndexError on pop but model succeeded"
            )
            assert sampler_result is not None and model_result is not None
            assert abs(sampler_result - model_result) < 1e-9, (
                f"pop result mismatch: sampler={sampler_result}, model={model_result}"
            )

    @rule()
    def clear_all(self) -> None:
        if self.model is None:
            return
        self.model.clear()
        self.sampler.clear()

    @rule(weight=st.floats(min_value=0.1, max_value=100.0))
    def check_contains(self, weight: float) -> None:
        if self.model is None:
            return
        model_contains = weight in self.model
        sampler_contains = weight in self.sampler
        assert model_contains == sampler_contains, (
            f"contains mismatch for {weight}: "
            f"model={model_contains}, sampler={sampler_contains}"
        )

    @rule(weight=st.floats(min_value=0.1, max_value=100.0))
    def find_index(self, weight: float) -> None:
        if self.model is None:
            return

        model_error = None
        model_result = None
        try:
            model_result = self.model.index(weight)
        except ValueError as e:
            model_error = e

        sampler_error = None
        sampler_result = None
        try:
            sampler_result = self.sampler.index(weight)
        except ValueError as e:
            sampler_error = e

        if model_error is not None:
            assert sampler_error is not None, (
                "Model raised ValueError on index "
                f"but sampler returned {sampler_result}"
            )
        else:
            assert sampler_error is None, (
                "Sampler raised ValueError on index but model succeeded"
            )
            assert sampler_result == model_result, \
                f"index result mismatch: sampler={sampler_result}, model={model_result}"

    @rule(weight=st.floats(min_value=0.1, max_value=100.0))
    def count_weight(self, weight: float) -> None:
        if self.model is None:
            return
        model_count = self.model.count(weight)
        sampler_count = self.sampler.count(weight)
        assert model_count == sampler_count, \
            f"count mismatch for {weight}: model={model_count}, sampler={sampler_count}"

    @rule()
    def check_list_conversion(self) -> None:
        if self.model is None:
            return
        model_list = list(self.model)
        sampler_list = list(self.sampler)
        assert len(model_list) == len(sampler_list), (
            f"list() length mismatch: {len(model_list)} vs {len(sampler_list)}"
        )
        for i, (mw, sw) in enumerate(zip(model_list, sampler_list, strict=True)):
            assert abs(mw - sw) < 1e-9, (
                f"to_list mismatch at {i}: model={mw}, sampler={sw}"
            )


# Configure and run the test
@pytest.mark.slow
class TestSamplerListConformance(SamplerListConformance.TestCase):  # pyright: ignore[reportUntypedBaseClass]
    """Stateful test class - slow due to comprehensive state exploration."""

    settings = settings(max_examples=100, stateful_step_count=50)
