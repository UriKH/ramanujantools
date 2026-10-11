import random

import pytest
import sympy as sp
from sympy.abc import n, x, y

from ramanujantools import Matrix, Position
from ramanujantools.flint_core import NumericMatrix
from ramanujantools.flint_core.numeric_matrix import (
    _StepEvaluator,
    _UnsupportedExpression,
)
from ramanujantools.matrix_benchmark import COMPANION_MATRICES


def test_conversion():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    for i in range(1, 10):
        assert m.subs({n: i}) == NumericMatrix.lambda_from_rt(m)({n: i}).to_rt()


def test_walk():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    assert (
        m.walk({n: 2}, 100, {n: 3})
        == NumericMatrix.walk(m, Position({n: 2}), 100, Position({n: 3})).to_rt()
    )


def test_walk_list():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    rt_walk_matrices = m.walk({n: 2}, list(range(1, 100)), {n: 3})
    numeric_walk_matrices = NumericMatrix.walk(
        m, Position({n: 2}), list(range(1, 100)), Position({n: 3})
    )
    for rt_walk_matrix, numeric_walk_matrix in zip(
        rt_walk_matrices, numeric_walk_matrices
    ):
        assert rt_walk_matrix == numeric_walk_matrix.to_rt()


@pytest.fixture
def assert_fast_walk_matches_reference(monkeypatch):
    """
    Returns a function that walks using `_StepEvaluator` (at any depth),
    and asserts that the result equals the reference walk, including raising
    `ZeroDivisionError` in exactly the same cases.
    """
    monkeypatch.setattr(NumericMatrix, "FAST_WALK_MIN_DEPTH", 0)

    def check(matrix, trajectory, iterations, start):
        trajectory, start = Position(trajectory), Position(start)
        _StepEvaluator(matrix, trajectory, start)  # The fast path must not fall back
        iterations_list = iterations if isinstance(iterations, list) else [iterations]
        try:
            expected = NumericMatrix._walk_reference(
                matrix, trajectory, iterations_list, start
            )
        except ZeroDivisionError:
            with pytest.raises(ZeroDivisionError):
                NumericMatrix.walk(matrix, trajectory, iterations, start)
            return
        actual = NumericMatrix.walk(matrix, trajectory, iterations, start)
        if not isinstance(iterations, list):
            actual = [actual]
        assert len(actual) == len(expected)
        for actual_matrix, expected_matrix in zip(actual, expected):
            assert isinstance(actual_matrix, NumericMatrix)
            assert actual_matrix == expected_matrix

    return check


@pytest.mark.parametrize("name", COMPANION_MATRICES.keys())
@pytest.mark.parametrize("start", [1, 0, -7])
def test_fast_walk_benchmark_matrices(assert_fast_walk_matches_reference, name, start):
    assert_fast_walk_matches_reference(
        COMPANION_MATRICES[name], {n: 1}, [1, 50, 120], {n: start}
    )


def _random_polynomial(rng, degree):
    return sp.Add(
        *[rng.randint(-(10**6), 10**6) * n**power for power in range(degree + 1)]
    )


def _random_rational_function(rng, degree):
    # Denominator roots are negative, so they are never hit by walks starting at n >= 0
    denominator = sp.Mul(
        *[(rng.randint(1, 5) * n + rng.randint(1, 50)) for _ in range(rng.randint(1, 4))]
    )
    return _random_polynomial(rng, degree) / denominator


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("rational", [False, True], ids=["polynomial", "rational"])
def test_fast_walk_random_matrices(assert_fast_walk_matches_reference, seed, rational):
    rng = random.Random(seed)
    dim = rng.randint(2, 5)
    degree = rng.randint(1, 40)
    entry = _random_rational_function if rational else _random_polynomial
    matrix = Matrix(dim, dim, [entry(rng, rng.randint(0, degree)) for _ in range(dim**2)])
    assert_fast_walk_matches_reference(
        matrix, {n: rng.randint(1, 3)}, [1, 37, 64], {n: rng.randint(0, 5)}
    )


@pytest.mark.parametrize(
    "trajectory, start",
    [({n: 1}, {n: 1}), ({n: 3}, {n: 2}), ({n: 2}, {n: 7})],
)
def test_fast_walk_denominators(assert_fast_walk_matches_reference, trajectory, start):
    matrix = Matrix([[1 / (n + 2), n**2 - 3], [sp.Rational(5, 7), 1 / n + n / 3]])
    assert_fast_walk_matches_reference(matrix, trajectory, 60, start)


def test_fast_walk_negative_start_and_trajectory(assert_fast_walk_matches_reference):
    matrix = Matrix([[0, -(n**2) - 1], [1, 2 * n + 1]])
    assert_fast_walk_matches_reference(matrix, {n: -1}, 60, {n: 0})
    assert_fast_walk_matches_reference(matrix, {n: 2}, 60, {n: -40})


def test_fast_walk_multivariate(assert_fast_walk_matches_reference):
    matrix = Matrix(
        [[x * y + 1, (x - y) / (x + y + 3)], [x**2 / 2, (y + 1) ** 3 / (x + 1)]]
    )
    assert_fast_walk_matches_reference(matrix, {x: 1, y: 2}, 60, {x: 1, y: 0})
    assert_fast_walk_matches_reference(
        matrix, {x: 2, y: -1}, [5, 33], {x: 3, y: 100}
    )


def test_fast_walk_extra_start_symbols(assert_fast_walk_matches_reference):
    matrix = Matrix([[n, 1], [1, n + 1]])
    assert_fast_walk_matches_reference(matrix, {n: 1, x: 2}, 40, {n: 1, x: 5})


def test_fast_walk_constant_matrix(assert_fast_walk_matches_reference):
    matrix = Matrix([[1, sp.Rational(1, 2)], [3, 0]])
    assert_fast_walk_matches_reference(matrix, {n: 1}, 40, {n: 1})


@pytest.mark.parametrize(
    "iterations",
    [40, [40], [0], [1], [0, 1], [1, 2, 3], list(range(1, 200)), list(range(0, 200, 7))],
)
def test_fast_walk_iterations(assert_fast_walk_matches_reference, iterations):
    matrix = Matrix([[0, -(n**3)], [1, n**2 + 1 / (n + 1)]])
    assert_fast_walk_matches_reference(matrix, {n: 1}, iterations, {n: 1})


def test_fast_walk_expression_shapes(assert_fast_walk_matches_reference):
    expressions = [
        sp.Add(n, sp.Mul(2, sp.Pow(n + 1, -2), evaluate=False), evaluate=False),
        sp.Mul(sp.Rational(-3, 4), n + 2, sp.Pow(n**2 + 1, -1), evaluate=False),
        sp.Pow(sp.Add(1, sp.Pow(n, -1), evaluate=False), -2, evaluate=False),
        sp.Pow(sp.Rational(2, 3), -3, evaluate=False),
        (n**2 - 1) / (n - 1),
        1 / n - 1 / (n + 1) + n / (n + 1),
        sp.Integer(0),
    ]
    for numerator in expressions:
        for denominator in [1, n + 5]:
            matrix = Matrix([[numerator / denominator, 1], [n, numerator]])
            assert_fast_walk_matches_reference(matrix, {n: 1}, [1, 40], {n: 2})


@pytest.mark.parametrize(
    "entry",
    [
        1 / n,
        1 / (n + 2),
        sp.Pow(n**2 - 1, 1) * sp.Pow(n - 1, -1),  # cancellable
        1 / (1 + 1 / n),  # nested denominator
        n * (n + 1) / (n + 3) ** 2,  # singular at the first step
    ],
)
@pytest.mark.parametrize("fast", [False, True], ids=["reference", "fast"])
def test_walk_singular_step(monkeypatch, entry, fast):
    if fast:
        monkeypatch.setattr(NumericMatrix, "FAST_WALK_MIN_DEPTH", 0)
    matrix = Matrix([[entry, 1], [1, n]])
    trajectory, start = Position({n: 1}), Position({n: -3})
    if fast:
        _StepEvaluator(matrix, trajectory, start)
    singular = next(step for step in range(10) if _is_singular(entry, -3 + step))

    def walk(iterations):
        return NumericMatrix.walk(matrix, trajectory, iterations, start)

    walk(singular)  # Stopping just before the singular step does not raise
    walk([0, singular])
    with pytest.raises(ZeroDivisionError):
        walk(singular + 1)
    with pytest.raises(ZeroDivisionError):
        walk([singular, singular + 1, singular + 40])


def _is_singular(entry, value):
    try:
        NumericMatrix.lambda_from_rt(Matrix([[entry]]))(Position({n: value}))
    except ZeroDivisionError:
        return True
    return False


def test_cancellable_singular_step_raises_like_reference():
    matrix = Matrix([[(n**2 - 1) / (n - 1), 0], [0, 1]])
    for walk in [NumericMatrix.walk, _fast_walk_without_threshold]:
        walk(matrix, Position({n: 1}), 1, Position({n: 0}))
        with pytest.raises(ZeroDivisionError):
            walk(matrix, Position({n: 1}), 2, Position({n: 0}))


def _fast_walk_without_threshold(matrix, trajectory, iterations, start):
    evaluator = _StepEvaluator(matrix, trajectory, start)
    steps = [evaluator(step) for step in range(iterations)]
    return NumericMatrix._product_checkpoints(steps, [iterations], matrix.rows)[0]


@pytest.mark.parametrize(
    "entry",
    [n ** sp.Rational(1, 2), n**n, sp.Float(1.5) * n, sp.sin(n)],
)
def test_unsupported_expressions_fall_back(entry):
    matrix = Matrix([[entry, 1], [1, n]])
    with pytest.raises(_UnsupportedExpression):
        _StepEvaluator(matrix, Position({n: 1}), Position({n: 1}))
    try:
        expected = NumericMatrix._walk_reference(
            matrix, Position({n: 1}), [40], Position({n: 1})
        )
    except Exception as error:
        with pytest.raises(type(error)):
            NumericMatrix.walk(matrix, Position({n: 1}), 40, Position({n: 1}))
        return
    assert [NumericMatrix.walk(matrix, Position({n: 1}), 40, Position({n: 1}))] == expected


def test_unsupported_start_falls_back():
    matrix = Matrix([[n, 1], [1, n + y]])  # y is missing from start
    with pytest.raises(_UnsupportedExpression):
        _StepEvaluator(matrix, Position({n: 1}), Position({n: 1}))
    with pytest.raises(KeyError):
        NumericMatrix.walk(matrix, Position({n: 1}), 40, Position({n: 1}))


@pytest.mark.parametrize(
    "trajectory, start",
    [
        ({n: 1}, {n: sp.Integer(1)}),
        ({n: sp.Integer(1)}, {n: 1}),
        ({n: 1}, {n: sp.Rational(1, 2)}),
    ],
)
def test_non_int_positions_fall_back(trajectory, start):
    matrix = Matrix([[0, -(n**2)], [1, n + 1]])
    with pytest.raises(_UnsupportedExpression):
        _StepEvaluator(matrix, Position(trajectory), Position(start))
    with pytest.raises(TypeError):
        NumericMatrix.walk(matrix, Position(trajectory), 40, Position(start))


def test_short_walks_use_reference(monkeypatch):
    def fail(*args):
        raise AssertionError("Short walks should not convert the matrix")

    monkeypatch.setattr(_StepEvaluator, "__init__", fail)
    matrix = Matrix([[0, -(n**2)], [1, n + 1]])
    depth = NumericMatrix.FAST_WALK_MIN_DEPTH - 1
    assert NumericMatrix.walk(
        matrix, Position({n: 1}), depth, Position({n: 1})
    ).to_rt() == matrix.walk({n: 1}, depth, {n: 1})
