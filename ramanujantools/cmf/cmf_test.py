import pytest
from pytest import raises

import sympy as sp
from sympy.abc import a, b, c, x, y, n

from ramanujantools import Position, Matrix, simplify
from ramanujantools.cmf import CMF, known_cmfs, pFq
from ramanujantools.flint_core import NumericMatrix
from ramanujantools.utils import batched, Batchable

a0, a1 = sp.symbols("a:2")
b0, b1 = sp.symbols("b:2")
c0, c1, c2, c3 = sp.symbols("c:4")
x0, x1, x2, x3 = sp.symbols("x:4")
y0, y1, y2, y3 = sp.symbols("y:4")


def test_symbols():
    cmf = known_cmfs.cmf1()
    expected_axes = {x, y}
    expected_parameters = {c0, c1, c2, c3}
    assert expected_axes == cmf.axes()
    assert expected_parameters == cmf.parameters()
    assert set().union(expected_axes, expected_parameters) == cmf.free_symbols()


def test_rank():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    assert 2 == cmf.rank()


def test_dim():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    assert 3 == cmf.dim()


def test_validate_conserving():
    m = Matrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    cmf = CMF(matrices={x: m, y: m}, validate=False)
    with raises(ValueError):
        cmf.validate_conserving()
    with raises(ValueError):
        cmf = CMF(matrices={x: m, y: m}, validate=True)


def test_validate_negative_cache():
    cmf = known_cmfs.cmf1()
    negative_matrices = {axis: cmf.M(axis, sign=False) for axis in cmf.matrices}
    CMF(cmf.matrices, _negative_matrices_cache=negative_matrices, validate=True)
    random_axis = list(negative_matrices.keys())[0]
    negative_matrices[random_axis] += Matrix.eye(cmf.rank())
    with raises(ValueError):
        CMF(cmf.matrices, _negative_matrices_cache=negative_matrices, validate=True)


def test_subs():
    cmf = known_cmfs.var_root_cmf()
    substitution = Position({c0: 1, c1: 2 * c1 - 3})
    assert cmf.subs(substitution) == CMF(
        {axis: matrix.subs(substitution) for axis, matrix in cmf.matrices.items()}
    )


def test_subs_axes_throw():
    cmf = known_cmfs.e()
    substitution = Position({x: 2})
    with raises(ValueError):
        cmf.subs(substitution)


def test_subs_non_linear_shift_throws():
    cmf = known_cmfs.e()
    substitution = Position({x: x**2})
    with raises(ValueError):
        cmf.subs(substitution)


def test_dual_conserving():
    cmf = known_cmfs.cmf2()
    dual_cmf = cmf.dual()
    dual_cmf.validate_conserving()


def test_dual_inverible():
    cmf = known_cmfs.cmf1()
    assert cmf == cmf.dual().dual()


def test_coboundary_eye():
    cmf = known_cmfs.e()
    assert cmf == cmf.coboundary(Matrix.eye(cmf.rank()))


def test_coboundary_inversible():
    cmf = known_cmfs.e()
    U = Matrix([[x, c0 * x], [y, x * y]])
    assert cmf == cmf.coboundary(U).coboundary(U.inverse())


def test_coboundary_is_conserving():
    cmf = pFq(3, 2, 1)
    x0 = sp.Symbol("x0")
    y0 = sp.Symbol("y0")
    U = Matrix([[x0, c0 * y0], [3 * x0 - 2 * y0, 17 * x0**2]])
    co_cmf = cmf.coboundary(U)
    co_cmf.validate_conserving()
    co_cmf.validate_negative_cache()


def test_sub_cmf_zero_trajectory_throws():
    cmf = pFq(2, 1, -1)
    with raises(ValueError):
        cmf.sub_cmf({a0: {x0: 0, x1: 0, y0: 0}})


def test_sub_cmf_missing_axes_throws():
    cmf = pFq(2, 1, -1)
    with raises(ValueError):
        cmf.sub_cmf({a0: {x0: 1, y0: 0}})


def test_sub_cmf_wrong_axes_throws():
    cmf = pFq(2, 1, -1)
    with raises(ValueError):
        cmf.sub_cmf({a0: {x3: 1, y3: 0}})


def test_sub_cmf_linear_dependent_basis_throws():
    cmf = pFq(2, 1, -1)
    with raises(ValueError):
        cmf.sub_cmf(
            {
                a0: {x0: 1, x1: 0, y0: 0},
                a1: {x0: 1, x1: 0, y0: 0},
            }
        )


def test_sub_cmf_trivial():
    cmf = pFq(2, 1, -1)
    basis = {
        a0: {x0: 1, x1: 0, y0: 0},
        a1: {x0: 0, x1: 1, y0: 0},
        b0: {x0: 0, x1: 0, y0: 1},
    }
    expected = CMF(
        matrices={
            a0: cmf.M(x0).subs({x0: a0, x1: a1, y0: b0}),
            a1: cmf.M(x1).subs({x0: a0, x1: a1, y0: b0}),
            b0: cmf.M(y0).subs({x0: a0, x1: a1, y0: b0}),
        }
    )
    assert expected == cmf.sub_cmf(basis)


def test_sub_cmf_conserving():
    cmf = pFq(3, 3, -1)
    basis = {
        a0: {x0: 0, x1: 1, x2: 0, y0: 0, y1: 1, y2: -1},
        a1: {x0: 1, x1: 0, x2: 1, y0: 0, y1: 0, y2: 0},
        b0: {x0: 0, x1: 0, x2: 0, y0: 0, y1: 1, y2: 1},
        b1: {x0: 1, x1: 0, x2: 0, y0: 1, y1: 0, y2: 0},
    }
    sub_cmf = cmf.sub_cmf(basis)
    sub_cmf.validate_conserving()


def test_trajectory_matrix_axis():
    cmf = known_cmfs.e()
    assert cmf.trajectory_matrix({x: 3, y: 0}, {x: x, y: y}).walk(
        {n: 1}, 1, {n: 0}
    ) == simplify(cmf.M(x).walk({x: 1, y: 0}, 3, {x: x, y: y}))
    assert cmf.trajectory_matrix({x: 0, y: 2}, {x: x, y: y}).walk(
        {n: 1}, 1, {n: 0}
    ) == simplify(cmf.M(y).walk({x: 0, y: 1}, 2, {x: x, y: y}))


def test_trajectory_matrix_diagonal():
    cmf = known_cmfs.e()
    trajectory = Position({x: 1, y: 1})
    start = Position({x: x, y: y})
    assert cmf.trajectory_matrix(trajectory, start) == simplify(
        (cmf.M(x) * cmf.M(y)({x: x + 1})).subs(
            cmf.trajectory_substitution(trajectory, start, n)
        )
    )


def test_back_negates_forward():
    cmf = known_cmfs.e()
    assert Matrix.eye(2) == (cmf.M(x, True) * cmf.M(x, False).subs({x: x + 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(x, False) * cmf.M(x, True).subs({x: x - 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(y, True) * cmf.M(y, False).subs({y: y + 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(y, False) * cmf.M(y, True).subs({y: y - 1})).reduce()


def test_trajectory_matrix_negative_axis():
    cmf = known_cmfs.e()
    start = {x: x, y: y}
    assert cmf.trajectory_matrix({x: -3, y: 0}, start).equal_projectively(
        cmf.M(x, False).walk(
            {x: -1, y: 0}, 3, cmf.trajectory_substitution({x: -3, y: 0}, start, n)
        )
    )
    assert cmf.trajectory_matrix({x: 0, y: -2}, start).equal_projectively(
        cmf.M(y, False).walk(
            {x: 0, y: -1}, 2, cmf.trajectory_substitution({x: 0, y: -2}, start, n)
        )
    )


def test_trajectory_matrix_negative():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    expected = (
        cmf.M(a, sign=True)
        * cmf.M(b, sign=False).subs({a: a + 1})
        * cmf.M(b, sign=False).subs({a: a + 1, b: b - 1})
        * cmf.M(c, sign=False).subs({a: a + 1, b: b - 2})
    )
    actual = cmf.trajectory_matrix({a: 1, b: -2, c: -1}, {a: a, b: b, c: c})
    assert actual.walk({n: 1}, 1, {n: 0}).equal_projectively(expected)


def test_trajectory_matrix_rational():
    cmf = known_cmfs.e()
    start = Position({x: sp.Rational(2, 3), y: sp.Rational(1, 4)})
    trajectory = {x: 1, y: 1}
    symbolic_start = cmf.trajectory_substitution(trajectory, start, n)
    expected = (
        cmf.M(x).subs(symbolic_start) * cmf.M(y).subs(symbolic_start + Position({x: 1}))
    ).factor()
    actual = cmf.trajectory_matrix({x: 1, y: 1}, start)
    assert expected == actual


def test_walk_axis():
    cmf = known_cmfs.e()
    start = {x: 1, y: 1}
    assert cmf.walk({x: 1, y: 0}, 17, start) == cmf.M(x).walk({x: 1, y: 0}, 17, start)
    assert cmf.walk({x: 0, y: 1}, 17, start) == cmf.M(y).walk({x: 0, y: 1}, 17, start)


def test_walk_axis_multiplicity():
    cmf = known_cmfs.e()
    start = {x: 1, y: 1}
    trajectory = Position({x: 1, y: 0})
    depth = 4
    assert cmf.walk(depth * trajectory, 1, start) == cmf.M(x).walk(
        trajectory, depth, start
    )


def test_walk_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 1}
    start = {x: 1, y: 1}
    Mxy = cmf.trajectory_matrix(trajectory, start)
    assert cmf.walk(trajectory, 9, start) == Mxy.walk({n: 1}, 9, {n: 0})


def test_limit_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 1}
    start = {x: 1, y: 1}
    Mxy = cmf.walk(trajectory, 1, {x: x, y: y})
    assert cmf.limit(trajectory, 17, start) == Mxy.limit(trajectory, 17, start)


def test_limit_vectors():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 3}
    depths = [12, 13, 17]
    start = {x: 2, y: 1}
    initial_values = Matrix([[1, 2, 3], [4, 5, 6]])
    final_projection = Matrix([[1, 2], [3, 4], [5, 6]])
    expected = cmf.limit(trajectory, depths, start, initial_values, final_projection)
    for lim in expected:
        lim.initial_values = initial_values
        lim.final_projection = final_projection
    assert expected == cmf.limit(
        trajectory, depths, start, initial_values, final_projection
    )


def test_walk_list():
    cmf = known_cmfs.e()
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29]
    assert cmf.walk(trajectory, iterations, start) == [
        cmf.walk(trajectory, i, start) for i in iterations
    ]


def test_trajectory_substitution_throws_on_collision():
    cmf = known_cmfs.e()
    trajectory = Position({x: 1, y: 0})
    start = Position({x: n, y: 0})
    with raises(ValueError):
        cmf.trajectory_substitution(trajectory, start, n)
    assert {x: n + a, y: 0} == cmf.trajectory_substitution(trajectory, start, a)


def test_trajectory_substitution_axis():
    cmf = known_cmfs.e()
    x_axis = {x: 1, y: 0}
    y_axis = {x: 0, y: 1}
    origin = {x: 0, y: 0}
    assert {x: n, y: 0} == cmf.trajectory_substitution(x_axis, origin, n)
    assert {x: 0, y: n} == cmf.trajectory_substitution(y_axis, origin, n)


def test_trajectory_substitution_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 2}
    start = {x: 3, y: 5}
    assert {x: 3 + n, y: 5 + 2 * n} == cmf.trajectory_substitution(trajectory, start, n)


def test_trajectory_matrix_walk_equivalence():
    cmf = known_cmfs.e()
    iterations = 7
    trajectory = {x: 1, y: 1}
    start = {x: 3, y: 5}
    trajectory_matrix = cmf.trajectory_matrix(trajectory, start)
    assert trajectory_matrix.walk({n: 1}, iterations, {n: 0}) == cmf.walk(
        trajectory, iterations, start
    )


def test_work_non_integer_trajectory_throws():
    cmf = known_cmfs.e()
    with raises(ValueError):
        cmf.work(Position({x: 1, y: 1}), Position({x: 2, y: sp.Rational(5, 2)}))
    with raises(ValueError):
        cmf.work(Position({x: 1, y: n}), Position({x: 1, y: 2 * n}))


def test_work_numeric():
    cmf = known_cmfs.e()
    start = Position({x: 1, y: -1})
    end = Position({x: 17, y: -13})
    trajectory = end - start
    expected = cmf.walk(trajectory, 1, start)
    actual = cmf.work(start, end)
    assert expected == actual


def test_work_symbolic():
    cmf = known_cmfs.e()
    start = Position({x: 1 + x, y: -1 + 7 * y / 2})
    end = Position({x: 17 + x, y: -13 + 7 * y / 2})
    trajectory = end - start
    expected = cmf.walk(trajectory, 1, start)
    actual = cmf.work(start, end)
    assert expected == actual


def test_delta_correctness():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    depth = 100
    trajectory = {a: 1, b: 0, c: 1}
    start = {a: 1, b: 1, c: 1}
    l1, l2 = cmf.limit(trajectory, [depth, 2 * depth], start)
    assert cmf.delta(trajectory, depth, start) == l1.delta(l2.as_float())
    assert cmf.delta(trajectory, depth, start, limit=l2.as_float()) == l1.delta(
        l2.as_float()
    )


def test_blind_delta_sequence_agrees_with_blind_delta():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    depth = 30
    trajectory = {a: 1, b: 2, c: 0}
    start = {a: 2, b: 3, c: 4}
    limit = cmf.limit(trajectory, 2 * depth, start).as_float()
    delta_sequence = cmf.delta_sequence(trajectory, depth, start)
    sequence_of_deltas = [
        cmf.delta(trajectory, i, start, limit=limit) for i in range(1, depth + 1)
    ]
    assert delta_sequence == sequence_of_deltas
    assert len(delta_sequence) == depth


def test_pickle_state():
    import pickle
    cmf = pFq(2, 1, sp.Rational(1, 2))
    x0 = sp.symbols('x0')
    _ = cmf.M(x0, sign=False)

    dumped = pickle.dumps(cmf)
    loaded_cmf = pickle.loads(dumped)

    assert hasattr(loaded_cmf, '_negative_matrices_cache'), \
        "Unpickled object is missing '_negative_matrices_cache' attribute"
    try:
        _ = loaded_cmf.M(x0, sign=False)
    except AttributeError as e:
        assert False, f"Failed to access negative matrix after unpickling: {e}"


@batched("iterations")
def _reference_numeric_walk(
    matrix, trajectory, iterations: Batchable[int], start
) -> Batchable[NumericMatrix]:
    return NumericMatrix._walk_reference(matrix, trajectory, iterations, start)


@pytest.fixture
def compare_to_reference_walk(monkeypatch):
    """
    Returns a function that runs `compute()` with the fast numeric walk (at any depth),
    then again with the reference numeric walk, and asserts the results are equal,
    or that both raise the same exception type.
    """

    def run(compute):
        Matrix._walk_inner.cache_clear()
        try:
            return compute(), None
        except Exception as error:
            return None, type(error)
        finally:
            Matrix._walk_inner.cache_clear()

    def compare(compute):
        monkeypatch.setattr(NumericMatrix, "FAST_WALK_MIN_DEPTH", 0)
        actual, actual_error = run(compute)
        with monkeypatch.context() as patch:
            patch.setattr(NumericMatrix, "walk", staticmethod(_reference_numeric_walk))
            expected, expected_error = run(compute)
        assert actual_error == expected_error
        assert actual == expected
        return actual

    return compare


@pytest.mark.parametrize(
    "cmf, start, trajectory",
    [
        (pFq(2, 1, -1), [1, 1, 1], [40, 40, 40]),
        (pFq(2, 1, -1), [1, 2, 3], [37, -41, 50]),
        (pFq(2, 1, -1), [1, 2, 50], [33, -36, 40]),
        (pFq(3, 2, 1), [1, 1, 1, 2, 2], [33, 34, 35, 36, 37]),
        (pFq(3, 2, 1), [1, 1, 2, 60, 61], [30, 31, 32, 40, 41]),
        (pFq(3, 2, 1), [1, 2, 3, 70, 70], [33, -1, 35, 50, 45]),
        (pFq(4, 3, 1), [1, 1, 2, 2, 80, 81, 82], [30, 31, 32, 33, 40, 41, 42]),
        (pFq(4, 3, 1), [2, 1, 1, 1, 90, 91, 92], [35, -2, 34, 30, 50, 40, 45]),
        # Singular trajectories: both walks must raise ZeroDivisionError
        (pFq(2, 1, -1), [-2, 1, 2], [3, 45, 70]),
        (pFq(3, 2, 1), [2, 1, 1, 3, 2], [-35, 40, 44, 0, 52]),
        (pFq(4, 3, 1), [1, 1, 2, 2, 3, 3, 4], [33, 34, 35, 36, 37, 38, 39]),
        (pFq(4, 3, 1), [1, 1, 1, 1, 2, 2, 3], [-33, 40, 41, 42, 0, 1, 45]),
    ],
)
def test_work_numeric_fast_walk(compare_to_reference_walk, cmf, start, trajectory):
    axes = sorted(cmf.axes(), key=str)
    start = Position(dict(zip(axes, start)))
    end = start + Position(dict(zip(axes, trajectory)))
    compare_to_reference_walk(lambda: cmf.work(start, end))


@pytest.mark.parametrize(
    "cmf, start, trajectory",
    [
        (pFq(2, 2, -1), [1, 1, -1, -1], [1, 1, 0, -1]),
        (pFq(2, 2, -1), [1, 1, -1, 1], [12, 13, -14, 20]),
        (pFq(3, 2, 1), [2, 2, 2, 4, 4], [5, 5, 5, 10, 10]),
        (pFq(3, 2, 1), [2, 2, 2, 4, 4], [5, 6, 5, 10, 11]),
        (pFq(4, 3, 1), [1, 1, 2, 2, 3, 3, 4], [1, 1, 2, 2, 3, 3, 4]),
        (pFq(4, 3, 1), [-1, 1, 2, -2, 3, 7, 4], [-1, 1, 2, -2, 3, 7, 4]),
        (pFq(4, 3, 1), [1, 1, 1, 1, 2, 2, 3], [0, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_walk_fast_walk_benchmark_trajectories(
    compare_to_reference_walk, cmf, start, trajectory
):
    axes = sorted(cmf.axes(), key=str)
    start = Position(dict(zip(axes, start)))
    trajectory = Position(dict(zip(axes, trajectory)))
    compare_to_reference_walk(lambda: cmf.walk(trajectory, [1, 50, 100], start))


def test_walk_fast_walk_simple_trajectory(compare_to_reference_walk):
    cmf = known_cmfs.e()
    compare_to_reference_walk(
        lambda: cmf.walk({x: 1, y: 1}, [1, 50, 100], {x: 1, y: 1})
    )


def test_walk_fast_walk_deep_4f3(compare_to_reference_walk):
    cmf = pFq(4, 3, 1)
    axes = pFq.x_axes(4) + pFq.y_axes(3)
    trajectory = Position(dict(zip(axes, [-31, 64, 64, 64, 0, 0, 56])))
    start = Position(dict(zip(axes, [-4, 2, 2, 2, 1, -1, 1])))
    matrix = cmf.trajectory_matrix(trajectory, start)
    (symbol,) = matrix.free_symbols
    walks = compare_to_reference_walk(
        lambda: matrix.walk({symbol: 1}, [900, 950, 1000], {symbol: 0})
    )
    Matrix._walk_inner.cache_clear()
    assert matrix.walk({symbol: 1}, 1000, {symbol: 0}) == walks[-1]


# Real pFq trajectories, in the style of Ramanujan's Dreams: (p, q, z, trajectory, start)
REAL_TRAJECTORIES = {
    "2F1(-1)_small": (2, 1, -1, [1, 1, 1], [1, 1, 2]),
    "2F1(-1)_skew": (2, 1, -1, [3, -2, 5], [2, 1, 3]),
    "2F1(-1)_long": (2, 1, -1, [12, -9, 17], [1, 2, 3]),
    "3F2(1)_diagonal": (3, 2, 1, [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]),
    "3F2(1)_skew": (3, 2, 1, [5, 6, 5, 10, 11], [2, 2, 2, 4, 4]),
    "3F2(-1)_long": (3, 2, -1, [-9, 18, 18, 0, 16], [-3, 2, 2, 1, 1]),
    "4F3(1)_diagonal": (4, 3, 1, [1, 1, 2, 2, 3, 3, 4], [1, 1, 2, 2, 3, 3, 4]),
    "4F3(1)_medium": (4, 3, 1, [-12, 24, 24, 24, 0, 0, 20], [-4, 2, 2, 2, 1, -1, 1]),
    "5F4(-1)_small": (5, 4, -1, [1, 2, 1, 2, 1, 3, 2, 3, 2], [1, 1, 1, 1, 1, 2, 2, 2, 2]),
    "5F4(-1)_medium": (
        5,
        4,
        -1,
        [-14, 28, 28, 28, 28, 0, 0, 0, 24],
        [-4, 2, 2, 2, 2, 1, -1, 1, 1],
    ),
}

# Trajectories whose trajectory matrix first hits a pole at `step`: (p, q, z, trajectory, start, step)
SINGULAR_TRAJECTORIES = {
    "2F1(-1)": (2, 1, -1, [1, -2, 2], [4, -6, -5], 9),
    "3F2(1)": (3, 2, 1, [-4, -2, 1, 1, -3], [4, 1, -5, 0, -6], 5),
    "4F3(1)": (4, 3, 1, [0, 5, -3, 3, -4, -1, -3], [1, 3, 4, 1, -2, -6, -1], 5),
}


def _pfq_positions(p, q, trajectory, start):
    axes = list(pFq.x_axes(p)) + list(pFq.y_axes(q))
    return Position(dict(zip(axes, trajectory))), Position(dict(zip(axes, start)))


@pytest.mark.parametrize("name", REAL_TRAJECTORIES.keys())
def test_walk_fast_walk_real_trajectories(compare_to_reference_walk, name):
    p, q, z, trajectory, start = REAL_TRAJECTORIES[name]
    cmf = pFq(p, q, z)
    trajectory, start = _pfq_positions(p, q, trajectory, start)
    walks = compare_to_reference_walk(
        lambda: cmf.walk(trajectory, [1, 100, 900, 950, 1000], start)
    )
    assert walks is not None  # These trajectories are not singular


@pytest.mark.parametrize("name", SINGULAR_TRAJECTORIES.keys())
@pytest.mark.parametrize("fast", [False, True], ids=["reference", "fast"])
def test_walk_singular_real_trajectories(monkeypatch, name, fast):
    p, q, z, trajectory, start, singular_step = SINGULAR_TRAJECTORIES[name]
    cmf = pFq(p, q, z)
    trajectory, start = _pfq_positions(p, q, trajectory, start)
    matrix = cmf.trajectory_matrix(trajectory, start)
    (symbol,) = matrix.free_symbols
    if fast:
        monkeypatch.setattr(NumericMatrix, "FAST_WALK_MIN_DEPTH", 0)
    else:
        monkeypatch.setattr(NumericMatrix, "walk", staticmethod(_reference_numeric_walk))

    def walk(iterations):
        return NumericMatrix.walk(
            matrix, Position({symbol: 1}), iterations, Position({symbol: 0})
        )

    walk(singular_step)  # All steps before the singular one are fine
    with raises(ZeroDivisionError):
        walk(singular_step + 1)
    Matrix._walk_inner.cache_clear()
    with raises(ZeroDivisionError):
        cmf.walk(trajectory, [900, 950, 1000], start)
    Matrix._walk_inner.cache_clear()
