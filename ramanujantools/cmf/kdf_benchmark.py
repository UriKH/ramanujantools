import sympy as sp

from ramanujantools.cmf import KdF

quarter = sp.Rational(1, 4)


def test_construct_l2chi3(benchmark):
    """The rank-8 KdF CMF of L(2, chi_-3), at x=y=1/4 (specialized construction)."""
    benchmark.pedantic(KdF, args=(3, 2, 1, 3, 1, 0, quarter, quarter), rounds=1)


def test_determinants_l2chi3(benchmark):
    cmf = KdF(3, 2, 1, 3, 1, 0, quarter, quarter)

    def determinants():
        cmf._determinants.clear()
        return [cmf.determinant(axis) for axis in cmf.axes()]

    benchmark.pedantic(determinants, rounds=1)


def test_determinants_parallel_l2chi3(benchmark):
    cmf = KdF(3, 2, 1, 3, 1, 0, quarter, quarter)

    def determinants():
        cmf._determinants.clear()
        return cmf.determinants()

    benchmark.pedantic(determinants, rounds=1)


def test_construct_and_determinants_generic(benchmark):
    """The same family with symbolic arguments (generic construction, rank 11)."""

    def construct_and_determinants():
        cmf = KdF(3, 2, 1, 3, 1, 0)
        return [cmf.determinant(axis) for axis in cmf.axes()]

    benchmark.pedantic(construct_and_determinants, rounds=1)
