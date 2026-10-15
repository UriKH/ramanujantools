
import flint
import sympy as sp
from sympy.abc import n, x, y

from ramanujantools.flint_core import context
from ramanujantools.flint_core.context import (
    _flint_composition,
    flint_ctx,
    flint_from_sympy,
    flint_to_sympy,
)


def test_ctx_names_match_generator_strings():
    """``ctx.names()`` (used instead of ``str(generator)``) gives the same names."""
    for fmpz in (True, False):
        ctx = flint_ctx([x, y, n], fmpz)
        assert list(ctx.names()) == [str(g) for g in ctx.gens()]


def test_sympy_round_trip():
    expr = 3 * x**2 * y - 5 * x * n + 7
    for fmpz in (True, False):
        ctx = flint_ctx([x, y, n], fmpz)
        assert sp.expand(flint_to_sympy(flint_from_sympy(expr, ctx), factor=False) - expr) == 0


def test_composition_substitutes_by_name():
    ctx = flint_ctx([x, y], True)
    gx, gy = ctx.gens()
    args = _flint_composition(ctx, {y: x + 1})
    assert args[0] == gx
    assert args[1] == gx + 1


def test_conversions_never_stringify_flint_polynomials(monkeypatch):
    """Regression: python-flint <= 0.9.0 leaks the C string returned by
    ``fmpz_mpoly_get_str_pretty`` on every ``str()`` of an mpoly (64 B per
    generator with 8 variables; thousands of calls per CMF walk).  This module
    must get generator names from ``ctx.names()`` instead."""

    def no_flint_str(obj=""):
        assert not isinstance(obj, (flint.fmpz_mpoly, flint.fmpq_mpoly)), "str() of a FLINT mpoly"
        return builtin_str(obj)

    builtin_str = str
    monkeypatch.setattr(context, "str", no_flint_str, raising=False)
    for fmpz in (True, False):
        ctx = flint_ctx([x, y, n], fmpz)
        poly = flint_from_sympy(sp.expand((x + 2 * y) ** 2), ctx)
        flint_to_sympy(poly, factor=False)
        flint_to_sympy(poly, factor=True)
        _flint_composition(ctx, {x: y + n})
