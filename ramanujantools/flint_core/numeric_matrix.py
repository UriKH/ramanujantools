from __future__ import annotations

from typing import Callable

import sympy as sp
from flint import fmpq_mat, fmpq, fmpq_poly, fmpz, fmpz_poly

import ramanujantools as rt
from ramanujantools import Position
from ramanujantools.utils import batched, Batchable


class _UnsupportedExpression(Exception):
    """Raised when an expression cannot be handled by `_StepEvaluator`."""


class _StepEvaluator:
    r"""
    Evaluates the step matrices of a walk, $M(s + k \cdot t)$ for $k = 0, 1, \ldots$,
    using FLINT polynomial arithmetic.

    Each entry is converted once into a pair of polynomials in the step index $k$,
    obtained by substituting $s + k \cdot t$ into the expression. The pair is kept
    exactly as the expression is written: denominators are never reduced against
    numerators. A step where any written denominator vanishes therefore raises
    `ZeroDivisionError`, exactly like `NumericMatrix.lambda_from_rt`.
    Denominators nested inside a negative power (as in `1/(1 + 1/n)`) do not end up in
    the final denominator, so they are kept as separate guard polynomials.

    Every distinct polynomial is stored as a rational content times a primitive
    `fmpz_poly`, and evaluated once per step using FLINT's Horner evaluation.

    Raises `_UnsupportedExpression` on construction for anything else (symbolic or
    non-integer exponents, symbols missing from `start`, non-rational numbers,
    positions that are not python ints, ...).
    """

    def __init__(self, matrix: rt.Matrix, trajectory: Position, start: Position):
        self.rows, self.cols = matrix.shape
        self._substitution = {}
        for symbol, value in start.items():
            step = trajectory[symbol]
            if type(value) is not int or type(step) is not int:
                # `lambda_from_rt` only accepts python ints (it raises TypeError otherwise)
                raise _UnsupportedExpression(f"Non-int position for {symbol}")
            self._substitution[symbol] = fmpq_poly([value, step])
        self._polynomials = []
        self._polynomial_indices = {}
        self._guards = []
        self._entries = []
        self._converted = {}
        for expr in matrix:
            numerator, denominator, guards = self._convert(sp.sympify(expr))
            numerator_content, numerator_index = self._register(numerator)
            denominator_content, denominator_index = self._register(denominator)
            for guard in guards:
                _, guard_index = self._register(guard)
                if guard_index not in self._guards:
                    self._guards.append(guard_index)
            self._entries.append(
                (
                    numerator_content / denominator_content,
                    numerator_index,
                    denominator_index,
                )
            )
        # Only needed during conversion, and large for high degree matrices
        del self._converted, self._polynomial_indices

    def _register(self, polynomial: fmpq_poly) -> tuple[fmpq, int | None]:
        """
        Splits `polynomial` into a rational content and a primitive integer polynomial,
        and returns the content and the index of the integer polynomial,
        or None if it is constant 1.
        A zero polynomial keeps a zero integer polynomial, so that dividing by it raises.
        """
        if polynomial == 0:
            content, primitive = fmpq(1), fmpz_poly(0)
        else:
            numerator = polynomial.numer()
            content = fmpq(numerator.content(), polynomial.denom())
            primitive = numerator // numerator.content()
            if primitive.degree() == 0:
                return content * fmpq(int(primitive[0])), None
        key = tuple(int(c) for c in primitive.coeffs())
        if key not in self._polynomial_indices:
            self._polynomial_indices[key] = len(self._polynomials)
            self._polynomials.append(primitive)
        return content, self._polynomial_indices[key]

    def _convert(self, expr) -> tuple[fmpq_poly, fmpq_poly, tuple[fmpq_poly, ...]]:
        """
        Returns `(numerator, denominator, guards)` of `expr` as polynomials in the step index.
        Results are memoized per subexpression, since factored matrices repeat factors across entries.
        """
        if expr not in self._converted:
            self._converted[expr] = self._convert_uncached(expr)
        return self._converted[expr]

    def _convert_uncached(self, expr):
        if isinstance(expr, sp.Rational):
            return fmpq_poly([fmpq(int(expr.p), int(expr.q))]), fmpq_poly([1]), ()
        if isinstance(expr, sp.Symbol):
            if expr not in self._substitution:
                raise _UnsupportedExpression(f"Symbol {expr} is missing from start")
            return self._substitution[expr], fmpq_poly([1]), ()
        if isinstance(expr, sp.Add):
            numerator, denominator, guards = self._convert(expr.args[0])
            for arg in expr.args[1:]:
                arg_numerator, arg_denominator, arg_guards = self._convert(arg)
                if arg_denominator == denominator:
                    numerator = numerator + arg_numerator
                else:
                    numerator = (
                        numerator * arg_denominator + arg_numerator * denominator
                    )
                    denominator = denominator * arg_denominator
                guards = guards + arg_guards
            return numerator, denominator, guards
        if isinstance(expr, sp.Mul):
            numerator, denominator, guards = fmpq_poly([1]), fmpq_poly([1]), ()
            for arg in expr.args:
                arg_numerator, arg_denominator, arg_guards = self._convert(arg)
                numerator = numerator * arg_numerator
                denominator = denominator * arg_denominator
                guards = guards + arg_guards
            return numerator, denominator, guards
        if isinstance(expr, sp.Pow):
            if not isinstance(expr.exp, sp.Integer):
                raise _UnsupportedExpression(f"Unsupported exponent in {expr}")
            exponent = int(expr.exp)
            numerator, denominator, guards = self._convert(expr.base)
            if exponent >= 0:
                return numerator**exponent, denominator**exponent, guards
            if denominator.degree() != 0:
                # The base's own denominator moves to the numerator,
                # but evaluating the base would still divide by it.
                guards = guards + (denominator,)
            return denominator ** (-exponent), numerator ** (-exponent), guards
        raise _UnsupportedExpression(f"Unsupported sympy type: {type(expr)}")

    def __call__(self, step: int) -> NumericMatrix:
        point = fmpz(step)
        values = [polynomial(point) for polynomial in self._polynomials]
        for guard_index in self._guards:
            if values[guard_index] == 0:
                raise ZeroDivisionError("Walk step is singular")
        cells = []
        for content, numerator_index, denominator_index in self._entries:
            value = content
            if numerator_index is not None:
                value = value * values[numerator_index]
            if denominator_index is not None:
                value = value / values[denominator_index]
            cells.append(value)
        return NumericMatrix(self.rows, self.cols, cells)


class NumericMatrix(fmpq_mat):
    # Minimal walk depth for which `walk` uses `_StepEvaluator`, chosen by benchmark:
    # below it, converting the matrix to polynomials costs more than it saves.
    FAST_WALK_MIN_DEPTH = 32

    @staticmethod
    def eye(N: int):
        """
        Returns an identity matrix of size N
        """
        retval = NumericMatrix(N, N)
        for i in range(N):
            retval[i, i] = 1
        return retval

    @staticmethod
    def lambda_from_rt(matrix: rt.Matrix) -> Callable:
        """
        Returns a function that evaluates the matrix at given a point
        and returns it as a NumericMatrix.
        """

        def _mul(args, subs):
            result = fmpq(1)
            for arg in args:
                val = arg(subs) if callable(arg) and subs is not None else arg
                result *= val
            return result

        def flint_lambdify(expr):
            """
            A recursive walker that constructs a flint based lambda expression
            from the sympy expression.
            """
            if isinstance(expr, sp.Integer):
                return fmpq(int(expr))
            elif isinstance(expr, sp.Rational):
                return fmpq(expr.p, expr.q)
            elif isinstance(expr, sp.Symbol):
                # Return a function that takes substitutions as a dict
                return lambda subs: fmpq(subs[expr])
            elif isinstance(expr, sp.Add):
                args = [flint_lambdify(arg) for arg in expr.args]
                if any(callable(arg) for arg in args):
                    return lambda subs: sum(
                        arg(subs) if callable(arg) else arg for arg in args
                    )
                else:
                    return sum(args)
            elif isinstance(expr, sp.Mul):
                args = [flint_lambdify(arg) for arg in expr.args]
                if any(callable(arg) for arg in args):
                    return lambda subs: _mul(args, subs)
                else:
                    return _mul(args, None)
            elif isinstance(expr, sp.Pow):
                base = flint_lambdify(expr.base)
                exp = flint_lambdify(expr.exp)
                if callable(base) or callable(exp):
                    return lambda subs: (base(subs) if callable(base) else base) ** int(
                        exp(subs) if callable(exp) else exp
                    )
                else:
                    return base ** int(exp)
            elif isinstance(expr, sp.MatrixBase):
                # Recursively convert each element
                rows, cols = expr.shape
                data = [
                    [flint_lambdify(expr[i, j]) for j in range(cols)]
                    for i in range(rows)
                ]

                return lambda subs: NumericMatrix(
                    [
                        [cell(subs) if callable(cell) else cell for cell in row]
                        for row in data
                    ]
                )

            else:
                raise NotImplementedError(f"Unsupported sympy type: {type(expr)}")

        return flint_lambdify(matrix)

    @staticmethod
    @batched("iterations")
    def walk(
        matrix: rt.Matrix,
        trajectory: Position,
        iterations: Batchable[int],
        start: Position,
    ) -> Batchable[NumericMatrix]:
        r"""
        Computes $\prod_{k=0}^{N-1} M(s + k \cdot t)$ at each requested depth $N$.

        Step matrices are evaluated by `_StepEvaluator`, which turns each entry into
        FLINT polynomials in $k$ once and evaluates them in C. If the matrix contains
        an expression the evaluator does not support, this falls back to
        `NumericMatrix.lambda_from_rt`. Both produce identical exact results, and both
        raise `ZeroDivisionError` when a step's denominator vanishes.

        Walks shorter than `FAST_WALK_MIN_DEPTH` also use `lambda_from_rt`, since there
        the one-time polynomial conversion costs more than it saves.

        All step matrices are evaluated before any multiplication, and then multiplied
        using a balanced product tree.
        """
        N = iterations[-1]
        if N < NumericMatrix.FAST_WALK_MIN_DEPTH:
            return NumericMatrix._walk_reference(matrix, trajectory, iterations, start)
        try:
            evaluator = _StepEvaluator(matrix, trajectory, start)
        except _UnsupportedExpression:
            return NumericMatrix._walk_reference(matrix, trajectory, iterations, start)
        step_matrices = [evaluator(step) for step in range(N)]
        del evaluator
        return NumericMatrix._product_checkpoints(
            step_matrices, iterations, matrix.rows
        )

    @staticmethod
    def _walk_reference(
        matrix: rt.Matrix,
        trajectory: Position,
        iterations: list[int],
        start: Position,
    ) -> list[NumericMatrix]:
        """
        The original walk, evaluating each step through `NumericMatrix.lambda_from_rt`.
        Used as a fallback by `NumericMatrix.walk`, and as a reference in tests.
        """
        N = iterations[-1]
        fast_subs = NumericMatrix.lambda_from_rt(matrix)

        # Pre-evaluate all per-step matrices into a flat list.
        position = start.copy()
        step_matrices = []
        for _ in range(N):
            step_matrices.append(fast_subs(position))
            position += trajectory
        return NumericMatrix._product_checkpoints(
            step_matrices, iterations, matrix.rows
        )

    @staticmethod
    def _product_checkpoints(
        step_matrices: list[NumericMatrix], iterations: list[int], dim: int
    ) -> list[NumericMatrix]:
        """
        Returns the products of the first `depth` step matrices, for each depth in `iterations`.
        """
        # Below this range size, a sequential loop is faster than recursion.
        SEQUENTIAL_THRESHOLD = 8

        def _product_tree(first, last):
            """Return step_matrices[first] * step_matrices[first+1] * … * step_matrices[last].

            Uses balanced divide-and-conquer so that intermediate products stay
            small, which is critical for FLINT rational matrices whose entry
            sizes grow with each multiplication.
            """
            span = last - first
            if span == 0:
                return step_matrices[first]
            if span <= SEQUENTIAL_THRESHOLD:
                result = step_matrices[first]
                for i in range(first + 1, last + 1):
                    result = result * step_matrices[i]
                return result
            mid = (first + last) >> 1
            return _product_tree(first, mid) * _product_tree(mid + 1, last)

        # Build results at each requested checkpoint.
        results = []
        accumulated = NumericMatrix.eye(dim)
        prev_depth = 0
        for target in iterations:
            if target > prev_depth:
                segment = _product_tree(prev_depth, target - 1)
                accumulated = accumulated * segment
            results.append(NumericMatrix(accumulated))
            prev_depth = target
        return results

    def to_rt(self) -> rt.Matrix:
        return rt.Matrix(self.nrows(), self.ncols(), list(self))

    def __neg__(self) -> NumericMatrix:
        return NumericMatrix(super().__neg__())

    def __add__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__add__(other))

    def __radd__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__radd__(other))

    def __sub__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__sub__(other))

    def __rsub__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__rsub__(other))

    def __mul__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__mul__(other))

    def __rmul__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__rmul__(other))

    def __div__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__div__(other))
