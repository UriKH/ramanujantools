from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral

import sympy as sp
from sympy.core.mul import _keep_coeff
from sympy.abc import x, y
from sympy.polys.matrices import DomainMatrix

from ramanujantools import Matrix
from ramanujantools.cmf import CMF
from ramanujantools.flint_core import FlintRational
from ramanujantools.flint_core.context import (
    FlintContext,
    _flint_polynomial_to_sympy,
    flint_converter,
    flint_ctx,
)

theta_x, theta_y = sp.symbols("theta_x theta_y")

# The CMF whose block determinants are computed by forked worker processes (see `KdF.determinants`).
# Forked workers inherit it, so the large FLINT matrices are never pickled.
_forked_cmf = None


def _forked_determinant(axis: sp.Symbol) -> sp.Expr:
    return _forked_cmf._representative_determinant(axis)


def _to_sympy(value: FlintRational) -> sp.Expr:
    numerator = _flint_polynomial_to_sympy(value.numerator)
    if value.denominator == 1:
        return numerator
    return numerator / _flint_polynomial_to_sympy(value.denominator)


def _factor(polynomial) -> tuple[sp.Rational, sp.Expr]:
    """
    Factors a FLINT polynomial into its content and a product of primitive irreducible factors,
    normalized as in `sp.factor`.
    """
    content, factors = polynomial.factor()
    content = sp.Rational(int(content.numerator), int(content.denominator))
    result = sp.S.One
    for factor, multiplicity in factors:
        factor_content, primitive = sp.factor(
            _flint_polynomial_to_sympy(factor)
        ).as_coeff_Mul()
        content *= factor_content**multiplicity
        result *= primitive**multiplicity
    return content, result


class ConstructionError(ValueError):
    """The generic basis of the KdF construction degenerates for the requested arguments."""


class KdF(CMF):
    r"""
    Represents the Kampé de Fériet CMF of $F^{p:q;k}_{l:m;n}(x, y)$, the two-variable generalization of pFq:
    https://en.wikipedia.org/wiki/Kamp%C3%A9_de_F%C3%A9riet_function

    The function is annihilated by the Horn operators

    $$D \cdot Q_x - x A P_x, \quad D \cdot Q_y - y A P_y$$

    with $\theta = \theta_x + \theta_y$, $A=\prod(\theta+a_i)$, $D=\prod(\theta-d_i)$ (physical lower $d_i$),
    $P_x=\prod(\theta_x+b_i)$, $Q_x=\theta_x \prod(\theta_x-e_i)$ (and similarly for $y$ with $c_i, f_i$).
    The CMF is built from a basis of the quotient module (of a rank that depends only on the six sizes),
    and the axes are the parameters $a_i, b_i, c_i$ (upper) and $nd_i, ne_i, nf_i$ (the *negatives* of the physical lower
    parameters $d_i, e_i, f_i$), so that every axis is incremented by one.

    Matrices act on the right of the row state `basis`, following the convention of the other CMFs.
    For generic arguments the construction is the paper's graded reduction (basis of the full Horn system).
    For arguments where that basis degenerates (e.g. $x=y$, as in the L(2, chi_-3) CMF at $x=y=1/4$),
    the contiguity module of the function itself is constructed instead, in a basis of ordinary derivatives.
    In both cases, the matrix of an axis is $(N I + \Theta) / N$, where $\Theta$ is the matrix of
    $\theta_x + \theta_y$, $\theta_x$ or $\theta_y$ (by the block of the axis) and $N$ is $a$ or $-(nd + 1)$.
    """

    def __init__(
        self,
        p: int,
        q: int,
        k: int,
        l: int,
        m: int,
        n: int,
        x: sp.Expr = x,
        y: sp.Expr = y,
        method: str = "auto",
        max_pairs: int = 256,
    ):
        r"""
        Constructs the KdF CMF $F^{p:q;k}_{l:m;n}(x, y)$.
        Args:
            p, q, k: The number of upper parameters of the shared ($a$), the $x$ ($b$) and the $y$ ($c$) blocks.
            l, m, n: The number of lower parameters of the shared ($d$), the $x$ ($e$) and the $y$ ($f$) blocks.
                Note that the $x$ and $y$ lower blocks implicitly contain an additional $1$ each (as in pFq).
            x, y: If given, will construct the CMF for specific arguments. Must be exact (no floats).
            method: "generic" uses the paper's basis, valid for generic arguments (the matrices act on the basis `basis`).
                "specialized" reconstructs the contiguity module of the function at the given arguments
                (basis of ordinary derivatives; the rank can differ from the generic rank).
                "auto" (default) tries the generic construction and falls back to the specialized one if the
                generic basis degenerates (e.g. at x=y=1).
            max_pairs: Bounds the Groebner completion of the specialized construction.
        Raises:
            ValueError: if a size is not a non-negative integer, an argument is not exact,
                or no CMF could be certified.
        """
        self.counts = KdF._validate_counts((p, q, k, l, m, n))
        self.x = sp.S(x)
        self.y = sp.S(y)
        if self.x.has(sp.Float) or self.y.has(sp.Float):
            raise ValueError("KdF arguments must be exact, e.g. Rational(1, 3)")
        if self.x.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or self.y.has(
            sp.zoo, sp.nan, sp.oo, -sp.oo
        ):
            raise ValueError("KdF arguments must be finite")
        if method not in ("auto", "generic", "specialized"):
            raise ValueError("method must be auto, generic or specialized")
        if not any(self.counts):
            raise ValueError("KdF requires at least one parameter (axis)")
        theta = None
        if method != "specialized":
            self.basis, self.basis_info = KdF.construct_basis(*self.counts)
            try:
                theta = self._generic_theta_matrices()
            except ConstructionError:
                if method == "generic":
                    raise
        if theta is None:
            theta = self._specialized_theta_matrices(max_pairs)
        self._theta = theta
        self._determinants = {}
        super().__init__(self._sympy_matrices(), validate=False)

    def _generic_theta_matrices(self) -> tuple[list, list, FlintContext]:
        theta_x_matrix, theta_y_matrix = KdF._theta_matrices(
            *self.counts, self.x, self.y
        )
        ctx = KdF._flint_context(self.counts, self.x, self.y)
        convert = flint_converter(ctx)
        return tuple(
            [
                [FlintRational.from_sympy(m[i, j], ctx, convert) for j in range(m.cols)]
                for i in range(m.rows)
            ]
            for m in (theta_x_matrix, theta_y_matrix)
        ) + (ctx,)

    def _specialized_theta_matrices(
        self, max_pairs: int
    ) -> tuple[list, list, FlintContext]:
        from ramanujantools.cmf.kdf_specialized import construct

        horn, elimination = KdF.horn_operators(*self.counts)
        powers, theta_x_matrix, theta_y_matrix, ctx = construct(
            (*horn, elimination), self.x, self.y, KdF.blocks(*self.counts), max_pairs
        )
        self.basis = tuple(
            sp.Symbol("partial_x") ** i * sp.Symbol("partial_y") ** j for i, j in powers
        )
        self.basis_info = dict(method="specialized", powers=powers)
        return theta_x_matrix, theta_y_matrix, ctx

    @staticmethod
    def _flint_context(counts, x_value, y_value) -> FlintContext:
        parameters = set(sum(KdF.blocks(*counts), ()))
        return flint_ctx(
            parameters | x_value.free_symbols | y_value.free_symbols, fmpz=False
        )

    def _operator(self, axis: sp.Symbol) -> tuple[str, sp.Expr]:
        r"""
        Returns the operator of the block of an axis ("total" for $\theta_x + \theta_y$, "x" or "y")
        and the normalizer of the axis ($a$ for upper parameters, $-(nd + 1)$ for negated lower ones).
        """
        a, b, c, nd, ne, nf = KdF.blocks(*self.counts)
        operator = "total" if axis in a + nd else "x" if axis in b + ne else "y"
        return operator, (-axis - 1 if axis in nd + ne + nf else axis)

    def _flint_operators(self) -> dict[str, list[list[FlintRational]]]:
        theta_x_matrix, theta_y_matrix, _ = self._theta
        total = [
            [u + v for u, v in zip(row_x, row_y)]
            for row_x, row_y in zip(theta_x_matrix, theta_y_matrix)
        ]
        return {"total": total, "x": theta_x_matrix, "y": theta_y_matrix}

    def _sympy_matrices(self) -> dict[sp.Symbol, Matrix]:
        r"""
        Converts the axis matrices $(N I + \Theta) / N$ to sympy.
        Every operator $\Theta$ is converted once, and only the diagonal is converted per axis.
        """
        _, _, ctx = self._theta
        convert = flint_converter(ctx)
        operators = self._flint_operators()
        converted = {}
        matrices = {}
        for axis in sum(KdF.blocks(*self.counts), ()):
            name, normalizer = self._operator(axis)
            operator = operators[name]
            if name not in converted:
                converted[name] = [
                    [_to_sympy(value) for value in row] for row in operator
                ]
            flint_normalizer = FlintRational(convert(normalizer), ctx.constant(1), ctx)
            rank = len(operator)
            matrices[axis] = Matrix(
                rank,
                rank,
                lambda i, j: (
                    _to_sympy((operator[i][i] + flint_normalizer) / flint_normalizer)
                    if i == j
                    else converted[name][i][j] / normalizer
                ),
            )
        return matrices

    def _flint_matrix(
        self, axis: sp.Symbol
    ) -> tuple[list[list[FlintRational]], FlintRational]:
        r"""
        Returns the matrix of an axis as FLINT rational functions, as a pair $(N I + \Theta, N)$
        such that the matrix is $(N I + \Theta) / N$ (see `_operator`).
        """
        if self._theta is None:  # e.g. after unpickling
            matrix = self.matrices[axis]
            ctx = KdF._flint_context(self.counts, self.x, self.y)
            convert = flint_converter(ctx)
            rows = [
                [
                    FlintRational.from_sympy(matrix[i, j], ctx, convert)
                    for j in range(matrix.cols)
                ]
                for i in range(matrix.rows)
            ]
            return rows, FlintRational(ctx.constant(1), ctx.constant(1), ctx)
        _, _, ctx = self._theta
        name, normalizer = self._operator(axis)
        operator = self._flint_operators()[name]
        normalizer = FlintRational(
            flint_converter(ctx)(normalizer), ctx.constant(1), ctx
        )
        rows = [
            [value + normalizer if i == j else value for j, value in enumerate(row)]
            for i, row in enumerate(operator)
        ]
        return rows, normalizer

    def __repr__(self) -> str:
        return f"KdF({self.counts + (self.x, self.y)})"

    def __getstate__(self):
        state = {"matrices": self.matrices}
        state.update({"counts": self.counts, "x": self.x, "y": self.y})
        state.update({"basis": self.basis, "basis_info": self.basis_info})
        return state

    def __setstate__(self, state):
        self._theta = None
        self._determinants = {}
        self.counts = state["counts"]
        self.x = state["x"]
        self.y = state["y"]
        self.basis = state["basis"]
        self.basis_info = state["basis_info"]
        super().__setstate__(state["matrices"])

    @staticmethod
    def _validate_counts(counts) -> tuple[int, ...]:
        if len(counts) != 6 or any(
            isinstance(c, bool) or not isinstance(c, (Integral, sp.Integer)) or c < 0
            for c in counts
        ):
            raise ValueError(
                "KdF requires six non-negative integer sizes (p,q,k,l,m,n)"
            )
        return tuple(int(c) for c in counts)

    @staticmethod
    def blocks(p, q, k, l, m, n) -> tuple[tuple[sp.Symbol, ...], ...]:
        r"""
        Returns the parameter blocks $(a, b, c, nd, ne, nf)$. The last three are the negated lower parameters.
        """
        return tuple(
            tuple(sp.symbols(f"{name}:{size}"))
            for name, size in zip(("a", "b", "c", "nd", "ne", "nf"), (p, q, k, l, m, n))
        )

    def axes_blocks(self) -> tuple[tuple[sp.Symbol, ...], ...]:
        return KdF.blocks(*self.counts)

    @staticmethod
    def predict_rank(p: int, q: int, k: int, l: int, m: int, n: int) -> int:
        """
        Returns the rank of the CMF (i.e, the rank of its matrices) for generic arguments.
        """
        return len(KdF.construct_basis(p, q, k, l, m, n)[0])

    @staticmethod
    def construct_basis(p, q, k, l, m, n) -> tuple[tuple[sp.Expr, ...], dict]:
        r"""
        Returns the two-list basis of the quotient module (as polynomials in $\theta_x, \theta_y$) and its metadata.
        The ordering is stable: in the first list the power of $\theta$ is outermost, in the second the power of
        the first variable is outermost.
        """
        s, t = m + 1, n + 1
        xi_x, xi_y = p + q - l - s, p + k - l - t
        swapped = xi_x > xi_y
        U, V = (theta_y, theta_x) if swapped else (theta_x, theta_y)
        if swapped:
            q, k, s, t, xi_x, xi_y = k, q, t, s, xi_y, xi_x
        if xi_y <= 0:
            h, ax, ay = l, s, t
        elif xi_x <= 0:
            h = min(l, p)
            ax, ay = s + l - h, k + p - h
        else:
            h, ax, ay = p, q, k
        basis = tuple((U + V) ** a * U**b for a in range(h) for b in range(s + k))
        basis += tuple((U + V) ** h * U**i * V**j for i in range(ax) for j in range(ay))
        return basis, dict(h=h, ax=ax, ay=ay, swapped=swapped, excesses=(xi_x, xi_y))

    @staticmethod
    def horn_operators(p, q, k, l, m, n):
        r"""
        Returns the two Horn operators and the elimination operator $K = y P_y Q_x - x P_x Q_y$
        (a consequence of the Horn equations), as polynomials in $\theta_x, \theta_y$ with coefficients in $x, y$.
        """
        a, b, c, nd, ne, nf = KdF.blocks(p, q, k, l, m, n)
        theta = theta_x + theta_y
        A = sp.prod(theta + a_i for a_i in a)
        D = sp.prod(theta - nd_i - 1 for nd_i in nd)
        Px = sp.prod(theta_x + b_i for b_i in b)
        Py = sp.prod(theta_y + c_i for c_i in c)
        Qx = theta_x * sp.prod(theta_x - ne_i - 1 for ne_i in ne)
        Qy = theta_y * sp.prod(theta_y - nf_i - 1 for nf_i in nf)
        return (
            (sp.expand(D * Qx - x * A * Px), sp.expand(D * Qy - y * A * Py)),
            sp.expand(y * Py * Qx - x * Px * Qy),
        )

    @staticmethod
    def _left(expr: sp.Expr, i: int, j: int) -> sp.Expr:
        """Normal-ordered $\\theta_x^i \\theta_y^j \\cdot$ expr, including the derivatives of the coefficients."""
        for _ in range(i):
            expr = sp.expand(theta_x * expr + x * sp.diff(expr, x))
        for _ in range(j):
            expr = sp.expand(theta_y * expr + y * sp.diff(expr, y))
        return expr

    @staticmethod
    def _theta_matrices(p, q, k, l, m, n, x_value, y_value) -> tuple[Matrix, Matrix]:
        r"""
        Returns the matrices of $\theta_x, \theta_y$ in the basis of the quotient module.
        Solves, degree by degree, an exact linear system for the reduction of every monomial to the basis.
        """
        basis, _ = KdF.construct_basis(p, q, k, l, m, n)
        rank = len(basis)
        parameters = sum(KdF.blocks(p, q, k, l, m, n), ())
        symbols = sorted(
            set(parameters) | x_value.free_symbols | y_value.free_symbols,
            key=sp.default_sort_key,
        )
        field = sp.QQ.frac_field(*symbols) if symbols else sp.QQ
        zero, one = field.zero, field.one
        arguments = {x: x_value, y: y_value}

        def poly(expr):
            return sp.Poly(expr, theta_x, theta_y)

        basis_polys = [poly(b) for b in basis]
        horn, elimination = KdF.horn_operators(p, q, k, l, m, n)
        relations = [poly(h) for h in (*horn, elimination) if h != 0]
        reductions = {}
        max_degree = max(int(b.total_degree()) for b in basis_polys) + 1

        for degree in range(max_degree + 1):
            monomials = [(i, degree - i) for i in range(degree + 1)]
            rows, rhs = [], []
            for index, b in enumerate(basis_polys):
                if b.total_degree() == degree:
                    rows.append(
                        [field.from_sympy(b.coeff_monomial(v)) for v in monomials]
                    )
                    rhs.append([one if index == j else zero for j in range(rank)])
            basis_count = len(rows)
            for h in relations:
                extra = degree - int(h.total_degree())
                for i in range(extra + 1):
                    # Differentiate before substituting the arguments, even for constants or x=y.
                    relation = poly(
                        KdF._left(h.as_expr(), i, extra - i).subs(
                            arguments, simultaneous=True
                        )
                    )
                    rows.append(
                        [
                            field.from_sympy(relation.coeff_monomial(v))
                            for v in monomials
                        ]
                    )
                    value = [zero] * rank
                    for powers, coefficient in relation.terms():
                        if sum(powers) >= degree:
                            continue
                        coefficient = field.from_sympy(coefficient)
                        for j, v in enumerate(reductions[powers]):
                            value[j] -= coefficient * v
                    rhs.append(value)
            if not rows:
                raise ValueError(f"No spanning relations at degree {degree}")
            system = DomainMatrix.from_list(rows, field)
            _, pivots = system.transpose().rref()
            if len(pivots) != degree + 1 or tuple(pivots[:basis_count]) != tuple(
                range(basis_count)
            ):
                raise ConstructionError(
                    f"The basis of the KdF construction degenerates at degree {degree} "
                    f"for the arguments ({x_value}, {y_value})"
                )
            chosen = DomainMatrix.from_list([rows[i] for i in pivots], field)
            target = DomainMatrix.from_list([rhs[i] for i in pivots], field)
            solution = chosen.inv().matmul(target)
            # Verify the unused relations too: the closure is never guessed.
            residual = system.matmul(solution) - DomainMatrix.from_list(rhs, field)
            if not residual.is_zero_matrix:
                raise ConstructionError(
                    f"Inconsistent exact reduction at degree {degree}"
                )
            for powers, value in zip(monomials, solution.to_list()):
                reductions[powers] = value

        def border(operator):
            columns = []
            for b in basis:
                value = [zero] * rank
                for powers, coefficient in poly(operator * b).terms():
                    coefficient = field.from_sympy(coefficient)
                    for j, v in enumerate(reductions[powers]):
                        value[j] += coefficient * v
                columns.append([field.to_sympy(v) for v in value])
            return Matrix(columns).T

        return border(theta_x), border(theta_y)

    def determinant(self, axis: sp.Symbol) -> sp.Expr:
        r"""
        Returns the determinant of an axis (basis) matrix in factored form.

        Unlike pFq and MeijerG, no closed formula in the six sizes is known for the KdF determinant
        (its factors depend on the regime of the basis and on the arguments), so it is computed exactly
        from the matrix: by Gaussian elimination over FLINT rational functions, followed by FLINT factorization.
        The entries can be very large while the determinant is a small product, and exact elimination over
        the field (unlike fraction-free elimination or `Matrix.det()`) keeps every pivot reduced.

        :param axis: one of the axes of the CMF.
        :raises ValueError: if axis is not an axis of the CMF.
        """
        if axis not in self.matrices:
            raise ValueError(f"not a KdF axis: {axis}")
        # The construction is symmetric within each block, so the determinant of any axis is obtained from
        # the first axis of its block by swapping the two symbols.
        representative = self._representative(axis)
        if representative not in self._determinants:
            self._determinants[representative] = self._representative_determinant(
                representative
            )
        return self._determinants[representative].xreplace(
            {representative: axis, axis: representative}
        )

    def determinants(self, processes: int | None = None) -> dict[sp.Symbol, sp.Expr]:
        r"""
        Returns the factored determinants of all axis matrices (see `determinant`).

        The blocks are independent, so their determinants are computed in parallel, in forked processes.
        Falls back to sequential computation where fork is unavailable (e.g. Windows).

        :param processes: the maximal number of processes. Defaults to the number of CPUs.
        """
        missing = [
            block[0]
            for block in KdF.blocks(*self.counts)
            if block and block[0] not in self._determinants
        ]
        processes = min(len(missing), processes or os.cpu_count() or 1)
        if processes > 1 and "fork" in multiprocessing.get_all_start_methods():
            global _forked_cmf
            _forked_cmf = self
            try:
                with ProcessPoolExecutor(
                    processes, mp_context=multiprocessing.get_context("fork")
                ) as executor:
                    results = list(executor.map(_forked_determinant, missing))
            finally:
                _forked_cmf = None
            self._determinants.update(zip(missing, results))
        return {axis: self.determinant(axis) for axis in self.matrices}

    def _representative(self, axis: sp.Symbol) -> sp.Symbol:
        return next(block[0] for block in KdF.blocks(*self.counts) if axis in block)

    def _representative_determinant(self, representative: sp.Symbol) -> sp.Expr:
        rows, normalizer = self._flint_matrix(representative)
        return KdF._flint_determinant(rows, normalizer)

    @staticmethod
    def _flint_determinant(
        rows: list[list[FlintRational]], normalizer: FlintRational
    ) -> sp.Expr:
        r"""
        Returns the factored determinant of `rows / normalizer`, a matrix of FLINT rational functions.
        Uses Gaussian elimination over the field, which keeps every pivot reduced. The entries can be large while
        the determinant is a small product, so this is much faster than fraction-free elimination.
        """
        rows = [list(row) for row in rows]
        size = len(rows)
        sign = 1
        determinant = None
        for k in range(size):
            # Full pivoting on the smallest entry: the matrices typically have a few columns of huge entries,
            # which should be eliminated last.
            entries = [
                (len(rows[i][j].numerator) + len(rows[i][j].denominator), i, j)
                for i in range(k, size)
                for j in range(k, size)
                if rows[i][j].numerator != 0
            ]
            if not entries:
                return sp.S.Zero
            _, i, j = min(entries)
            if i != k:
                rows[k], rows[i] = rows[i], rows[k]
                sign = -sign
            if j != k:
                for row in rows:
                    row[k], row[j] = row[j], row[k]
                sign = -sign
            determinant = (
                rows[k][k] if determinant is None else determinant * rows[k][k]
            )
            for i in range(k + 1, size):
                if rows[i][k].numerator != 0:
                    scale = rows[i][k] / rows[k][k]
                    for j in range(k + 1, size):
                        rows[i][j] = rows[i][j] - scale * rows[k][j]
        determinant = determinant * sign
        for _ in rows:
            determinant = determinant / normalizer
        numerator_content, numerator = _factor(determinant.numerator)
        denominator_content, denominator = _factor(determinant.denominator)
        # Keeps the constant from being distributed over a single sum factor, as `sp.factor` does.
        return _keep_coeff(
            numerator_content / denominator_content, numerator / denominator
        )

    def subs(self, substitutions) -> KdF:
        self._validate_axes_substitutions(substitutions)
        return KdF(*self.counts, self.x.subs(substitutions), self.y.subs(substitutions))
