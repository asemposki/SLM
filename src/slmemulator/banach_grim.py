"""
Banach GRIM kernel interpolator (Lyons & McLeod, arXiv:2205.07495).

The default fitting method ("grim") implements the Banach Greedy
Recombination Interpolation Method: the dense kernel interpolant
phi = sum_i k(., p_i) C_i is treated as a positive combination of atoms and
thinned by recombination (Caratheodory-type support reduction that exactly
preserves the matched linear functionals and the total mass) against a
greedily grown set of evaluation functionals — at each step the functionals
with the largest absolute residual |sigma(phi - u)| are added, and
recombination guarantees support <= |L| + 1.

A second method ("pgreedy") provides f*P-greedy kernel interpolation
(RKHS power-function selection with incremental Cholesky, no recombination),
which is faster for large snapshot counts such as dense 1-D curves.

Can interpolate:
  * lists of matrix/vector snapshots A(p_i) over parameters p_i in R^d, or
  * any plain-text/CSV data file whose first column is x (or time) and
    whose remaining columns are outputs y1, y2, ... (see `fit_file`).

Command-line usage:
    python -m slmemulator.banach_grim data.dat --max-basis 40 -o interp.dat
"""

import numpy as np

__all__ = ["BanachGRIMInterpolator", "rbf_kernel", "recombination_thinning"]


def recombination_thinning(M, weights, tol=1e-12, max_iter=None):
    """
    Reduce the support of nonnegative weights while preserving linear moments.

    Given ``M`` of shape (r, N) and ``weights`` w >= 0 of shape (N,), returns
    new weights b >= 0 with ``M @ b == M @ w`` (to numerical precision) and
    support of at most r atoms (fewer if M's active columns are rank
    deficient). This is the recombination step of the Banach GRIM algorithm
    (Lemma 3.1 of arXiv:2205.07495): with an all-ones row included in M, the
    total mass ``sum(b) == sum(w)`` is preserved as well.

    Parameters:
        M (np.ndarray): Constraint matrix, shape (r, N). Row k holds the
            values of the k-th linear functional on the N atoms.
        weights (np.ndarray): Nonnegative weights, shape (N,).
        tol (float): Singular values below ``tol * largest`` count as null
            directions.
        max_iter (int, optional): Safety cap on elimination steps
            (default N).

    Returns:
        b (np.ndarray): Nonnegative weights, shape (N,), with
            ``np.count_nonzero(b) <= r`` and ``M @ b == M @ weights``.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    b = np.asarray(weights, dtype=float).copy()
    if b.ndim != 1 or M.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: M {M.shape}, weights {b.shape}.")
    if np.any(b < 0):
        raise ValueError("recombination requires nonnegative weights.")

    support = np.flatnonzero(b > 0.0)
    if support.size <= 1:
        return b

    # Null basis of the active columns, computed once: rows of Vt whose
    # singular value is (numerically) zero span the right null space.
    M_s = M[:, support]
    _, s, Vt = np.linalg.svd(M_s, full_matrices=True)
    rank = int(np.sum(s > tol * max(s[0] if s.size else 1.0, 1.0)))
    V = Vt[rank:].T.copy()  # (s_count, q): columns are null vectors

    if max_iter is None:
        max_iter = b.shape[0]

    for _ in range(min(max_iter, V.shape[1] if V.size else 0)):
        if V.size == 0 or support.size <= 1:
            break

        # Step along the first null direction until a weight hits zero,
        # keeping every weight nonnegative. Positive entries of v limit the
        # step; flip v if it has none.
        v = V[:, 0]
        if not np.any(v > tol):
            v = -v
            V[:, 0] = v
        pos = v > tol
        if not np.any(pos):
            # direction is (numerically) zero: discard it
            V = V[:, 1:]
            continue
        ratios = b[support[pos]] / v[pos]
        j_local_pos = int(np.argmin(ratios))
        gamma = ratios[j_local_pos]
        j_local = int(np.flatnonzero(pos)[j_local_pos])

        b[support] = b[support] - gamma * v
        b[support[j_local]] = 0.0  # exactly zero the eliminated atom
        np.clip(b, 0.0, None, out=b)

        # Update the null basis for the reduced support: use v (which has
        # v[j_local] > 0) to cancel the j_local-component of the remaining
        # directions, then drop v and the eliminated row.
        if V.shape[1] > 1:
            V[:, 1:] -= np.outer(v, V[j_local, 1:] / v[j_local])
        V = np.delete(V[:, 1:], j_local, axis=0)
        keep = np.ones(support.size, dtype=bool)
        keep[j_local] = False
        support = support[keep]
        # guard against numerical blow-up in the eliminated basis
        if V.size:
            norms = np.linalg.norm(V, axis=0)
            good = norms > tol
            V = V[:, good] / np.where(norms[good] > 0, norms[good], 1.0)

    return b

try:
    from scipy.linalg import solve_triangular as _solve_tri

    def _solve_lower(L, b):
        return _solve_tri(L, b, lower=True)

    def _solve_upper(U, b):
        return _solve_tri(U, b, lower=False)

except ImportError:  # numpy-only fallback (slower: LU instead of back-substitution)

    def _solve_lower(L, b):
        return np.linalg.solve(L, b)

    def _solve_upper(U, b):
        return np.linalg.solve(U, b)


def rbf_kernel(p, q, length_scale=1.0):
    """Gaussian/RBF kernel. `q` may be a single point (d,) or a batch (m, d);
    a batch returns shape (m,)."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    q = np.asarray(q, dtype=float)
    if q.ndim <= 1:
        d2 = np.sum((p - np.atleast_1d(q)) ** 2)
    else:
        d2 = np.sum((q - p[None, :]) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * length_scale**2))


class BanachGRIMInterpolator:
    """
    Greedy recombination-style kernel interpolator.

    Parameters
    ----------
    kernel_func : callable or None
        K(p, q) -> float, positive definite. May optionally accept a batch of
        second arguments (m, d) and return (m,) — this is auto-detected and
        used for speed. If None, a Gaussian RBF is used whose length scale is
        `length_scale`, or the median pairwise distance of the training
        parameters when `length_scale` is None.
    reg : float, default 1e-12
        Tikhonov jitter added to the Gram diagonal for numerical stability.
    use_residual_weight : bool, default True
        If True, the power-function score is multiplied by the current
        data-space residual norm at each candidate (f*P-greedy), biasing
        selection toward high-error regions.
    dual_selector : callable or None
        Overrides selection scoring. Signature:
        dual_selector(i, A_true_i, A_pred_i, power_i, state) -> float
        (higher is better).
    length_scale : float or None
        Length scale for the default RBF kernel (ignored if kernel_func given).
    center : bool, default False
        If True, interpolate deviations from the mean training snapshot and
        add the mean back on evaluation. This makes the interpolant exactly
        reproduce snapshot components that are constant across the training
        set, and keeps predictions anchored to the mean (instead of decaying
        toward zero) at query points far from all training points.

    Fitted attributes
    -----------------
    selected_idx : list of int      indices of the selected training points
    P_sel        : (m, d) array     selected parameter points
    L            : (m, m) array     Cholesky factor of K_sel + reg*I
    C            : (m, D) array     interpolation coefficients
    """

    def __init__(self, kernel_func=None, reg=1e-12, use_residual_weight=True,
                 dual_selector=None, length_scale=None, center=False):
        self.kernel = kernel_func
        self.reg = float(reg)
        self.use_residual_weight = bool(use_residual_weight)
        self.dual_selector = dual_selector
        self.length_scale = length_scale
        self.center = bool(center)

        # Fitted state
        self.selected_idx = []
        self.L = None
        self.C = None
        self.P_sel = None
        self.B_all = None
        self.snapshot_shape = None
        self._B_mean = None
        self._k = None
        self._kernel_vectorized = False

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _as_param_matrix(P):
        """Normalize parameters to a 2-D float array of shape (n, d)."""
        P = np.asarray(P, dtype=float)
        if P.ndim == 0:
            P = P.reshape(1, 1)
        elif P.ndim == 1:
            P = P.reshape(-1, 1)
        elif P.ndim != 2:
            raise ValueError(f"Parameters must be (n,) or (n, d); got shape {P.shape}.")
        return P

    @staticmethod
    def _stack_snapshots(A_data):
        """Flatten snapshots of any common shape into (n, D); returns (B, shape).
        Accepts a list of arrays or a 2-D array whose rows are snapshots."""
        arrs = [np.asarray(a, dtype=float) for a in A_data]
        if not arrs:
            raise ValueError("A_data is empty.")
        shape = arrs[0].shape
        for a in arrs:
            if a.shape != shape:
                raise ValueError(f"All snapshots must share one shape; got {shape} and {a.shape}.")
        B = np.stack([a.ravel() for a in arrs], axis=0)
        return B, shape

    @staticmethod
    def _median_pairwise_distance(P, max_points=512, seed=0):
        """Median nonzero pairwise distance — heuristic RBF length scale."""
        if len(P) > max_points:
            rng = np.random.default_rng(seed)
            P = P[rng.choice(len(P), max_points, replace=False)]
        d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        d = np.sqrt(d2[np.triu_indices(len(P), k=1)])
        d = d[d > 0]
        return float(np.median(d)) if d.size else 1.0

    def _setup_kernel(self, P):
        if self.kernel is None:
            ls = self.length_scale if self.length_scale else self._median_pairwise_distance(P)
            self.length_scale_ = ls
            self._k = lambda p, q: rbf_kernel(p, q, length_scale=ls)
            self._kernel_vectorized = True
        else:
            self._k = self.kernel
            self._kernel_vectorized = self._detect_vectorized(P)

    def _detect_vectorized(self, P):
        """True if the kernel accepts a batched second argument and agrees
        with pointwise evaluation."""
        if len(P) < 2:
            return False
        try:
            r = np.asarray(self._k(P[0], P[:2]), dtype=float)
            if r.shape != (2,):
                return False
            r0 = float(self._k(P[0], P[0]))
            r1 = float(self._k(P[0], P[1]))
            return bool(np.allclose(r, [r0, r1], rtol=1e-10, atol=1e-12))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Linear algebra core
    # ------------------------------------------------------------------
    def _kernel_row(self, p, Q):
        """k(p) against the points in Q: shape (len(Q),)."""
        if Q is None or len(Q) == 0:
            return np.zeros(0, dtype=float)
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        if self._kernel_vectorized:
            return np.asarray(self._k(p, Q), dtype=float).ravel()
        return np.array([self._k(p, q) for q in Q], dtype=float)

    def _chol_append(self, k_vec, kpp):
        """
        Append one row/col to the Cholesky factor L of K_reg = K_sel + reg*I.
        Returns False (without modifying L) if the new point is numerically
        dependent on the current selection, so the caller can stop cleanly
        instead of continuing with a garbage factorization.
        """
        d_first = kpp + self.reg
        if self.L is None:
            if d_first <= 0.0:
                return False
            self.L = np.array([[np.sqrt(d_first)]], dtype=float)
            return True

        y = _solve_lower(self.L, k_vec)               # L y = k
        diag_sq = float(kpp + self.reg - y @ y)
        # A duplicate of a selected point yields diag_sq ~ 2*reg; below ~4*reg
        # the regularization noise floor dominates and appending the point
        # would only degrade conditioning without improving the fit.
        if diag_sq <= max(4.0 * self.reg, 1e-12 * d_first):
            return False

        m = self.L.shape[0]
        L_new = np.zeros((m + 1, m + 1), dtype=float)
        L_new[:m, :m] = self.L
        L_new[m, :m] = y
        L_new[m, m] = np.sqrt(diag_sq)
        self.L = L_new
        return True

    def _solve_for_C(self):
        """Solve (K_sel + reg*I) C = B_sel via the Cholesky factor."""
        B_sel = self.B_all[self.selected_idx, :]      # (m, D)
        Z = _solve_lower(self.L, B_sel)
        self.C = _solve_upper(self.L.T, Z)            # (m, D)

    def _power_function(self, p):
        """RKHS power function P(p) = sqrt(K(p,p) - k^T (K_reg)^{-1} k),
        stabilized with the regularized Gram matrix."""
        kpp = float(self._k(p, p))
        if self.L is None:
            return np.sqrt(max(kpp, 0.0))
        v = _solve_lower(self.L, self._kernel_row(p, self.P_sel))
        return np.sqrt(max(kpp - float(v @ v), 0.0))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, A_data, P_train, max_basis_size=None, tol=1e-8, verbose=False,
            method="grim", n_per_step=1):
        """
        Fit the interpolator on the training set.

        Parameters
        ----------
        A_data : list of arrays (any common shape) or 2-D array (n, D)
            Snapshots A(p_i); rows of a 2-D array are treated as snapshots.
        P_train : array-like, shape (n,) or (n, d)
            Parameter (or x/time) values for each snapshot.
        max_basis_size : int or None
            Maximum number of support points (default: all n).
        tol : float
            method="grim": stop when the largest absolute residual of the
            matched linear functionals' pool (i.e. max entry-wise training
            error) falls below tol.
            method="pgreedy": stop when the best selection score falls below
            tol (with use_residual_weight=True the score is power * residual
            norm, so tol carries data units as well).
        method : {"grim", "pgreedy"}
            "grim" (default) is the Banach GRIM algorithm of Lyons & McLeod
            (arXiv:2205.07495): the dense kernel interpolant is thinned by
            recombination against a greedily grown set of evaluation
            functionals. "pgreedy" is f*P-greedy kernel interpolation
            (power-function selection, no recombination).
        n_per_step : int
            method="grim" only: number of new functionals added per greedy
            extension step. Default 1.
        """
        P = self._as_param_matrix(P_train)
        self.B_all, self.snapshot_shape = self._stack_snapshots(A_data)
        if len(self.snapshot_shape) == 2:             # kept for back-compat
            self.M, self.N = self.snapshot_shape
        if self.center:
            self._B_mean = self.B_all.mean(axis=0)
            self.B_all = self.B_all - self._B_mean
        else:
            self._B_mean = None
        n = self.B_all.shape[0]
        if P.shape[0] != n:
            raise ValueError(f"len(A_data)={n} must match len(P_train)={P.shape[0]}.")

        self._setup_kernel(P)
        m_max = n if max_basis_size is None else min(int(max_basis_size), n)

        self.selected_idx = []
        self.P_sel = None
        self.L = None
        self.C = None

        if method == "grim":
            return self._fit_grim(P, m_max, tol, n_per_step, verbose)
        if method != "pgreedy":
            raise ValueError(f"Unknown method '{method}'; use 'grim' or 'pgreedy'.")
        return self._fit_pgreedy(P, m_max, tol, verbose)

    def _fit_grim(self, P, m_max, tol, n_per_step, verbose):
        """
        Banach GRIM (arXiv:2205.07495): write the dense kernel interpolant as
        phi = sum_i e_i with unit (positive) weights over the atoms
        e_i = k(., p_i) C_i, where C solves the full regularized kernel
        system. Then alternate:
          * recombination thinning: nonnegative reweighting of the atoms with
            support <= |L|+1 that exactly preserves the matched functionals
            L (plus total mass, via the all-ones row);
          * greedy extension: add the n_per_step evaluation functionals
            sigma_{j,d}(phi - u) with the largest absolute residual.
        Functional pool: evaluation of snapshot component d at training
        point p_j, for all (j, d).
        """
        n, D = self.B_all.shape

        # Dense kernel interpolant: atoms e_i = k(., p_i) C_i, weights a = 1
        K = np.stack([self._kernel_row(P[i], P) for i in range(n)])  # (n, n)
        C = np.linalg.solve(K + self.reg * np.eye(n), self.B_all)   # (n, D)
        Phi_eval = K @ C          # phi evaluated on the functional pool
        a = np.ones(n)

        # Initial functional: argmax |sigma(phi)| (u_0 = 0)
        j0, d0 = np.unravel_index(int(np.argmax(np.abs(Phi_eval))), (n, D))
        matched = [(int(j0), int(d0))]
        rows = [np.ones(n), K[j0] * C[:, d0]]

        b = a
        max_resid = np.inf
        # |L| functionals give support <= |L|+1, so cap |L| at m_max - 1
        max_functionals = max(1, m_max - 1)

        while True:
            b = recombination_thinning(np.vstack(rows), a)
            U_eval = K @ (b[:, None] * C)
            R = np.abs(Phi_eval - U_eval)
            max_resid = float(R.max())

            if verbose:
                print(f"[GRIM] |L| = {len(matched)}, support = "
                      f"{int(np.count_nonzero(b))}, max residual = {max_resid:.3e}")

            if max_resid <= tol or len(matched) >= max_functionals:
                break

            # Greedy extension: add the worst unmatched functionals
            R_masked = R.copy()
            for j, d in matched:
                R_masked[j, d] = -1.0
            order = np.argsort(R_masked.ravel())[::-1]
            n_add = min(n_per_step, max_functionals - len(matched))
            for flat in order[:n_add]:
                j, d = np.unravel_index(int(flat), (n, D))
                matched.append((int(j), int(d)))
                rows.append(K[j] * C[:, d])

        support = np.flatnonzero(b > 0.0)
        self.selected_idx = support.tolist()
        self.P_sel = P[support]
        self.C = b[support, None] * C[support]
        self.n_functionals_ = len(matched)
        self.matched_functionals_ = matched
        self.max_residual_ = max_resid

        if verbose:
            print(f"[GRIM] done: {len(support)} support points, "
                  f"{len(matched)} matched functionals, "
                  f"max residual = {max_resid:.3e}")
        return self

    def _fit_pgreedy(self, P, m_max, tol, verbose):
        """f*P-greedy kernel interpolation (power-function selection with
        incremental Cholesky; no recombination)."""
        n = self.B_all.shape[0]
        Kdiag = np.array([float(self._k(P[i], P[i])) for i in range(n)])
        selected = set()

        for it in range(m_max):
            cand = [i for i in range(n) if i not in selected]
            if not cand:
                break

            # Scores for all candidates at once
            if self.L is None:
                power = np.sqrt(np.maximum(Kdiag[cand], 0.0))
                Pred = np.zeros((len(cand), self.B_all.shape[1]))
                K_cand = None
            else:
                K_cand = np.stack([self._kernel_row(P[i], self.P_sel) for i in cand])
                V = _solve_lower(self.L, K_cand.T)                     # (m, nc)
                power = np.sqrt(np.maximum(Kdiag[cand] - np.sum(V * V, axis=0), 0.0))
                Pred = K_cand @ self.C                                 # (nc, D)

            resid = np.linalg.norm(self.B_all[cand] - Pred, axis=1)
            if self.dual_selector is not None:
                scores = np.array([
                    self.dual_selector(i, A_true_i=self.B_all[i], A_pred_i=Pred[j],
                                       power_i=power[j],
                                       state={"iter": it, "selected": self.selected_idx})
                    for j, i in enumerate(cand)
                ])
            elif self.use_residual_weight:
                scores = power * (resid + 1e-15)
            else:
                scores = power

            j_best = int(np.argmax(scores))
            best_i, best_score = cand[j_best], scores[j_best]

            if not np.isfinite(best_score) or best_score < tol:
                if verbose:
                    print(f"Converged at iter {it}: best score {best_score:.2e} < tol {tol:.2e}")
                break

            k_vec = (self._kernel_row(P[best_i], self.P_sel) if K_cand is None
                     else K_cand[j_best])
            if not self._chol_append(k_vec, Kdiag[best_i]):
                if verbose:
                    print(f"Stopping at iter {it}: point {best_i} is numerically "
                          f"dependent on the current selection.")
                break

            selected.add(best_i)
            self.selected_idx.append(best_i)
            p_star = P[best_i]
            self.P_sel = (p_star[None, :] if self.P_sel is None
                          else np.vstack([self.P_sel, p_star]))
            self._solve_for_C()

            if verbose:
                print(f"[Iter {it}] selected idx = {best_i}, score = {best_score:.3e}")

        return self

    # ------------------------------------------------------------------
    # File-based interface: first column x (or time), remaining columns y
    # ------------------------------------------------------------------
    @staticmethod
    def load_columns(path, delimiter=None, comments="#", skiprows=0):
        """
        Load a numeric table from a text/CSV file. The first column is x
        (or time), the remaining columns are outputs. Lines starting with
        `comments` are ignored; the delimiter is sniffed (comma vs
        whitespace) when not given; a single non-numeric header row is
        tolerated. Returns (x, Y) with shapes (n,) and (n, n_out).
        """
        if delimiter is None:
            with open(path) as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith(comments):
                        continue
                    if "," in s:
                        delimiter = ","
                    break
        try:
            data = np.loadtxt(path, delimiter=delimiter, comments=comments,
                              skiprows=skiprows, ndmin=2)
        except ValueError:
            # Tolerate one non-numeric header row (e.g. CSV column names)
            data = np.loadtxt(path, delimiter=delimiter, comments=comments,
                              skiprows=skiprows + 1, ndmin=2)
        if data.shape[1] < 2:
            raise ValueError(f"{path}: need at least two columns (x + y...).")
        return data[:, 0], data[:, 1:]

    def fit_file(self, path, max_basis_size=None, tol=1e-8, verbose=False,
                 delimiter=None, comments="#", method="pgreedy"):
        """
        Fit y(x) from a data file whose first column is x (or time) and whose
        remaining columns are y values. After fitting, `self(x_query)` returns
        the interpolated y columns, shape (n_query, n_out).

        Defaults to method="pgreedy": data files often have hundreds of rows,
        where the per-iteration recombination of "grim" is much slower for
        little accuracy gain on 1-D curve data.
        """
        x, Y = self.load_columns(path, delimiter=delimiter, comments=comments)
        self.fit(Y, x, max_basis_size=max_basis_size, tol=tol, verbose=verbose,
                 method=method)
        self.x_train_, self.Y_train_ = x, Y
        return self

    @classmethod
    def from_file(cls, path, kernel_func=None, reg=1e-12, max_basis_size=None,
                  tol=1e-8, length_scale=None, verbose=False, **load_kwargs):
        """Convenience constructor: build and fit an interpolator from a file."""
        return cls(kernel_func=kernel_func, reg=reg,
                   length_scale=length_scale).fit_file(
            path, max_basis_size=max_basis_size, tol=tol, verbose=verbose,
            **load_kwargs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, p_query):
        """Evaluate the interpolant at a single point. Returns an array with
        the original snapshot shape (e.g. (M, N), or (n_out,) for file data)."""
        if self.B_all is None:
            raise RuntimeError("Interpolator is not fitted; call fit() or fit_file() first.")
        if self.C is None or not self.selected_idx:
            flat = np.zeros(self.B_all.shape[1])
        else:
            p = np.atleast_1d(np.asarray(p_query, dtype=float))
            flat = self._kernel_row(p, self.P_sel) @ self.C
        if self._B_mean is not None:
            flat = flat + self._B_mean
        return flat.reshape(self.snapshot_shape)

    def evaluate_many(self, P_query):
        """Evaluate at many points. P_query: scalar, (nq,), or (nq, d).
        Returns shape (nq, *snapshot_shape)."""
        P = self._as_param_matrix(P_query)
        return np.stack([self.evaluate(p) for p in P], axis=0)

    __call__ = evaluate_many

    def banach_grim_sparse_matrix_interp(self, A_data, P_train, p_query,
                                         max_basis_size, tol=1e-8, verbose=False):
        """Drop-in wrapper matching the earlier signature: fit then evaluate
        at a single query point."""
        self.fit(A_data, P_train, max_basis_size=max_basis_size, tol=tol,
                 verbose=verbose)
        return self.evaluate(p_query)


# -------------------------
# Command-line interface
# -------------------------
def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Greedy kernel interpolation of a data file whose first "
                    "column is x (or time) and remaining columns are y.")
    ap.add_argument("datafile", help="text/CSV file: first column x, rest y")
    ap.add_argument("--max-basis", type=int, default=None,
                    help="maximum number of selected points (default: all)")
    ap.add_argument("--tol", type=float, default=1e-8, help="greedy stopping tolerance")
    ap.add_argument("--reg", type=float, default=1e-10, help="Tikhonov jitter")
    ap.add_argument("--length-scale", type=float, default=None,
                    help="RBF length scale (default: median pairwise distance)")
    ap.add_argument("-o", "--output", default=None,
                    help="write interpolated table (x, y...) to this file")
    ap.add_argument("--num", type=int, default=None,
                    help="evaluate on a uniform x grid of this many points "
                         "(default: the training x values)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    interp = BanachGRIMInterpolator(reg=args.reg, length_scale=args.length_scale)
    interp.fit_file(args.datafile, max_basis_size=args.max_basis,
                    tol=args.tol, verbose=args.verbose)

    x, Y = interp.x_train_, interp.Y_train_
    err = np.linalg.norm(interp(x) - Y) / max(np.linalg.norm(Y), 1e-300)
    print(f"selected {len(interp.selected_idx)} / {len(x)} points "
          f"(length scale = {getattr(interp, 'length_scale_', args.length_scale):.4g}); "
          f"relative L2 training error = {err:.3e}")

    if args.output:
        xq = (np.linspace(x.min(), x.max(), args.num) if args.num else x)
        table = np.column_stack([xq, interp(xq)])
        np.savetxt(args.output, table)
        print(f"wrote {table.shape[0]} rows x {table.shape[1]} cols to {args.output}")


if __name__ == "__main__":
    _main()


# -------------------------
# Minimal usage examples
# -------------------------
# 1) Matrix snapshots over a parameter space (original use case):
#    A_data = [np.random.randn(5, 7) for _ in range(30)]
#    P_train = np.random.rand(30, 3)
#    grim = BanachGRIMInterpolator(kernel_func=rbf_kernel, reg=1e-10)
#    A_pred = grim.banach_grim_sparse_matrix_interp(
#        A_data, P_train, np.array([0.2, 0.5, 0.8]), max_basis_size=10)
#    print(A_pred.shape)  # -> (5, 7)
#
# 2) Any data file with x in the first column and y in the rest:
#    grim = BanachGRIMInterpolator.from_file("QuarkyoniaEOS.dat", max_basis_size=50)
#    y_query = grim(np.linspace(0.1, 1.0, 200))   # -> (200, n_columns-1)
