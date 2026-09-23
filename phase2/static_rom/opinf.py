"""
Static (closed-form) Operator Inference on POD coefficients -- the fitting
engine behind both arms of Step 4a. Everything here is deterministic:
float64 SVD for the POD basis, block-diagonal ridge least squares for the
regression, no gradient training and no random seeds (see PATH_FORWARD.md
decision 0.3).

Pipeline: pod_basis() -> build_regressors() -> fit_ridge(), with
select_lambda() choosing the two ridge penalties (lam_lin for the linear
block, lam_quad for quadratic + extra blocks; "const" is never penalized)
by an 80/20 contiguous time-block split and a short validation rollout.
discrete_target()/continuous_target() build the two supervised targets
(the one-step map, primary; and the 4th-order-central-difference dz/dt,
with a truncation-error estimate). rollout() replays a fitted model
forward, either by discrete iteration or by RK4 at a fixed dt, with
divergence detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as pt

from .term_library import Library, EXTRA_FAMILY_REGISTRY

LAMBDA_GRID = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)


@dataclass
class PODBasis:
    modes: pt.Tensor    # (n_features, r)
    mean: pt.Tensor     # (n_features,) -- zeros if center=False
    s: pt.Tensor        # (r,) singular values


def pod_basis(X: pt.Tensor, r: int, center: bool = True) -> PODBasis:
    """float64 POD via torch.linalg.svd (not flowTorch's SVD wrapper --
    see PATH_FORWARD.md Step 4a: this module must not depend on flowTorch's
    Gram-matrix SVD path). center=False gives the RAW (uncentred) basis
    used by the DMD-reproduction sanity check in run_step4a.py, matching
    flowTorch's DMD, which is fit on the uncentred data matrix."""
    assert X.dtype == pt.float64, f"pod_basis requires float64 input, got {X.dtype}"
    mean = X.mean(dim=1) if center else pt.zeros(X.shape[0], dtype=X.dtype)
    centered = X - mean.unsqueeze(-1)
    U, S, _Vh = pt.linalg.svd(centered, full_matrices=False)
    return PODBasis(modes=U[:, :r], mean=mean, s=S[:r])


def project(basis: PODBasis, X: pt.Tensor) -> pt.Tensor:
    """Projects (possibly new) snapshots onto the basis: Z = U^T (X - mean)."""
    return basis.modes.T @ (X - basis.mean.unsqueeze(-1))


def reconstruct(basis: PODBasis, Z: pt.Tensor) -> pt.Tensor:
    return basis.modes @ Z + basis.mean.unsqueeze(-1)


def _quadratic_columns(Z: pt.Tensor) -> pt.Tensor:
    """Unique quadratic terms z_i * z_j, i <= j -- r(r+1)/2 rows."""
    r, n = Z.shape
    rows = []
    for i in range(r):
        for j in range(i, r):
            rows.append(Z[i, :] * Z[j, :])
    return pt.stack(rows, dim=0) if rows else pt.zeros((0, n), dtype=Z.dtype)


def build_regressors(Z: pt.Tensor, library: Library, basis_ctx=None) -> tuple[pt.Tensor, dict[str, list[int]]]:
    """Builds the regressor matrix D (p, n_samples) for POD-coefficient
    matrix Z (r, n_samples), and a column_groups map family name -> column
    indices (used by fit_ridge to apply the right penalty to each column,
    and by select_lambda to size the penalty vector). Column order is
    fixed: const, lin, quad, then each of library.extras in order."""
    r, n = Z.shape
    dtype = Z.dtype
    blocks: list[pt.Tensor] = []
    column_groups: dict[str, list[int]] = {}
    col = 0

    if "const" in library.families:
        blocks.append(pt.ones((1, n), dtype=dtype))
        column_groups["const"] = [col]
        col += 1

    if "lin" in library.families:
        blocks.append(Z)
        column_groups["lin"] = list(range(col, col + r))
        col += r

    if "quad" in library.families:
        quad_cols = _quadratic_columns(Z)
        blocks.append(quad_cols)
        column_groups["quad"] = list(range(col, col + quad_cols.shape[0]))
        col += quad_cols.shape[0]

    for name in library.extras:
        fn = EXTRA_FAMILY_REGISTRY[name]
        extra_cols = fn(Z, basis_ctx)
        blocks.append(extra_cols)
        column_groups[name] = list(range(col, col + extra_cols.shape[0]))
        col += extra_cols.shape[0]

    D = pt.cat(blocks, dim=0) if blocks else pt.zeros((0, n), dtype=dtype)
    return D, column_groups


def fit_ridge(D: pt.Tensor, Y: pt.Tensor, lam_lin: float, lam_quad: float,
              column_groups: dict[str, list[int]]) -> pt.Tensor:
    """Closed-form ridge: Coef (r, p) minimizing ||Y - Coef @ D||^2 + sum_j
    penalty_j * Coef[:,j]^2, with penalty 0 for "const" columns, lam_lin
    for "lin" columns, and lam_quad for "quad" and any extra-family
    columns. Falls back to torch.linalg.lstsq when both penalties are
    zero (ordinary least squares, no regularization to add)."""
    p = D.shape[0]
    if lam_lin == 0.0 and lam_quad == 0.0:
        sol = pt.linalg.lstsq(D.T, Y.T).solution   # (p, r)
        return sol.T
    penalty = pt.zeros(p, dtype=D.dtype)
    for family, idxs in column_groups.items():
        if family == "const":
            lam = 0.0
        elif family == "lin":
            lam = lam_lin
        else:
            lam = lam_quad
        for i in idxs:
            penalty[i] = lam
    G = D @ D.T + pt.diag(penalty)
    rhs = D @ Y.T                      # (p, r)
    coef_T = pt.linalg.solve(G, rhs)   # (p, r)
    return coef_T.T


def discrete_target(Z: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor, list[int]]:
    """Primary target: the one-step map z_{k+1} = f(z_k). Returns
    (D_input=Z[:,:-1], Y=Z[:,1:], time_index) where time_index[k] is the
    Z-column each sample's INPUT state came from (used by select_lambda to
    find where the validation block starts in Z's own time axis)."""
    return Z[:, :-1], Z[:, 1:], list(range(Z.shape[1] - 1))


def _fifth_difference_truncation_estimate(Z: pt.Tensor, dt: float) -> float:
    """Estimates the leading truncation-error magnitude of the 4th-order
    central-difference dz/dt formula, (dt^4/30) * z^(5)(t), by approximating
    z^(5) with the discrete binomial 5th-difference operator (needs 6
    consecutive samples) divided by dt^5, and taking the largest magnitude
    over the fit window. This is a magnitude estimate, not an exact bound."""
    n = Z.shape[1]
    if n < 6:
        return float("nan")
    coeffs = pt.tensor([-1.0, 5.0, -10.0, 10.0, -5.0, 1.0], dtype=Z.dtype)
    max_d5 = 0.0
    for k in range(n - 5):
        window = Z[:, k:k + 6]
        d5 = (window * coeffs).sum(dim=1) / (dt ** 5)
        max_d5 = max(max_d5, d5.abs().max().item())
    return (dt ** 4 / 30.0) * max_d5


def continuous_target(Z: pt.Tensor, dt: float) -> tuple[pt.Tensor, pt.Tensor, list[int], float]:
    """4th-order-central-difference dz/dt target, trimming 2 samples off
    each end: dz/dt|_k = (-z[k+2] + 8 z[k+1] - 8 z[k-1] + z[k-2]) / (12 dt).
    Returns (D_input=Z[:,2:-2], dZdt, time_index, truncation_estimate)."""
    n = Z.shape[1]
    if n < 5:
        raise ValueError(f"continuous_target needs >=5 snapshots, got {n}")
    r = Z.shape[0]
    dZdt = pt.zeros((r, n - 4), dtype=Z.dtype)
    for k in range(2, n - 2):
        dZdt[:, k - 2] = (-Z[:, k + 2] + 8 * Z[:, k + 1] - 8 * Z[:, k - 1] + Z[:, k - 2]) / (12.0 * dt)
    time_index = list(range(2, n - 2))
    trunc = _fifth_difference_truncation_estimate(Z, dt)
    return Z[:, 2:n - 2], dZdt, time_index, trunc


def build_target(Z: pt.Tensor, kind: str, dt: float = 0.025):
    if kind == "discrete":
        D_input, Y, time_index = discrete_target(Z)
        return D_input, Y, time_index, None
    if kind == "continuous":
        return continuous_target(Z, dt)
    raise ValueError(f"unknown target kind {kind!r}")


def _rel_l2(pred: pt.Tensor, true: pt.Tensor) -> float:
    denom = pt.linalg.norm(true)
    if denom == 0:
        return float("inf")
    return (pt.linalg.norm(pred - true) / denom).item()


def rollout(coef: pt.Tensor, z0: pt.Tensor, n_steps: int, kind: str, library: Library,
            dt: float = 0.025, max_norm: float = float("inf"), basis_ctx=None
            ) -> tuple[pt.Tensor, bool, int | None]:
    """Replays the fitted model forward n_steps from z0: discrete iteration
    for kind="discrete", RK4 at fixed dt for kind="continuous". Divergence
    (non-finite state, or ||z|| > max_norm) is flagged and the step index
    recorded; the trajectory continues (propagating NaN) rather than
    stopping early, so the caller always gets a full-length tensor."""
    r = z0.shape[0]
    traj = pt.zeros((r, n_steps + 1), dtype=z0.dtype)
    traj[:, 0] = z0
    diverged = False
    divergence_step = None

    def f(z: pt.Tensor) -> pt.Tensor:
        D, _ = build_regressors(z.unsqueeze(1), library, basis_ctx)
        return coef @ D[:, 0]

    z = z0.clone()
    for k in range(n_steps):
        if kind == "discrete":
            z_next = f(z)
        elif kind == "continuous":
            k1 = f(z)
            k2 = f(z + 0.5 * dt * k1)
            k3 = f(z + 0.5 * dt * k2)
            k4 = f(z + dt * k3)
            z_next = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown target kind {kind!r}")

        bad = (not bool(pt.isfinite(z_next).all())) or (z_next.norm().item() > max_norm)
        if bad and not diverged:
            diverged = True
            divergence_step = k + 1
        z = z_next
        traj[:, k + 1] = z
    return traj, diverged, divergence_step


def select_lambda(Z_fit: pt.Tensor, library: Library, kind: str, dt: float = 0.025,
                   grid: tuple[float, ...] = LAMBDA_GRID, val_fraction: float = 0.2,
                   rollout_steps: int = 20, basis_ctx=None) -> tuple[float, float, dict]:
    """Selects (lam_lin, lam_quad) over `grid x grid` by an 80/20
    contiguous time-block split of the fit window, scored by a
    `rollout_steps`-step validation rollout starting at the first
    validation sample. Ties (score within 1e-15) go to the smaller lambda
    pair, deterministic because `grid` is iterated in ascending order and
    the first strictly-better score wins."""
    Z_input_all, Y_all, time_index, _trunc = build_target(Z_fit, kind, dt)
    D_all, column_groups = build_regressors(Z_input_all, library, basis_ctx)
    n = D_all.shape[1]
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    if n_train < 5:
        raise ValueError(f"select_lambda: fit window too short for an 80/20 split (n={n})")

    D_train, Y_train = D_all[:, :n_train], Y_all[:, :n_train]
    val_start = time_index[n_train]

    max_norm_train = Z_fit.norm(dim=0).max().item()
    max_allowed = 1e3 * max_norm_train

    scores: dict[tuple[float, float], float] = {}
    best = None
    for lam_lin in grid:
        for lam_quad in grid:
            coef = fit_ridge(D_train, Y_train, lam_lin, lam_quad, column_groups)
            n_steps = min(rollout_steps, Z_fit.shape[1] - 1 - val_start)
            if n_steps <= 0:
                score = float("inf")
            else:
                z0 = Z_fit[:, val_start]
                traj, diverged, _ = rollout(coef, z0, n_steps, kind, library, dt=dt,
                                             max_norm=max_allowed, basis_ctx=basis_ctx)
                true = Z_fit[:, val_start:val_start + n_steps + 1]
                score = float("inf") if (diverged or not pt.isfinite(traj).all()) else _rel_l2(traj, true)
            scores[(lam_lin, lam_quad)] = score
            if best is None or score < best[0] - 1e-15:
                best = (score, lam_lin, lam_quad)

    return best[1], best[2], {"scores": {f"{k[0]:g},{k[1]:g}": v for k, v in scores.items()},
                               "best_score": best[0], "val_start_time_index": val_start,
                               "n_train": n_train, "n_val": n - n_train}
