"""
State estimation algorithms.

Each algorithm is a plain function with a common return signature:
a dict containing ``x_est``, ``converged``, ``n_iter``, ``residuals``, and ``J``.

Users can substitute their own algorithm by passing any callable with the
same signature to ``SE.run(algorithm=my_func)``.
"""

import logging

import numpy as np
from numpy.linalg import solve

logger = logging.getLogger(__name__)


def wls(evaluator, x0, tol=1e-4, max_iter=100):
    """
    Weighted Least Squares state estimation via Gauss-Newton iteration.

    Solves::

        min  (z - h(x))^T  W  (z - h(x))

    using the normal equation::

        G  dx = H^T  W  r

    where ``G = H^T W H`` is the gain matrix and ``r = z - h(x)``.

    Parameters
    ----------
    evaluator : StaticEvaluator
        Provides ``h()``, ``H_numerical()``, ``residual()``, ``weight_matrix()``.
    x0 : ndarray, shape (2*nb,)
        Initial state estimate ``[theta_1..theta_nb, V_1..V_nb]``.
    tol : float
        Convergence tolerance on max absolute state correction.
    max_iter : int
        Maximum number of Gauss-Newton iterations.

    Returns
    -------
    dict
        ``x_est``       : estimated state vector
        ``converged``   : bool
        ``n_iter``      : number of iterations
        ``residuals``   : final measurement residuals z - h(x)
        ``J``           : final objective value (weighted sum of squared residuals)
        ``gain_matrix`` : final gain matrix H^T W H
    """
    nb = evaluator.nb
    x = x0.copy()
    w = 1.0 / evaluator.meas.sigma ** 2  # weight vector: 1/sigma^2
    G = None

    for k in range(max_iter):
        theta = x[:nb]
        Vm = x[nb:]

        r = evaluator.residual(theta, Vm)
        H = evaluator.H_numerical(theta, Vm)

        # Row-scale H and r by weights (avoids nm×nm dense matrix)
        WH = w[:, np.newaxis] * H
        G = H.T @ WH
        rhs = H.T @ (w * r)

        try:
            dx = solve(G, rhs)
        except np.linalg.LinAlgError:
            logger.error("Gain matrix is singular at iteration %d. "
                         "System may be unobservable.", k)
            J = float(np.sum(w * r ** 2))
            return dict(x_est=x, converged=False, n_iter=k + 1,
                        residuals=r, J=J, gain_matrix=G)

        x += dx
        max_dx = np.max(np.abs(dx))

        logger.debug("WLS iter %d: max|dx| = %.6g", k + 1, max_dx)

        if max_dx < tol:
            theta_est = x[:nb]
            Vm_est = x[nb:]
            r = evaluator.residual(theta_est, Vm_est)
            J = float(np.sum(w * r ** 2))
            logger.info("WLS converged in %d iterations, J = %.6g", k + 1, J)
            return dict(x_est=x, converged=True, n_iter=k + 1,
                        residuals=r, J=J, gain_matrix=G)

    # Did not converge
    theta_est = x[:nb]
    Vm_est = x[nb:]
    r = evaluator.residual(theta_est, Vm_est)
    J = float(np.sum(w * r ** 2))
    logger.warning("WLS did not converge after %d iterations, J = %.6g", max_iter, J)
    return dict(x_est=x, converged=False, n_iter=max_iter,
                residuals=r, J=J, gain_matrix=G)


def lav(evaluator, x0, tol=1e-4, max_iter=100):
    """
    Least Absolute Value state estimation via iteratively reweighted
    least squares (IRLS).

    Solves::

        min  sum_i  w_i * |z_i - h_i(x)|

    by iteratively solving reweighted WLS subproblems where each
    measurement weight is scaled by ``1 / |r_i|``.  This gives large
    residuals (bad data) less influence than in standard WLS, making the
    estimator robust to gross measurement errors.

    Parameters
    ----------
    evaluator : StaticEvaluator
        Provides ``h()``, ``H_numerical()``, ``residual()``, ``weight_matrix()``.
    x0 : ndarray, shape (2*nb,)
        Initial state estimate ``[theta_1..theta_nb, V_1..V_nb]``.
    tol : float
        Convergence tolerance on max absolute state correction.
    max_iter : int
        Maximum number of IRLS iterations.

    Returns
    -------
    dict
        Same keys as :func:`wls`:
        ``x_est``, ``converged``, ``n_iter``, ``residuals``, ``J``, ``gain_matrix``.
    """
    nb = evaluator.nb
    x = x0.copy()
    sigma = evaluator.meas.sigma
    w0 = 1.0 / sigma  # LAV base weight: 1/sigma (not 1/sigma^2 as in WLS)

    # Floor for |r_i| to avoid huge weights from near-zero residuals.
    # Scaled by sigma so the floor is meaningful relative to each
    # measurement's noise level.
    eps = 0.01 * sigma
    G = None

    for k in range(max_iter):
        theta = x[:nb]
        Vm = x[nb:]

        r = evaluator.residual(theta, Vm)
        H = evaluator.H_numerical(theta, Vm)

        # Reweight: scale each base weight by 1/max(|r_i|, eps_i)
        abs_r = np.maximum(np.abs(r), eps)
        w = w0 / abs_r

        # Row-scale H and r by weights (avoids nm×nm dense matrix)
        WH = w[:, np.newaxis] * H
        G = H.T @ WH
        rhs = H.T @ (w * r)

        try:
            dx = solve(G, rhs)
        except np.linalg.LinAlgError:
            logger.error("Gain matrix is singular at iteration %d. "
                         "System may be unobservable.", k)
            J = float(np.sum(np.abs(r) / sigma))
            return dict(x_est=x, converged=False, n_iter=k + 1,
                        residuals=r, J=J, gain_matrix=G)

        x += dx
        max_dx = np.max(np.abs(dx))

        logger.debug("LAV iter %d: max|dx| = %.6g", k + 1, max_dx)

        if max_dx < tol:
            theta_est = x[:nb]
            Vm_est = x[nb:]
            r = evaluator.residual(theta_est, Vm_est)
            J = float(np.sum(np.abs(r) / sigma))
            logger.info("LAV converged in %d iterations, J = %.6g", k + 1, J)
            return dict(x_est=x, converged=True, n_iter=k + 1,
                        residuals=r, J=J, gain_matrix=G)

    # Did not converge
    theta_est = x[:nb]
    Vm_est = x[nb:]
    r = evaluator.residual(theta_est, Vm_est)
    J = float(np.sum(np.abs(r) / sigma))
    logger.warning("LAV did not converge after %d iterations, J = %.6g", max_iter, J)
    return dict(x_est=x, converged=False, n_iter=max_iter,
                residuals=r, J=J, gain_matrix=G)
