"""
Shared helpers for adaptive integration methods.
"""

import numpy as np


def weighted_rms_error(err_vec, x_prev, x_new, abstol, reltol, err_wt):
    """
    Compute weighted RMS norm of an error vector.

    Parameters
    ----------
    err_vec : np.ndarray
        Error estimate vector.
    x_prev : np.ndarray
        Previous accepted state.
    x_new : np.ndarray
        Candidate new state.
    abstol : float
        Absolute tolerance.
    reltol : float
        Relative tolerance.
    err_wt : np.ndarray
        Pre-allocated work array.
    """
    np.maximum(np.abs(x_prev), np.abs(x_new), out=err_wt)
    err_wt *= reltol
    err_wt += abstol
    np.divide(err_vec, err_wt, out=err_wt)
    return np.sqrt(np.dot(err_wt, err_wt) / len(err_wt))


def propose_step_factor(err_est, order, safety=0.9, min_factor=0.2, max_factor=5.0):
    """
    Propose a multiplicative step-size factor from a normalized error estimate.
    """
    if err_est <= 0:
        return max_factor

    factor = safety * err_est ** (-1.0 / (order + 1))
    if factor < min_factor:
        return min_factor
    if factor > max_factor:
        return max_factor
    return factor


def candidate_h(err_est, h, order, safety=0.9, min_factor=0.2, max_factor=5.0):
    """
    Propose a candidate next step size from normalized error.
    """
    return h * propose_step_factor(err_est, order,
                                   safety=safety,
                                   min_factor=min_factor,
                                   max_factor=max_factor)


def check_adaptive_bust(tds):
    """
    Shared bust-check for adaptive methods (TrapezoidAdaptive, QNDF).

    Called after ``step()`` has written ``tds.deltat``. If the step was
    rejected and deltat has fallen to the adaptive minimum, marks simulation
    as busted.
    """
    if not tds.converged and tds.deltat <= tds.deltatmin_adapt:
        rejected_h = tds.deltat
        tds.deltat = 0
        tds.busted = True
        tds.err_msg = (
            "Step size below adaptive minimum after rejection "
            f"(deltat={rejected_h:.4g}, dtmin_adapt={tds.deltatmin_adapt:.4g})."
        )


def accept_reject(err_est, h, deltatmax, order,
                  fail_count=0,
                  accept_threshold=1.0,
                  accept_safety=0.9,
                  accept_min_factor=0.2,
                  accept_max_factor=2.0,
                  reject_safety=0.9,
                  reject_min_factor=0.2,
                  reject_max_factor=0.9,
                  repeat_reject_after=1,
                  repeat_reject_factor=0.5):
    """
    Shared accept/reject controller for adaptive methods.

    Returns
    -------
    tuple[bool, float, int]
        ``(accepted, h_next, fail_count_next)``.
    """
    if err_est <= accept_threshold:
        h_next = candidate_h(err_est, h, order,
                             safety=accept_safety,
                             min_factor=accept_min_factor,
                             max_factor=accept_max_factor)
        return True, min(h_next, deltatmax), 0

    fail_count_next = fail_count + 1
    h_next = candidate_h(err_est, h, order,
                         safety=reject_safety,
                         min_factor=reject_min_factor,
                         max_factor=reject_max_factor)
    if fail_count_next > repeat_reject_after:
        h_next = min(h_next, h * repeat_reject_factor)

    return False, min(h_next, deltatmax), fail_count_next
