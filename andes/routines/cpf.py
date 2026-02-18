"""
Continuation Power Flow (CPF) routine.

Traces the PV curve from a solved base case toward a target loading
using a predictor-corrector method.  The predictor uses a tangent
vector; the corrector solves a standard power flow at fixed lambda.
"""

import logging
from collections import OrderedDict

import numpy as np
from numpy.linalg import norm

from andes.routines.base import BaseRoutine
from andes.shared import matrix, sparse, spmatrix
from andes.utils.misc import elapsed

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Parameterization strategies
# ------------------------------------------------------------------

class NaturalParam:
    """P = lam - lam_prev - step.  Singular at nose."""

    def constraint(self, xy, lam, xy_prev, lam_prev, step, z, nm):
        return lam - lam_prev - step

    def jacobian(self, xy, lam, xy_prev, lam_prev, z, nm):
        dP_row = spmatrix([], [], [], (1, nm), 'd')
        dP_dlam = 1.0
        return dP_row, dP_dlam


class ArcLengthParam:
    """P = ||dx||^2 + dlam^2 - step^2.  Traces full curve."""

    def constraint(self, xy, lam, xy_prev, lam_prev, step, z, nm):
        dxy = xy - xy_prev
        return float(np.dot(dxy, dxy) + (lam - lam_prev)**2 - step**2)

    def jacobian(self, xy, lam, xy_prev, lam_prev, z, nm):
        dxy = xy - xy_prev
        nz = np.flatnonzero(dxy)
        vals = (2.0 * dxy[nz]).tolist()
        dP_row = spmatrix(vals, [0] * len(nz), nz.tolist(),
                          (1, nm), 'd')
        dlam = lam - lam_prev
        dP_dlam = 2.0 * dlam if dlam != 0.0 else 1.0
        return dP_row, dP_dlam


class PseudoArcLengthParam:
    r"""P = z' \* dx + z_lam \* dlam - step.  Most robust at nose."""

    def constraint(self, xy, lam, xy_prev, lam_prev, step, z, nm):
        dxy = xy - xy_prev
        return float(np.dot(z[:nm], dxy) + z[nm] * (lam - lam_prev) - step)

    def jacobian(self, xy, lam, xy_prev, lam_prev, z, nm):
        z_xy = z[:nm]
        nz = np.flatnonzero(z_xy)
        dP_row = spmatrix(z_xy[nz].tolist(), [0] * len(nz), nz.tolist(),
                          (1, nm), 'd')
        dP_dlam = float(z[nm])
        return dP_row, dP_dlam


_PARAM_MAP = {
    'natural': NaturalParam(),
    'arclength': ArcLengthParam(),
    'pseudo_arclength': PseudoArcLengthParam(),
}


class CPF(BaseRoutine):
    """
    Continuation Power Flow routine.

    Traces the nose curve (PV curve) by smoothly increasing load and
    generation from a solved base case toward a target loading condition.

    The continuation parameter lambda interpolates linearly::

        p0(lam) = p0_base + lam * (p0_target - p0_base)

    At lambda=0 the system is at the base case; at lambda=1 the system
    is at the target.  The nose point (maximum lambda) gives the
    steady-state loadability limit.

    Examples
    --------
    Uniform load scaling::

        ss = andes.load('ieee14.raw')
        ss.PFlow.run()
        ss.CPF.run(load_scale=2.0)
        print(ss.CPF.max_lam)

    Per-device target::

        ss.CPF.run(p0_target=my_p_array, q0_target=my_q_array)

    Reactive power limits are enforced through the PV model's built-in
    PV-to-PQ conversion.  Pass ``config_option=["PV.pv2pq=1"]`` to
    ``andes.load()`` to enable Q-limit checking at each corrector step.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((
            ('parameterization', 'pseudo_arclength'),
            ('step', 0.1),
            ('step_min', 1e-4),
            ('step_max', 0.5),
            ('adapt_step', 1),
            ('tol', 1e-6),
            ('max_iter', 20),
            ('max_steps', 500),
            ('stop_at', 'NOSE'),
            ('report', 1),
        )))
        self.config.add_extra("_help",
                              parameterization="continuation method",
                              step="initial continuation step size for lambda",
                              step_min="minimum step size",
                              step_max="maximum step size",
                              adapt_step="enable adaptive step sizing",
                              tol="convergence tolerance for corrector",
                              max_iter="max corrector (NR) iterations per step",
                              max_steps="max continuation steps",
                              stop_at="termination: 'NOSE', 'FULL', or a float lambda",
                              report="write output report",
                              )
        self.config.add_extra("_alt",
                              parameterization=('natural', 'arclength',
                                                'pseudo_arclength'),
                              step="float",
                              step_min="float",
                              step_max="float",
                              adapt_step=(0, 1),
                              tol="float",
                              max_iter=">=1",
                              max_steps=">=1",
                              stop_at=('NOSE', 'FULL', 'float'),
                              report=(0, 1),
                              )

        # --- results ---
        self.converged = False
        self.max_lam = 0.0
        self.lam = None       # 1-D array of lambda values
        self.V = None         # (nbus, nsteps) voltage magnitudes
        self.theta = None     # (nbus, nsteps) voltage angles
        self.steps = None     # 1-D array of step sizes
        self.events = []      # list of event dicts
        self.done_msg = ''

        # --- QV curve results ---
        self.qv_q = None       # 1-D: Q at target bus (load convention, pu)
        self.qv_v = None       # 1-D: V at target bus (pu)
        self.qv_bus = None     # bus idx used in last run_qv()

        # --- internal state ---
        self.models = None
        self._p0_base = None
        self._q0_base = None
        self._pg_base = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def init(self):
        """
        Initialize CPF routine.

        Finds power flow models and resets result storage.
        Requires a converged power flow.

        Returns
        -------
        bool
            True if initialization succeeded.
        """
        system = self.system

        if not system.PFlow.converged:
            logger.error("Power flow has not converged. Run PFlow first.")
            return False

        self.models = system.find_models('pflow')
        self._param = _PARAM_MAP[self.config.parameterization]

        self.converged = False
        self.max_lam = 0.0
        self.lam = None
        self.V = None
        self.theta = None
        self.steps = None
        self.events = []
        self.done_msg = ''
        self.qv_q = None
        self.qv_v = None
        self.qv_bus = None

        return True

    def summary(self):
        out = ['',
               '-> Continuation Power Flow',
               f'{"Parameterization":>16s}: {self.config.parameterization}',
               f'{"Step size":>16s}: {self.config.step}',
               f'{"Adaptive step":>16s}: {"On" if self.config.adapt_step else "Off"}',
               f'{"Stop at":>16s}: {self.config.stop_at}',
               ]
        logger.info('\n'.join(out))

    def run(self, load_scale=None, p0_target=None, q0_target=None,
            pg_target=None, **kwargs):
        """
        Run continuation power flow.

        Parameters
        ----------
        load_scale : float, optional
            Uniform scaling factor for all PQ loads.  ``load_scale=2.0``
            means trace toward 2x the base-case loading.
        p0_target : array-like, optional
            Per-device target active power for PQ loads (system base).
            Length must equal ``system.PQ.n``.
        q0_target : array-like, optional
            Per-device target reactive power for PQ loads (system base).
        pg_target : array-like, optional
            Per-device target active power for PV generators (system base).
            Length must equal ``system.PV.n``.

        Returns
        -------
        bool
            True if CPF terminated normally.
        """
        system = self.system

        if not self.init():
            return False

        self.summary()
        t0, _ = elapsed()

        self._snapshot_base()
        try:
            self._build_targets(load_scale, p0_target, q0_target, pg_target)
            success = self._continuation()
        finally:
            self._restore_base()

        t1, s1 = elapsed(t0)
        self.exec_time = t1 - t0

        if success:
            logger.info('CPF completed in %d steps in %s. max lambda = %.6f',
                        len(self.lam) - 1, s1, self.max_lam)
        else:
            logger.warning('CPF failed. %s', self.done_msg)

        system.exit_code = 0 if success else 1
        return success

    def run_qv(self, bus_idx, q_range=5.0, **kwargs):
        """
        Run QV curve analysis at a specific bus.

        Fixes active power everywhere and varies only reactive power
        at ``bus_idx`` using the continuation engine.  After completion
        the arrays :attr:`qv_q` and :attr:`qv_v` hold the Q-V curve
        data (load sign convention).

        The existing ``config.stop_at`` is respected.  Pass
        ``stop_at='FULL'`` to trace both branches (needed for true
        reactive power margin).

        Parameters
        ----------
        bus_idx : int or str
            Bus index at which to vary reactive power.  At least one
            PQ device must exist at this bus.
        q_range : float, optional
            Additional reactive power absorption (pu, system base) to
            sweep at the target bus.  Default 5.0.
        stop_at : str or float, optional
            Termination mode override.  If given, temporarily replaces
            ``config.stop_at`` for this call.  Use ``'FULL'`` to trace
            both branches and locate the reactive power margin.
        **kwargs
            Forwarded to :meth:`run`.  The ``load_scale``,
            ``p0_target``, ``q0_target``, and ``pg_target`` arguments
            are not accepted here.

        Returns
        -------
        bool
            True if CPF terminated normally.
        """
        self.qv_q = None
        self.qv_v = None
        self.qv_bus = None

        _blocked = {'load_scale', 'p0_target', 'q0_target', 'pg_target'}
        bad = _blocked & kwargs.keys()
        if bad:
            raise ValueError(
                f"{', '.join(sorted(bad))} cannot be used with run_qv()."
            )

        system = self.system

        pq_bus = np.array(system.PQ.bus.v)
        mask = (pq_bus == bus_idx)

        if not np.any(mask):
            raise ValueError(
                f"No PQ device found at bus {bus_idx}. "
                f"Add a PQ device with p0=0, q0=0 at that bus first."
            )

        q0_base = system.PQ.q0.v.copy()
        q0_target = q0_base.copy()
        n_pq = int(np.sum(mask))
        q0_target[mask] = q0_base[mask] + q_range / n_pq

        q_base_at_bus = float(np.sum(q0_base[mask]))

        stop_at = kwargs.pop('stop_at', None)
        old_stop = self.config.stop_at
        if stop_at is not None:
            self.config.stop_at = stop_at

        try:
            result = self.run(q0_target=q0_target, **kwargs)
        finally:
            self.config.stop_at = old_stop

        if result and self.lam is not None and len(self.lam) > 0:
            bus_uid = system.Bus.idx2uid(bus_idx)
            # Linear in lambda: _build_targets distributes q_range equally
            # across all PQ devices at the bus, so total Q = base + lam * range.
            self.qv_q = q_base_at_bus + self.lam * q_range
            self.qv_v = self.V[bus_uid, :].copy()
            self.qv_bus = bus_idx

        return result

    def report(self):
        """Print CPF summary."""
        if self.lam is None:
            return
        out = ['',
               '-> CPF Report',
               f'{"Converged":>16s}: {self.converged}',
               f'{"Steps":>16s}: {len(self.lam) - 1}',
               f'{"Max lambda":>16s}: {self.max_lam:.6f}',
               f'{"Termination":>16s}: {self.done_msg}',
               ]
        logger.info('\n'.join(out))

    def plot(self, bus_idx, fig=None, ax=None, show=True):
        """
        Plot PV curve for a specific bus.

        Parameters
        ----------
        bus_idx : int or str
            Bus idx to plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for plotting.")
            return

        uid = self.system.Bus.idx2uid(bus_idx)

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.lam, self.V[uid, :], '-o', markersize=3)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(f'|V| at Bus {bus_idx} (pu)')
        ax.set_title('CPF Nose Curve')
        ax.grid(True, alpha=0.3)

        # mark nose point
        nose_idx = np.argmax(self.lam)
        ax.plot(self.lam[nose_idx], self.V[uid, nose_idx], 'r*', markersize=12,
                label=f'Nose ($\\lambda$={self.max_lam:.4f})')
        ax.legend()

        if show:
            plt.show()

        return fig, ax

    def plot_qv(self, fig=None, ax=None, show=True):
        """
        Plot QV curve from the last :meth:`run_qv` call.

        Uses the standard convention: x-axis is voltage magnitude,
        y-axis is reactive power injection (positive = capacitive
        support to the bus, i.e. negated load convention).

        Parameters
        ----------
        fig : matplotlib Figure, optional
        ax : matplotlib Axes, optional
        show : bool, optional
            Call ``plt.show()`` if True (default).

        Returns
        -------
        tuple
            (fig, ax)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for plotting.")
            return

        if self.qv_q is None or self.qv_v is None:
            logger.warning("No QV results. Run run_qv() first.")
            return

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        q_inj = -self.qv_q

        ax.plot(self.qv_v, q_inj, '-o', markersize=3)
        ax.set_xlabel(f'|V| at Bus {self.qv_bus} (pu)')
        ax.set_ylabel('Q injected (pu, system base)')
        ax.set_title(f'QV Curve at Bus {self.qv_bus}')
        ax.grid(True, alpha=0.3)

        nose_idx = np.argmin(q_inj)
        ax.plot(self.qv_v[nose_idx], q_inj[nose_idx], 'r*', markersize=12,
                label=f'Q margin = {-q_inj[nose_idx]:.4f} pu')
        ax.legend()

        if show:
            plt.show()

        return fig, ax

    # ------------------------------------------------------------------
    #  Base/target management
    # ------------------------------------------------------------------

    def _snapshot_base(self):
        """Snapshot base-case parameter values from the solved power flow."""
        system = self.system
        self._p0_base = system.PQ.p0.v.copy()
        self._q0_base = system.PQ.q0.v.copy()
        # PV.p is a ConstService(v_str='p0') evaluated only at init.
        # CPF modifies p.v directly during tracing; p0.v is untouched.
        self._pg_base = system.PV.p.v.copy()
        self._x_base = system.dae.x.copy()
        self._y_base = system.dae.y.copy()

        # Disable pq2z so loads stay constant-P/Q during CPF tracing.
        # The vcmp limiter converts loads to constant-Z below vmin,
        # which creates a false nose at v=vmin instead of the true
        # voltage collapse point.
        self._vcmp_enabled = system.PQ.vcmp.enable
        system.PQ.vcmp.enable = False

    def _build_targets(self, load_scale, p0_target, q0_target, pg_target):
        """Compute target arrays from user input."""
        system = self.system

        if load_scale is not None:
            if any(x is not None for x in (p0_target, q0_target, pg_target)):
                logger.warning("load_scale is set; p0_target/q0_target/pg_target will be ignored.")
            self._p0_target = self._p0_base * load_scale
            self._q0_target = self._q0_base * load_scale
            self._pg_target = self._pg_base * load_scale
        else:
            if p0_target is not None:
                p0_target = np.asarray(p0_target)
                if len(p0_target) != system.PQ.n:
                    raise ValueError(f"p0_target length {len(p0_target)} != PQ.n={system.PQ.n}")
            if q0_target is not None:
                q0_target = np.asarray(q0_target)
                if len(q0_target) != system.PQ.n:
                    raise ValueError(f"q0_target length {len(q0_target)} != PQ.n={system.PQ.n}")
            if pg_target is not None:
                pg_target = np.asarray(pg_target)
                if len(pg_target) != system.PV.n:
                    raise ValueError(f"pg_target length {len(pg_target)} != PV.n={system.PV.n}")

            self._p0_target = p0_target if p0_target is not None else self._p0_base.copy()
            self._q0_target = q0_target if q0_target is not None else self._q0_base.copy()
            self._pg_target = pg_target if pg_target is not None else self._pg_base.copy()

        self._dp0 = self._p0_target - self._p0_base
        self._dq0 = self._q0_target - self._q0_base
        self._dpg = self._pg_target - self._pg_base

        total = norm(self._dp0) + norm(self._dq0) + norm(self._dpg)
        if total < 1e-12:
            logger.warning("Transfer direction is zero. "
                           "Target equals base case.")

    def _set_loading(self, lam):
        """Overwrite model parameters to reflect loading at lambda."""
        system = self.system
        system.PQ.p0.v[:] = self._p0_base + lam * self._dp0
        system.PQ.q0.v[:] = self._q0_base + lam * self._dq0
        system.PV.p.v[:] = self._pg_base + lam * self._dpg

    def _restore_base(self):
        """Restore base-case parameter values, DAE state, and PQ vcmp limiter."""
        system = self.system
        system.PQ.p0.v[:] = self._p0_base
        system.PQ.q0.v[:] = self._q0_base
        system.PV.p.v[:] = self._pg_base
        system.dae.x[:] = self._x_base
        system.dae.y[:] = self._y_base
        system.vars_to_models()
        system.PQ.vcmp.enable = self._vcmp_enabled

    # ------------------------------------------------------------------
    #  Continuation algorithm
    # ------------------------------------------------------------------

    def _continuation(self):
        """
        Main continuation loop with tangent predictor and augmented
        corrector.  Parameterization method is selected via config.
        """
        system = self.system
        dae = system.dae
        n, m = dae.n, dae.m
        nm = n + m

        # parse stop_at
        stop_at = self.config.stop_at
        try:
            stop_at_lam = float(stop_at)
            stop_at_mode = 'LAM'
        except (ValueError, TypeError):
            stop_at_mode = str(stop_at).upper()
            stop_at_lam = None

        lam_list = []
        V_list = []
        theta_list = []
        step_list = []
        self.events = []

        # --- initial point (lam=0, base case already converged) ---
        lam = 0.0
        self._set_loading(lam)
        self._fg_update()

        lam_list.append(lam)
        V_list.append(self._bus_vmag().copy())
        theta_list.append(self._bus_angle().copy())

        xy = np.concatenate([dae.x[:n].copy(), dae.y[:m].copy()])
        self._xy_last = xy.copy()
        self._lam_last = lam

        step = self.config.step
        fail_count = 0
        nose_detected = False
        self._failed = False

        # initial tangent (seed with pure lambda direction, then refine)
        z = np.zeros(nm + 1)
        z[nm] = 1.0
        z = self._compute_tangent(lam, z)

        # previous converged point (for secant orientation)
        xy_prev = None
        lam_prev = None

        for k in range(self.config.max_steps):
            # ---- predictor: tangent step ----
            step_k = step
            xy_pred = xy + step_k * z[:nm]
            lam_pred = lam + step_k * z[nm]

            # clamp toward target lambda
            if stop_at_mode == 'LAM' and abs(z[nm]) > 1e-12:
                s_to_target = (stop_at_lam - lam) / z[nm]
                if 0 < s_to_target < step_k:
                    step_k = s_to_target
                    xy_pred = xy + step_k * z[:nm]
                    lam_pred = lam + step_k * z[nm]

            # apply predicted state
            dae.x[:n] = xy_pred[:n]
            dae.y[:m] = xy_pred[n:]
            system.vars_to_models()

            # dfg/dlam at predicted point
            dfg = self._dfg_dlam(lam_pred)

            # ---- corrector: augmented NR ----
            # Reference point is (xy, lam) — the last converged point
            success, niter, lam_new = self._corrector(
                lam_pred, xy, lam, step_k, z, dfg)

            if not success:
                step = step / 2
                fail_count += 1

                if step < self.config.step_min or fail_count > 10:
                    if not nose_detected:
                        nose_detected = True
                        self.events.append({
                            'step': k + 1, 'type': 'NOSE',
                            'msg': f'Nose point at lambda={lam:.6f}'
                        })

                        if stop_at_mode == 'NOSE':
                            self.done_msg = (f'Nose point at '
                                             f'lambda={lam:.6f}')
                            break

                        # ---- branch switch ----
                        self.events.append({
                            'step': k + 1, 'type': 'BRANCH_SWITCH',
                            'msg': f'Branch switch at lambda={lam:.6f}'
                        })
                        logger.info("Branch switch at step %d, "
                                    "lambda=%.6f", k, lam)
                        z[nm] = -z[nm]  # flip only lambda, not xy
                        step = self.config.step
                        fail_count = 0
                    else:
                        self._failed = True
                        self.done_msg = (f'Corrector failed at '
                                         f'lambda={lam:.6f}')
                        logger.info("Corrector failed on lower branch. "
                                    "Last converged lambda=%.6f", lam)
                        break

                # restore last good point
                dae.x[:n] = self._xy_last[:n]
                dae.y[:m] = self._xy_last[n:]
                system.vars_to_models()
                logger.debug("Step %d: failed at lam=%.4f, "
                             "halving step to %.6f", k, lam_pred, step)
                continue

            # ---- accept step ----
            fail_count = 0

            # save previous point for secant
            xy_prev = xy.copy()
            lam_prev = lam

            xy = np.concatenate([dae.x[:n].copy(), dae.y[:m].copy()])
            lam = lam_new
            self._xy_last = xy.copy()
            self._lam_last = lam

            lam_list.append(lam)
            V_list.append(self._bus_vmag().copy())
            theta_list.append(self._bus_angle().copy())
            step_list.append(step_k)

            logger.debug("Step %d: lam=%.6f converged in %d iters",
                         k, lam, niter)

            # ---- nose detection by lambda decrease ----
            if not nose_detected and lam_prev is not None:
                if lam < lam_prev - 1e-8:
                    nose_detected = True
                    self.events.append({
                        'step': k + 1, 'type': 'NOSE',
                        'msg': f'Nose point at lambda={lam_prev:.6f}'
                    })
                    self.events.append({
                        'step': k + 1, 'type': 'BRANCH_SWITCH',
                        'msg': f'Branch switch at lambda={lam:.6f}'
                    })
                    logger.info("Nose detected at step %d, lambda=%.6f",
                                k, lam)

                    if stop_at_mode == 'NOSE':
                        self.done_msg = (f'Nose point at '
                                         f'lambda={lam_prev:.6f}')
                        break

            # ---- FULL mode: stop when lam returns to 0 on lower branch ----
            if stop_at_mode == 'FULL' and nose_detected and lam_prev is not None:
                if lam <= 0.0:
                    # refine to lam=0 via fixed-lambda NR
                    ok_ref, _ = self._corrector_fixed_lam(0.0)
                    if ok_ref:
                        xy = np.concatenate([dae.x[:n].copy(),
                                             dae.y[:m].copy()])
                        lam = 0.0
                        lam_list[-1] = lam
                        V_list[-1] = self._bus_vmag().copy()
                        theta_list[-1] = self._bus_angle().copy()
                        self.done_msg = 'Full curve traced (returned to lambda=0)'
                        self.events.append({
                            'step': k + 1, 'type': 'TARGET_LAM',
                            'msg': self.done_msg
                        })
                    else:
                        self._failed = True
                        self.done_msg = (
                            f'Full curve traced but refinement to lambda=0 '
                            f'failed; last lambda={lam:.6f}')
                        logger.warning(self.done_msg)
                    break

            # ---- target lambda crossing ----
            if stop_at_mode == 'LAM' and lam_prev is not None:
                if abs(lam_prev - stop_at_lam) > 1e-12:
                    crossed = ((lam_prev - stop_at_lam) *
                               (lam - stop_at_lam) <= 0.0)
                    if crossed:
                        # refine to exact target via fixed-lambda NR
                        ok_ref, _ = self._corrector_fixed_lam(stop_at_lam)
                        if ok_ref:
                            xy = np.concatenate([dae.x[:n].copy(),
                                                 dae.y[:m].copy()])
                            lam = stop_at_lam
                            lam_list[-1] = lam
                            V_list[-1] = self._bus_vmag().copy()
                            theta_list[-1] = self._bus_angle().copy()
                            self.done_msg = (f'Reached target '
                                             f'lambda={stop_at_lam}')
                            self.events.append({
                                'step': k + 1, 'type': 'TARGET_LAM',
                                'msg': self.done_msg
                            })
                        else:
                            self._failed = True
                            self.done_msg = (
                                f'Crossed target lambda={stop_at_lam} but '
                                f'refinement failed; last lambda={lam:.6f}')
                            logger.warning(self.done_msg)
                        break

            # ---- compute new tangent ----
            z_old = z.copy()
            z = self._compute_tangent(lam, z_old)

            # After nose: enforce z_lam < 0 for lower branch
            if nose_detected and z[nm] > 0:
                z[nm] = -z[nm]

            # ---- adaptive step (prediction-error based) ----
            if self.config.adapt_step:
                cpf_error = max(
                    norm(xy - xy_pred, np.inf),
                    abs(lam - lam_pred),
                )
                adapt_tol = 0.05
                damping = 0.7
                step_scale = min(
                    2.0,
                    1.0 + damping * (adapt_tol /
                                     max(cpf_error, 1e-12) - 1),
                )
                step = step * step_scale
                step = max(step, self.config.step_min)
                step = min(step, self.config.step_max)

        else:
            self._failed = True
            self.done_msg = f'Reached max steps ({self.config.max_steps})'

        # ---- finalize ----
        self.lam = np.array(lam_list)
        self.V = np.column_stack(V_list) if V_list else np.empty((0, 0))
        self.theta = (np.column_stack(theta_list) if theta_list
                      else np.empty((0, 0)))
        self.steps = np.array(step_list)
        self.max_lam = (float(np.max(self.lam)) if len(self.lam) > 0
                        else 0.0)
        self.converged = len(self.lam) > 1 and not self._failed

        if self.config.report:
            self.report()

        return self.converged

    # ------------------------------------------------------------------
    #  Augmented predictor-corrector building blocks
    # ------------------------------------------------------------------

    def _dfg_dlam(self, lam, eps=1e-7):
        """Compute dfg/dlam by finite difference at current xy."""
        dae = self.system.dae
        n, m = dae.n, dae.m

        self._set_loading(lam)
        self._fg_update()
        fg0 = np.concatenate([dae.f[:n].copy(), dae.g[:m].copy()])

        self._set_loading(lam + eps)
        self._fg_update()
        fg1 = np.concatenate([dae.f[:n].copy(), dae.g[:m].copy()])

        self._set_loading(lam)
        return (fg1 - fg0) / eps

    def _build_augmented(self, J, dfg_dlam_vec, dP_row, dP_dlam):
        """Build (nm+1) x (nm+1) augmented Jacobian from sparse blocks."""
        nm = J.size[0]

        # dfg_dlam as sparse column (nm, 1)
        nz = np.flatnonzero(dfg_dlam_vec)
        dfg_col = spmatrix(dfg_dlam_vec[nz].tolist(), nz.tolist(),
                           [0] * len(nz), (nm, 1), 'd')

        # dP_dlam as (1, 1) sparse
        dP_corner = spmatrix([float(dP_dlam)], [0], [0], (1, 1), 'd')

        # column-major: sparse([[col1_top, col1_bot], [col2_top, col2_bot]])
        return sparse([[J, dP_row], [dfg_col, dP_corner]])

    def _compute_tangent(self, lam, z_prev):
        """
        Compute normalised tangent vector z of length nm+1.

        Solves  J @ (dxy/dlam) = -dfg/dlam  then forms
        z = [dxy/dlam; 1] / ||z||, oriented by continuity with *z_prev*.
        """
        system = self.system
        dae = system.dae
        n, m = dae.n, dae.m
        nm = n + m

        self._fg_update()
        system.j_update(self._pflow_models())

        if n > 0:
            J = sparse([[dae.fx, dae.gx], [dae.fy, dae.gy]])
        else:
            J = dae.gy

        dfg = self._dfg_dlam(lam)
        dxy_dlam = self._solve_sparse(J, -dfg)

        z = np.zeros(nm + 1)
        if np.any(np.isnan(dxy_dlam)) or np.any(np.isinf(dxy_dlam)):
            # Near singularity — fall back to z_prev with z_lam=0
            if z_prev is not None:
                z[:] = z_prev
                z[nm] = 0.0
            else:
                z[nm] = 1.0
        else:
            z[:nm] = dxy_dlam
            z[nm] = 1.0

        z_norm = norm(z)
        if z_norm > 0:
            z /= z_norm

        # Orient by continuity with previous tangent
        if z_prev is not None and len(z_prev) == len(z):
            if float(np.dot(z, z_prev)) < 0.0:
                z = -z

        return z

    def _corrector(self, lam, xy_prev, lam_prev, step, z, dfg):
        """
        Augmented Newton-Raphson corrector with parameterization constraint.

        Parameters
        ----------
        lam : float
            Predicted lambda (initial guess).
        xy_prev, lam_prev : array, float
            Previous converged point.
        step : float
            Step size for parameterization constraint.
        z : np.ndarray
            Tangent vector (length nm+1).
        dfg : np.ndarray
            dfg/dlam vector (length nm), computed by FD.

        Returns
        -------
        tuple
            (success, niter, lam_final)
        """
        system = self.system
        dae = system.dae
        n, m = dae.n, dae.m
        nm = n + m
        param = self._param

        self._set_loading(lam)

        for niter in range(self.config.max_iter):
            self._fg_update()

            # power flow mismatch
            mis = 0.0
            if n > 0:
                mis = max(mis, np.max(np.abs(dae.f)))
            if m > 0:
                mis = max(mis, np.max(np.abs(dae.g)))

            # parameterization constraint
            xy = np.concatenate([dae.x[:n], dae.y[:m]])
            P_val = param.constraint(xy, lam, xy_prev, lam_prev, step, z, nm)
            mis = max(mis, abs(P_val))

            if mis < self.config.tol:
                return True, niter + 1, lam

            if np.isnan(mis):
                return False, niter + 1, lam

            # build Jacobian
            system.j_update(self._pflow_models())
            if n > 0:
                J = sparse([[dae.fx, dae.gx], [dae.fy, dae.gy]])
            else:
                J = dae.gy

            dP_row, dP_dlam = param.jacobian(
                xy, lam, xy_prev, lam_prev, z, nm)

            J_aug = self._build_augmented(J, dfg, dP_row, dP_dlam)

            # RHS
            fg = np.concatenate([-dae.f[:n], -dae.g[:m]])
            rhs = np.concatenate([fg, [-P_val]])

            inc = self._solve_sparse(J_aug, rhs)

            dae.x[:n] += inc[:n]
            dae.y[:m] += inc[n:nm]
            lam += inc[nm]
            self._set_loading(lam)
            system.vars_to_models()

        return False, self.config.max_iter, lam

    # ------------------------------------------------------------------
    #  Legacy corrector (fixed-lambda NR, used by natural param only)
    # ------------------------------------------------------------------

    def _corrector_fixed_lam(self, lam):
        """
        Standard Newton-Raphson at fixed lambda.

        Sets loading to ``lam``, then iterates NR on the power flow
        equations without changing lambda.  The initial guess is the
        current ``dae.xy``.

        Returns
        -------
        tuple
            (success, niter)
        """
        system = self.system
        dae = system.dae
        n = dae.n
        m = dae.m

        self._set_loading(lam)

        for niter in range(self.config.max_iter):
            self._fg_update()

            mis = 0.0
            if n > 0:
                mis = max(mis, np.max(np.abs(dae.f)))
            if m > 0:
                mis = max(mis, np.max(np.abs(dae.g)))

            if mis < self.config.tol:
                return True, niter + 1

            if np.isnan(mis):
                return False, niter + 1

            system.j_update(self._pflow_models())

            # build residual and Jacobian
            res = np.concatenate([-dae.f[:n], -dae.g[:m]]) if n > 0 \
                else -dae.g[:m].copy()

            if n > 0:
                J = sparse([[dae.fx, dae.gx],
                            [dae.fy, dae.gy]])
            else:
                J = dae.gy

            # solve
            inc = self._solve_sparse(J, res)

            dae.x[:n] += inc[:n]
            dae.y[:m] += inc[n:]
            system.vars_to_models()

        return False, self.config.max_iter

    def _solve_sparse(self, J, rhs):
        """Solve J @ x = rhs using the routine's linear solver."""
        rhs_m = matrix(rhs, (len(rhs), 1), 'd')

        if not self.config.linsolve:
            return np.ravel(self.solver.solve(J, rhs_m))
        else:
            return np.ravel(self.solver.linsolve(J, rhs_m))

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _pflow_models(self):
        """Return the dict of PFlow models."""
        return self.models

    def _fg_update(self):
        """Evaluate residual equations (same as PFlow.fg_update)."""
        system = self.system
        system.dae.clear_fg()
        system.l_update_var(self._pflow_models(), niter=0, err=1.0)
        system.s_update_var(self._pflow_models())
        system.f_update(self._pflow_models())
        system.g_update(self._pflow_models())
        system.l_update_eq(self._pflow_models(), niter=0)
        system.fg_to_dae()

    def _bus_vmag(self):
        """Return current bus voltage magnitudes."""
        return np.array(self.system.Bus.v.v)

    def _bus_angle(self):
        """Return current bus voltage angles."""
        return np.array(self.system.Bus.a.v)
