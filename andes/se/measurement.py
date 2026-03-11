"""
Measurement containers and evaluators for state estimation.
"""

import logging

import numpy as np

from andes.linsolvers.scipy import spmatrix_to_csc

logger = logging.getLogger(__name__)


def gaussian_noise(sigma, size, rng):
    """Default Gaussian noise: e ~ N(0, sigma^2)."""
    return rng.normal(0, sigma, size)


class Measurements:
    """
    Measurement data container for state estimation.

    Each measurement specifies what is measured (model + variable or a computed
    quantity), where (device idx), the observed value, and its standard
    deviation.

    The core method is ``add()``, which accepts an ANDES model name and
    variable name.  Convenience wrappers like ``add_bus_voltage()`` delegate
    to it.

    Parameters
    ----------
    system : andes.system.System
        The ANDES system instance (must have completed ``setup()``).
    """

    def __init__(self, system):
        self.system = system

        # Parallel lists -- one entry per scalar measurement
        self._models = []      # str model name
        self._vars = []        # str variable name
        self._idx = []         # device idx within that model
        self._sigma = []       # standard deviation
        self._kind = []        # 'direct' | 'p_inj' | 'q_inj' | 'p_flow' | 'q_flow'

        # Populated by finalize()
        self.z = None          # (nm,) measured values
        self.sigma = None      # (nm,) standard deviations
        self._finalized = False

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sigma(sigma, name='sigma'):
        """Raise ValueError if any sigma value is non-positive."""
        arr = np.atleast_1d(np.asarray(sigma, dtype=float))
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be positive.")

    def _resolve_idx(self, model, idx):
        """Normalize idx: None -> all devices, scalar -> [scalar], else list."""
        if idx is None:
            return list(getattr(self.system, model).idx.v)
        if not hasattr(idx, '__len__'):
            return [idx]
        return list(idx)

    def _append(self, model, var, idx, sigma, kind):
        """Append a single scalar measurement."""
        self._models.append(model)
        self._vars.append(var)
        self._idx.append(idx)
        self._sigma.append(float(sigma))
        self._kind.append(kind)

    # ------------------------------------------------------------------
    #  Core add method
    # ------------------------------------------------------------------

    # Measurement types supported by StaticEvaluator as direct lookups
    _SUPPORTED_DIRECT = {('Bus', 'v'), ('Bus', 'a')}

    def add(self, model, var, idx=None, sigma=0.01):
        """
        Add measurements of ``model.var`` for selected devices.

        Parameters
        ----------
        model : str
            ANDES model name (e.g. ``'Bus'``, ``'GENROU'``, ``'Line'``).
        var : str
            Variable name within the model (e.g. ``'v'``, ``'a'``, ``'delta'``).
        idx : array-like or None
            Device indices.  ``None`` selects all devices of the model.
        sigma : float or array-like
            Standard deviation(s) for the measurements.  Must be positive.
        """
        mdl = self.system.__dict__.get(model)
        if mdl is None:
            raise ValueError(f"Model '{model}' not found in system.")

        if not hasattr(mdl, var):
            raise ValueError(f"Variable '{var}' not found on model '{model}'.")

        if (model, var) not in self._SUPPORTED_DIRECT:
            raise ValueError(
                f"Direct measurement '{model}.{var}' is not supported by "
                f"StaticEvaluator. Supported: {self._SUPPORTED_DIRECT}."
            )

        idx_list = self._resolve_idx(model, idx)
        sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (len(idx_list),))
        self._check_sigma(sigma_arr)

        for i, dev_idx in enumerate(idx_list):
            self._append(model, var, dev_idx, sigma_arr[i], 'direct')

        self._finalized = False

    # ------------------------------------------------------------------
    #  Convenience wrappers
    # ------------------------------------------------------------------

    def add_bus_voltage(self, bus_idx=None, sigma=0.01):
        """Add bus voltage magnitude measurements."""
        self.add('Bus', 'v', idx=bus_idx, sigma=sigma)

    def add_bus_angle(self, bus_idx=None, sigma=0.01):
        """Add bus voltage angle measurements (PMU)."""
        self.add('Bus', 'a', idx=bus_idx, sigma=sigma)

    def _add_paired(self, model, idx, p_var, q_var, sigma_p, sigma_q):
        """Add paired active/reactive measurements for each device."""
        self._check_sigma(sigma_p, 'sigma_p')
        self._check_sigma(sigma_q, 'sigma_q')
        for dev_idx in self._resolve_idx(model, idx):
            self._append(model, p_var, dev_idx, float(sigma_p), p_var)
            self._append(model, q_var, dev_idx, float(sigma_q), q_var)
        self._finalized = False

    def add_bus_injection(self, bus_idx=None, sigma_p=0.02, sigma_q=0.03):
        """Add active and reactive power injection measurements.

        These are 'computed' measurements evaluated via Y-bus formulas,
        not direct DAE variable lookups.
        """
        self._add_paired('Bus', bus_idx, 'p_inj', 'q_inj', sigma_p, sigma_q)

    def add_line_flow(self, line_idx=None, sigma_p=0.02, sigma_q=0.03):
        """Add active and reactive power flow measurements (from-end)."""
        self._add_paired('Line', line_idx, 'p_flow', 'q_flow', sigma_p, sigma_q)

    # ------------------------------------------------------------------
    #  Finalize and generate values
    # ------------------------------------------------------------------

    def finalize(self):
        """Convert internal lists to arrays and resolve addresses."""
        self.sigma = np.array(self._sigma, dtype=float)
        if self.z is None:
            self.z = np.zeros(self.nm, dtype=float)
        self._finalized = True

    def generate_from_pflow(self, noise_func=None, seed=None):
        """
        Set measurement values from the converged power flow solution.

        Computes ``z = h(x_true) + noise``.

        Parameters
        ----------
        noise_func : callable or None
            ``noise_func(sigma, size, rng) -> array``.
            Defaults to Gaussian noise.
        seed : int or None
            Random seed for reproducibility.
        """
        if not self.system.PFlow.converged:
            logger.warning("Power flow has not converged. Measurement values "
                           "from generate_from_pflow() may be invalid.")

        if not self._finalized:
            self.finalize()

        if noise_func is None:
            noise_func = gaussian_noise

        rng = np.random.default_rng(seed)

        evaluator = StaticEvaluator(self.system, self)
        h_true = evaluator.h(
            np.array(self.system.Bus.a.v, dtype=float),
            np.array(self.system.Bus.v.v, dtype=float),
        )
        self.z = h_true + noise_func(self.sigma, self.nm, rng)

    @property
    def nm(self):
        """Number of measurements."""
        return len(self._models)


class StaticEvaluator:
    """
    Evaluate measurement functions h(x) and Jacobian H(x) for static SE.

    Uses the Y-bus matrix for power injection and flow calculations.
    Direct variable measurements (voltage magnitude, angle) are simple
    lookups.

    Parameters
    ----------
    system : andes.system.System
    measurements : Measurements
    """

    def __init__(self, system, measurements):
        self.system = system
        self.meas = measurements
        self.nb = system.Bus.n

        # Sparse Y-bus from system-level aggregation (Line + Shunt + ...)
        self.Y = spmatrix_to_csc(system.build_ybus())

        # Precompute per-line parameters as vectorized arrays (all lines)
        line = system.Line
        bus = system.Bus
        self._from_uid = np.array(bus.idx2uid(list(line.bus1.v)))
        self._to_uid = np.array(bus.idx2uid(list(line.bus2.v)))
        self._ys = line.u.v / (line.r.v + 1j * line.x.v)
        self._ysh = line.u.v * (line.g.v + 1j * line.b.v) / 2
        self._y1 = line.u.v * (line.g1.v + 1j * line.b1.v)
        self._tap = np.array(line.tap.v, dtype=float)
        self._phi = np.array(line.phi.v, dtype=float)

        # Classify measurements into numpy index arrays
        self._classify(measurements, bus, line)

    def _classify(self, meas, bus, line):
        """Build numpy index arrays for each measurement type."""
        groups = {'v': ([], []), 'a': ([], []),
                  'p_inj': ([], []), 'q_inj': ([], []),
                  'p_flow': ([], []), 'q_flow': ([], [])}

        for i in range(meas.nm):
            kind = meas._kind[i]
            idx = meas._idx[i]
            if kind == 'direct':
                key = meas._vars[i]  # 'v' or 'a'
                groups[key][0].append(i)
                groups[key][1].append(bus.idx2uid(idx))
            elif kind in ('p_inj', 'q_inj'):
                groups[kind][0].append(i)
                groups[kind][1].append(bus.idx2uid(idx))
            elif kind in ('p_flow', 'q_flow'):
                groups[kind][0].append(i)
                groups[kind][1].append(line.idx2uid(idx))

        self._v_pos, self._v_uid = (np.array(v, dtype=int) for v in groups['v'])
        self._a_pos, self._a_uid = (np.array(v, dtype=int) for v in groups['a'])
        self._pi_pos, self._pi_uid = (np.array(v, dtype=int) for v in groups['p_inj'])
        self._qi_pos, self._qi_uid = (np.array(v, dtype=int) for v in groups['q_inj'])
        self._pf_pos, self._pf_luid = (np.array(v, dtype=int) for v in groups['p_flow'])
        self._qf_pos, self._qf_luid = (np.array(v, dtype=int) for v in groups['q_flow'])

    def _line_S_from(self, V):
        """Compute from-end apparent power for all lines (vectorized)."""
        Vf = V[self._from_uid]
        Vt = V[self._to_uid]
        m = self._tap * np.exp(1j * self._phi)
        tap2 = self._tap ** 2
        I_ij = (Vf / tap2 - Vt / np.conj(m)) * self._ys + Vf / tap2 * (self._ysh + self._y1)
        return Vf * np.conj(I_ij)

    def h(self, theta, Vm):
        """
        Evaluate all measurement functions.

        Parameters
        ----------
        theta : ndarray, shape (nb,)
            Bus voltage angles in radians.
        Vm : ndarray, shape (nb,)
            Bus voltage magnitudes in per unit.

        Returns
        -------
        ndarray, shape (nm,)
            Computed measurement values.
        """
        hx = np.zeros(self.meas.nm, dtype=float)
        V = Vm * np.exp(1j * theta)

        # Direct measurements (vectorized fancy-indexing)
        if len(self._v_pos):
            hx[self._v_pos] = Vm[self._v_uid]
        if len(self._a_pos):
            hx[self._a_pos] = theta[self._a_uid]

        # Power injections: S_i = V_i * conj(sum_j Y_ij V_j)
        if len(self._pi_pos) or len(self._qi_pos):
            S_inj = V * np.conj(self.Y @ V)
            if len(self._pi_pos):
                hx[self._pi_pos] = S_inj[self._pi_uid].real
            if len(self._qi_pos):
                hx[self._qi_pos] = S_inj[self._qi_uid].imag

        # Line flows (from-end, vectorized)
        if len(self._pf_pos) or len(self._qf_pos):
            S_from = self._line_S_from(V)
            if len(self._pf_pos):
                hx[self._pf_pos] = S_from[self._pf_luid].real
            if len(self._qf_pos):
                hx[self._qf_pos] = S_from[self._qf_luid].imag

        return hx

    def H_numerical(self, theta, Vm, eps=1e-5):
        """
        Numerical Jacobian of h(x) via central differences.

        Parameters
        ----------
        theta : ndarray, shape (nb,)
        Vm : ndarray, shape (nb,)
        eps : float
            Perturbation size.

        Returns
        -------
        ndarray, shape (nm, 2*nb)
            Jacobian matrix.  Columns: [dh/dtheta_1..dh/dtheta_nb, dh/dV_1..dh/dV_nb].
        """
        nb = self.nb
        H = np.zeros((self.meas.nm, 2 * nb), dtype=float)

        for j in range(nb):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += eps
            tm[j] -= eps
            H[:, j] = (self.h(tp, Vm) - self.h(tm, Vm)) / (2 * eps)

        for j in range(nb):
            Vp, Vm_ = Vm.copy(), Vm.copy()
            Vp[j] += eps
            Vm_[j] -= eps
            H[:, nb + j] = (self.h(theta, Vp) - self.h(theta, Vm_)) / (2 * eps)

        return H

    def residual(self, theta, Vm):
        """Measurement residual: z - h(x)."""
        return self.meas.z - self.h(theta, Vm)

    def weight_matrix(self):
        """Diagonal weight matrix W = diag(1/sigma^2)."""
        return np.diag(1.0 / self.meas.sigma ** 2)
