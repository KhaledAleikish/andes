"""
Frequency Divider model based on Milano & Ortega (2017).

Estimates bus frequencies algebraically from generator rotor speeds
and network susceptance, without washout filters or numerical
differentiation.
"""

import logging

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from andes.core import ModelData, Model, IdxParam, Algeb
from andes.linsolvers.scipy import spmatrix_to_csc

logger = logging.getLogger(__name__)


class FreqDivData(ModelData):
    """FreqDiv parameter data."""

    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(model='ACNode',
                            info='bus idx',
                            mandatory=True,
                            status_parent=True,
                            )


class FreqDivModel(Model):
    """
    Frequency Divider implementation.

    Computes bus frequencies from the acausal frequency divider formula:

    .. math::

        0 = (B_{BB} + B_{B0}) (f_B - 1) + B_{BG} (\\omega_G - 1)

    where :math:`B_{BB}` is the bus susceptance matrix,
    :math:`B_{B0}` is a diagonal correction for generator internal
    reactances, and :math:`B_{BG}` couples generator rotor speeds
    to bus frequencies.

    **Jacobian strategy**: The initial :math:`B_{sum} = B_{BB} + B_{B0}`
    is stored as ``gyc`` (constant, in ``dae.tpl``).  The ``gxc`` entries
    for :math:`B_{BG}` (generator coupling) are also constant because
    they depend only on generator reactances, not topology.

    When topology or admittance changes occur (e.g., line trip via
    ``Toggle``, shunt switching), ``_B_sum_perm`` is rebuilt so that
    ``g_numeric`` computes the correct residual.  Generator status
    changes also trigger a rebuild of the generator coupling terms.
    The Jacobian (``gyc``/``gxc``) is intentionally kept at its initial
    value: updating it to match the changed topology shifts Newton
    convergence paths, which perturbs the accumulated integration
    trajectory.  The constant Jacobian is a good approximation and
    Newton converges in all tested scenarios.

    Requires one FreqDiv device per bus.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'FreqMeasurement'
        self.flags.update({'tds': True, 'g_num': True,
                           'j_setup': True, 'j_num': True})

        self.f = Algeb(info='bus frequency',
                       tex_name='f',
                       v_str='1',
                       e_str='0',
                       diag_eps=True,
                       )

        # Populated by j_setup
        self._B_sum_perm = None   # scipy csc (n_fd x n_fd)
        self._B_BG_perm = None    # scipy csc (n_fd x n_gen)
        self._gen_omega_a = None  # dae.x addresses of generator omegas
        self._bus2fd = None       # bus_uid -> freqdiv device uid mapping

        # Change tracking (populated by j_setup)
        self._P = None            # scipy permutation matrix (reused on rebuild)
        self._B_B0 = None         # scipy gen correction diagonal
        self._ybus_version = 0    # tracks conn._ybus_version
        self._gen_status_hash = None  # fingerprint of SynGen u.v arrays

    def _collect_gen_data(self):
        """Scan SynGen models and return active generator coupling data.

        Returns
        -------
        gen_bus_uids : np.ndarray (int)
            Bus uid for each active generator.
        gen_omega_a : np.ndarray (int)
            ``dae.x`` address of each active generator's omega state.
        gen_bg : np.ndarray (float)
            ``1/x_G`` admittance for each active generator.
        """
        system = self.system
        gen_bus_uids = []
        gen_omega_a = []
        gen_bg = []

        for mdl_name in system.groups['SynGen'].models:
            mdl = system.__dict__[mdl_name]
            if mdl.n == 0 or not mdl.in_use:
                continue

            for i in range(mdl.n):
                if mdl.u.v[i] == 0:
                    continue

                bus_uid = system.Bus.idx2uid(mdl.bus.v[i])

                # Determine x_G: subtransient > transient > synchronous
                if hasattr(mdl, 'xd2') and hasattr(mdl, 'xq2'):
                    xg = 0.5 * (mdl.xd2.v[i] + mdl.xq2.v[i])
                elif hasattr(mdl, 'xd1'):
                    xg = mdl.xd1.v[i]
                elif hasattr(mdl, 'xs'):
                    xg = mdl.xs.v[i]
                else:
                    logger.warning(
                        'FreqDiv: %s idx=%s has no recognized reactance. '
                        'Skipping.', mdl.class_name, mdl.idx.v[i])
                    continue

                if xg <= 0:
                    logger.warning(
                        'FreqDiv: %s idx=%s has xg=%.4g <= 0. '
                        'Skipping.', mdl.class_name, mdl.idx.v[i], xg)
                    continue

                gen_bus_uids.append(bus_uid)
                gen_omega_a.append(mdl.omega.a[i])
                gen_bg.append(1.0 / xg)

        return (np.array(gen_bus_uids, dtype=int),
                np.array(gen_omega_a, dtype=int),
                np.array(gen_bg))

    def _gen_status_fingerprint(self):
        """Return a hashable fingerprint of SynGen ``u.v`` arrays."""
        parts = []
        for mdl_name in self.system.groups['SynGen'].models:
            mdl = self.system.__dict__[mdl_name]
            if mdl.n == 0 or not mdl.in_use:
                continue
            parts.append(mdl.u.v.tobytes())
        return hash(b''.join(parts))

    def j_setup(self):
        """Build B matrices and store constant Jacobian triplets."""
        system = self.system
        nb = system.Bus.n
        n_fd = self.n

        if n_fd == 0:
            return

        # --- Validate: one FreqDiv per bus, all buses covered ---
        bus_uids = np.array([system.Bus.idx2uid(b) for b in self.bus.v])

        if n_fd != nb:
            raise RuntimeError(
                f'FreqDiv requires exactly one device per bus '
                f'(got {n_fd} devices, system has {nb} buses). '
                f'Add FreqDiv devices for all buses:\n'
                f'  for bus_idx in ss.Bus.idx.v:\n'
                f'      ss.add("FreqDiv", bus=bus_idx)'
            )
        if len(set(bus_uids)) != nb:
            raise RuntimeError(
                'FreqDiv devices must cover all buses without duplicates.'
            )

        # bus_uid -> FreqDiv device uid mapping
        bus2fd = np.empty(nb, dtype=int)
        bus2fd[bus_uids] = np.arange(n_fd)
        self._bus2fd = bus2fd

        # --- B_BB: imaginary part of bus admittance matrix ---
        Y = system.build_ybus()
        B_BB = spmatrix_to_csc(Y.imag())

        # --- Collect generator data ---
        gen_bus_uids, gen_omega_a, gen_bg = self._collect_gen_data()
        n_gen = len(gen_bg)
        self._gen_omega_a = gen_omega_a

        # --- B_B0: diagonal, -1/x_G at generator bus positions ---
        # (duplicate bus_uids sum automatically in COO -> CSC)
        B_B0 = coo_matrix((-gen_bg, (gen_bus_uids, gen_bus_uids)),
                          shape=(nb, nb)).tocsc()

        # --- B_BG: nb x n_gen, +1/x_G at (gen_bus, gen_col) ---
        if n_gen > 0:
            B_BG = coo_matrix((gen_bg, (gen_bus_uids, np.arange(n_gen))),
                              shape=(nb, n_gen)).tocsc()
        else:
            B_BG = csc_matrix((nb, 0))

        B_sum = B_BB + B_B0  # nb x nb

        # --- Permute to FreqDiv device order ---
        # Permutation matrix P: P[fd_uid, bus_uid] = 1
        P = coo_matrix((np.ones(nb), (bus2fd, np.arange(nb))),
                       shape=(nb, nb)).tocsc()

        self._P = P
        self._B_B0 = B_B0
        self._B_sum_perm = P @ B_sum @ P.T  # n_fd x n_fd
        self._B_BG_perm = P @ B_BG          # n_fd x n_gen

        # --- Store Jacobian triplets ---
        f_a = self.f.a  # dae.y addresses for FreqDiv.f

        # gyc (constant): initial B_sum_perm baked into dae.tpl.
        # The Jacobian is intentionally NOT updated during topology changes;
        # see class docstring for rationale.
        Bs_coo = self._B_sum_perm.tocoo()
        if Bs_coo.nnz > 0:
            rows_gy = f_a[Bs_coo.row]
            cols_gy = f_a[Bs_coo.col]
            self.triplets.append_ijv('gyc', rows_gy, cols_gy,
                                     np.array(Bs_coo.data, dtype=float))

        # gxc: d(g)/d(omega_G) = B_BG_perm (topology-invariant)
        if n_gen > 0:
            Bg_coo = self._B_BG_perm.tocoo()
            if Bg_coo.nnz > 0:
                rows_gx = f_a[Bg_coo.row]
                cols_gx = gen_omega_a[Bg_coo.col]
                self.triplets.append_ijv('gxc', rows_gx, cols_gx, Bg_coo.data)

        self._ybus_version = system.conn._ybus_version
        self._gen_status_hash = self._gen_status_fingerprint()

    def _rebuild_B(self):
        """Rebuild ``_B_sum_perm`` from current Ybus (topology or shunt change)."""
        Y = self.system.build_ybus()
        B_BB = spmatrix_to_csc(Y.imag())
        B_sum = B_BB + self._B_B0
        self._B_sum_perm = self._P @ B_sum @ self._P.T
        self._ybus_version = self.system.conn._ybus_version

        logger.debug('FreqDiv: rebuilt B_sum for ybus version %d.',
                     self._ybus_version)

    def _rebuild_gen_coupling(self):
        """Rebuild generator coupling after a generator status change.

        Re-scans SynGen models for active generators and rebuilds
        ``_B_B0``, ``_B_BG_perm``, ``_gen_omega_a``, and
        ``_B_sum_perm``.
        """
        system = self.system
        nb = system.Bus.n

        gen_bus_uids, gen_omega_a, gen_bg = self._collect_gen_data()
        n_gen = len(gen_bg)
        self._gen_omega_a = gen_omega_a

        B_B0 = coo_matrix((-gen_bg, (gen_bus_uids, gen_bus_uids)),
                          shape=(nb, nb)).tocsc()
        self._B_B0 = B_B0

        if n_gen > 0:
            B_BG = coo_matrix((gen_bg, (gen_bus_uids, np.arange(n_gen))),
                              shape=(nb, n_gen)).tocsc()
        else:
            B_BG = csc_matrix((nb, 0))
        self._B_BG_perm = self._P @ B_BG

        # Rebuild B_sum with new B_B0
        Y = system.build_ybus()
        B_BB = spmatrix_to_csc(Y.imag())
        B_sum = B_BB + B_B0
        self._B_sum_perm = self._P @ B_sum @ self._P.T

        self._gen_status_hash = self._gen_status_fingerprint()
        self._ybus_version = system.conn._ybus_version

        logger.debug('FreqDiv: rebuilt gen coupling (%d active generators).',
                     n_gen)

    def _check_rebuild(self):
        """Check if matrices need rebuilding; rebuild if so.

        Returns True if any rebuild occurred.
        """
        if self._B_sum_perm is None:
            return False

        gen_hash = self._gen_status_fingerprint()
        if gen_hash != self._gen_status_hash:
            self._rebuild_gen_coupling()
            return True

        if self.system.conn._ybus_version != self._ybus_version:
            self._rebuild_B()
            return True

        return False

    def j_numeric(self, **kwargs):
        """Detect Ybus or generator-status changes and rebuild matrices.

        The rebuilt matrices are used by ``g_numeric`` for the correct
        residual.  The Jacobian (``gyc``/``gxc``) remains at its initial
        value; see class docstring for rationale.
        """
        self._check_rebuild()

    def g_numeric(self, **kwargs):
        """Compute the frequency divider residual.

        ``self.f.e_str`` is ``'0'`` (placeholder), so ``calls.g`` produces
        zeros and this method is the sole contributor to ``f.e``.  The
        overwrite (not ``+=``) is safe because the symbolic contribution
        is zero.

        Detects Ybus and generator-status changes so the residual is
        correct from the first Newton iteration.
        """
        if self._B_sum_perm is None:
            return

        self._check_rebuild()

        f_dev = self.f.v - 1.0

        if self._gen_omega_a is not None and len(self._gen_omega_a) > 0:
            omega_dev = self.system.dae.x[self._gen_omega_a] - 1.0
            self.f.e[:] = (self._B_sum_perm @ f_dev
                           + self._B_BG_perm @ omega_dev)
        else:
            self.f.e[:] = self._B_sum_perm @ f_dev


class FreqDiv(FreqDivData, FreqDivModel):
    """
    Frequency Divider model.

    Estimates bus frequencies from generator rotor speeds using the
    network susceptance matrix, based on Milano & Ortega (2017).

    Requires one device per bus.  Typical usage::

        ss = andes.load('case.json', setup=False)
        for bus_idx in ss.Bus.idx.v:
            ss.add('FreqDiv', bus=bus_idx)
        ss.setup()
    """

    def __init__(self, system, config):
        FreqDivData.__init__(self)
        FreqDivModel.__init__(self, system, config)
