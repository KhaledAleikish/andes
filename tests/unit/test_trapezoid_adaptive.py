import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from andes.routines.daeint import TrapezoidAdaptive


class _DummyDAE:
    def __init__(self):
        self.n = 1
        self.x = np.array([1.0])
        self.y = np.array([2.0])
        self.f = np.array([3.0])
        self.t = 0.0


class _DummySystem:
    def __init__(self):
        self.dae = _DummyDAE()

    def vars_to_models(self):
        pass


class _DummyTDS:
    def __init__(self):
        self.system = _DummySystem()
        self.config = SimpleNamespace(abstol=1e-6, reltol=1e-3)
        self.h = 0.1
        self.deltat = self.h
        self.deltatmax = 1.0
        self.deltatmin_adapt = 1e-4
        self.niter = 7
        self._adaptive_nolte_steps = 0
        self._last_switch_t = -999.0
        self.converged = True
        self.last_converged = True
        self.method = SimpleNamespace(nolte_event_window=0.0)
        self.x0 = np.zeros(1)
        self.y0 = np.zeros(1)
        self.f0 = np.zeros(1)


class TestTrapezoidAdaptiveUnit(unittest.TestCase):

    def test_nolte_next_h_applies_min_floor(self):
        tds = _DummyTDS()
        tds.niter = 1
        h_new = TrapezoidAdaptive._nolte_next_h(tds, 1e-6)
        self.assertGreaterEqual(h_new, tds.deltatmin_adapt)

    def test_reject_restores_state_and_snapshots(self):
        tds = _DummyTDS()
        dae = tds.system.dae

        state0 = (dae.x.copy(), dae.y.copy(), dae.f.copy())
        dae.x[:] = 9.0
        dae.y[:] = 8.0
        dae.f[:] = 7.0
        tds.x0[:] = 6.0
        tds.y0[:] = 5.0
        tds.f0[:] = 4.0

        TrapezoidAdaptive._reject(tds, h_next=0.01, state=state0)

        np.testing.assert_allclose(dae.x, [1.0])
        np.testing.assert_allclose(dae.y, [2.0])
        np.testing.assert_allclose(dae.f, [3.0])
        np.testing.assert_allclose(tds.x0, dae.x)
        np.testing.assert_allclose(tds.y0, dae.y)
        np.testing.assert_allclose(tds.f0, dae.f)

    def test_lte_reject_sets_short_recovery_window(self):
        tds = _DummyTDS()
        dae = tds.system.dae
        x_start = dae.x.copy()
        y_start = dae.y.copy()
        f_start = dae.f.copy()
        call_count = {'k': 0}

        def fake_solve_once(t, h, method):
            call_count['k'] += 1
            if call_count['k'] == 1:
                dae.x[:] = 10.0
                dae.y[:] = 20.0
                dae.f[:] = 30.0
                return True
            if call_count['k'] == 2:
                dae.x[:] = 11.0
                dae.y[:] = 21.0
                dae.f[:] = 31.0
                return True
            dae.x[:] = 12.0
            dae.y[:] = 22.0
            dae.f[:] = 32.0
            return True

        with patch('andes.routines.daeint.ImplicitIter.solve_once', side_effect=fake_solve_once), \
                patch('andes.routines.daeint.accept_reject', return_value=(False, 0.01, 0)):
            ok = TrapezoidAdaptive.step(tds)

        self.assertFalse(ok)
        self.assertEqual(tds._adaptive_nolte_steps, 20)
        np.testing.assert_allclose(dae.x, x_start)
        np.testing.assert_allclose(dae.y, y_start)
        np.testing.assert_allclose(dae.f, f_start)

    def test_half_step_failure_rolls_back_state(self):
        tds = _DummyTDS()
        dae = tds.system.dae
        x_start = dae.x.copy()
        y_start = dae.y.copy()
        f_start = dae.f.copy()
        call_count = {'k': 0}

        def fake_solve_once(t, h, method):
            call_count['k'] += 1
            if call_count['k'] == 1:
                dae.x[:] = 10.0
                dae.y[:] = 20.0
                dae.f[:] = 30.0
                return True
            return False

        with patch('andes.routines.daeint.ImplicitIter.solve_once', side_effect=fake_solve_once):
            ok = TrapezoidAdaptive.step(tds)

        self.assertFalse(ok)
        np.testing.assert_allclose(dae.x, x_start)
        np.testing.assert_allclose(dae.y, y_start)
        np.testing.assert_allclose(dae.f, f_start)

    def test_algebraic_only_fallback_path(self):
        tds = _DummyTDS()
        tds.system.dae.n = 0

        with patch('andes.routines.daeint.ImplicitIter.solve_once', return_value=True):
            ok = TrapezoidAdaptive.step(tds)

        self.assertTrue(ok)
        self.assertEqual(tds.deltat, tds.h)


if __name__ == '__main__':
    unittest.main()
