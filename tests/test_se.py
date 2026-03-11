"""
Tests for the state estimation routine and measurement framework.
"""

import unittest

import numpy as np

import andes
from andes.se.measurement import Measurements, StaticEvaluator
from andes.se.algorithms import lav


class TestSEConvergence(unittest.TestCase):
    """SE convergence tests on IEEE 14-bus."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def test_zero_noise_recovers_pflow(self):
        """With zero noise, SE must recover PFlow solution to machine precision."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_angle(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.add_line_flow(sigma_p=0.02, sigma_q=0.03)

        # Set z = h(x_true) exactly (no noise)
        m.finalize()
        ev = StaticEvaluator(ss, m)
        theta_true = np.array(ss.Bus.a.v, dtype=float)
        Vm_true = np.array(ss.Bus.v.v, dtype=float)
        m.z = ev.h(theta_true, Vm_true)

        ss.SE.run(measurements=m)

        self.assertTrue(ss.SE.converged)
        np.testing.assert_allclose(ss.SE.v_est, Vm_true, atol=1e-10)
        np.testing.assert_allclose(ss.SE.a_est, theta_true, atol=1e-10)

    def test_gaussian_noise_converges(self):
        """With Gaussian noise, SE must converge and estimate within ~2 sigma."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)

        ss.SE.run(measurements=m)

        self.assertTrue(ss.SE.converged)
        v_err = np.max(np.abs(ss.SE.v_est - np.array(ss.Bus.v.v)))
        a_err = np.max(np.abs(ss.SE.a_est - np.array(ss.Bus.a.v)))
        self.assertLess(v_err, 0.05, "Voltage error too large")
        self.assertLess(a_err, 0.1, "Angle error too large")

    def test_default_measurements(self):
        """Default SE.run() with auto-generated measurements converges."""
        ss = self.ss
        self.assertTrue(ss.SE.run())
        self.assertTrue(ss.SE.converged)

    def test_chi_squared(self):
        """Chi-squared test passes on a well-behaved estimation."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)
        ss.SE.run(measurements=m)

        passed, J, threshold, dof = ss.SE.chi_squared_test()
        self.assertGreater(dof, 0)
        self.assertTrue(passed, f"J={J:.4f} > threshold={threshold:.4f}")


class TestSEMultiIsland(unittest.TestCase):
    """Test SE on multi-island systems."""

    def test_two_island_converges(self):
        """Kundur 2-island system: angle refs added per island, SE converges."""
        ss = andes.load(andes.get_case('kundur/kundur_islands.xlsx'),
                        default_config=True)
        ss.PFlow.run()
        self.assertEqual(len(ss.Bus.island_sets), 2)

        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)

        self.assertTrue(ss.SE.run(measurements=m))

    def test_two_island_zero_noise(self):
        """Kundur 2-island with zero noise recovers PFlow."""
        ss = andes.load(andes.get_case('kundur/kundur_islands.xlsx'),
                        default_config=True)
        ss.PFlow.run()

        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_angle(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)

        m.finalize()
        ev = StaticEvaluator(ss, m)
        theta_true = np.array(ss.Bus.a.v, dtype=float)
        Vm_true = np.array(ss.Bus.v.v, dtype=float)
        m.z = ev.h(theta_true, Vm_true)

        ss.SE.run(measurements=m)

        self.assertTrue(ss.SE.converged)
        np.testing.assert_allclose(ss.SE.v_est, Vm_true, atol=1e-10)
        np.testing.assert_allclose(ss.SE.a_est, theta_true, atol=1e-10)


class TestMeasurementValidation(unittest.TestCase):
    """Input validation tests for Measurements."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def test_zero_sigma_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add('Bus', 'v', idx=[1], sigma=0.0)

    def test_negative_sigma_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add('Bus', 'v', idx=[1], sigma=-0.01)

    def test_unsupported_direct_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add('GENCLS', 'delta', sigma=0.01)

    def test_nonexistent_variable_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add('Bus', 'nonexistent', sigma=0.01)

    def test_nonexistent_model_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add('FakeModel', 'v', sigma=0.01)

    def test_injection_zero_sigma_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add_bus_injection(sigma_p=0.0)

    def test_line_flow_zero_sigma_raises(self):
        m = Measurements(self.ss)
        with self.assertRaises(ValueError):
            m.add_line_flow(sigma_q=0.0)


class TestStaticEvaluator(unittest.TestCase):
    """Tests for h(x) and H(x) evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def test_h_voltage_is_identity(self):
        """h(x) for voltage measurements returns Vm directly."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.finalize()

        ev = StaticEvaluator(ss, m)
        theta = np.array(ss.Bus.a.v, dtype=float)
        Vm = np.array(ss.Bus.v.v, dtype=float)

        hx = ev.h(theta, Vm)
        np.testing.assert_allclose(hx, Vm, atol=1e-15)

    def test_h_angle_is_identity(self):
        """h(x) for angle measurements returns theta directly."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_angle(sigma=0.01)
        m.finalize()

        ev = StaticEvaluator(ss, m)
        theta = np.array(ss.Bus.a.v, dtype=float)
        Vm = np.array(ss.Bus.v.v, dtype=float)

        hx = ev.h(theta, Vm)
        np.testing.assert_allclose(hx, theta, atol=1e-15)

    def test_jacobian_shape(self):
        """Numerical Jacobian has correct shape (nm, 2*nb)."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.finalize()

        ev = StaticEvaluator(ss, m)
        theta = np.array(ss.Bus.a.v, dtype=float)
        Vm = np.array(ss.Bus.v.v, dtype=float)

        H = ev.H_numerical(theta, Vm)
        self.assertEqual(H.shape, (m.nm, 2 * ss.Bus.n))

    def test_jacobian_voltage_rows(self):
        """Jacobian rows for V_mag measurements: dV_i/dV_i = 1, rest = 0."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.finalize()

        ev = StaticEvaluator(ss, m)
        theta = np.array(ss.Bus.a.v, dtype=float)
        Vm = np.array(ss.Bus.v.v, dtype=float)

        H = ev.H_numerical(theta, Vm)
        nb = ss.Bus.n

        # dh/dtheta should be ~0 for voltage magnitude measurements
        np.testing.assert_allclose(H[:, :nb], 0, atol=1e-10)
        # dh/dVm should be identity
        np.testing.assert_allclose(H[:, nb:], np.eye(nb), atol=1e-10)


class TestBuildYbus(unittest.TestCase):
    """Tests for the flags.ybus protocol and System.build_ybus()."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def test_discover_line_and_shunt(self):
        """Models with flags.ybus=True must appear in exist.ybus."""
        self.assertIn('Line', self.ss.exist.ybus)
        self.assertIn('Shunt', self.ss.exist.ybus)

    def test_protocol_contract(self):
        """Every model in exist.ybus must implement callable build_ybus()."""
        for name, mdl in self.ss.exist.ybus.items():
            self.assertTrue(
                callable(getattr(mdl, 'build_ybus', None)),
                f"{name} has flags.ybus=True but no callable build_ybus()",
            )

    def test_numeric_correctness(self):
        """system.build_ybus() must match manual line+shunt construction."""
        from andes.linsolvers.scipy import spmatrix_to_csc
        ss = self.ss

        Y_sys = spmatrix_to_csc(ss.build_ybus()).toarray()

        # Manual reference: line contribution only
        Y_ref = spmatrix_to_csc(ss.Line.build_ybus()).toarray()
        # Add shunt contributions
        shunt = ss.Shunt
        for i in range(shunt.n):
            if shunt.u.v[i] == 0:
                continue
            uid = ss.Bus.idx2uid(shunt.bus.v[i])
            Y_ref[uid, uid] += complex(shunt.g.v[i], shunt.b.v[i])

        np.testing.assert_allclose(Y_sys, Y_ref, atol=1e-12)


class TestLAV(unittest.TestCase):
    """Tests for LAV (Least Absolute Value) state estimation."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def test_zero_noise_recovers_pflow(self):
        """LAV with zero noise must recover PFlow solution."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_angle(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.add_line_flow(sigma_p=0.02, sigma_q=0.03)

        m.finalize()
        ev = StaticEvaluator(ss, m)
        theta_true = np.array(ss.Bus.a.v, dtype=float)
        Vm_true = np.array(ss.Bus.v.v, dtype=float)
        m.z = ev.h(theta_true, Vm_true)

        ss.SE.run(measurements=m, algorithm=lav)

        self.assertTrue(ss.SE.converged)
        np.testing.assert_allclose(ss.SE.v_est, Vm_true, atol=1e-6)
        np.testing.assert_allclose(ss.SE.a_est, theta_true, atol=1e-6)

    def test_gaussian_noise_converges(self):
        """LAV with Gaussian noise must converge (IRLS needs more iterations)."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)

        ss.SE.run(measurements=m, algorithm=lav)

        self.assertTrue(ss.SE.converged)
        v_err = np.max(np.abs(ss.SE.v_est - np.array(ss.Bus.v.v)))
        a_err = np.max(np.abs(ss.SE.a_est - np.array(ss.Bus.a.v)))
        self.assertLess(v_err, 0.05, "Voltage error too large")
        self.assertLess(a_err, 0.1, "Angle error too large")

    def test_robust_to_bad_data(self):
        """LAV should be more robust than WLS to gross measurement errors."""
        ss = self.ss
        theta_true = np.array(ss.Bus.a.v, dtype=float)
        Vm_true = np.array(ss.Bus.v.v, dtype=float)

        # Build measurements with a gross error
        m_wls = Measurements(ss)
        m_wls.add_bus_voltage(sigma=0.01)
        m_wls.add_bus_angle(sigma=0.01)
        m_wls.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m_wls.add_line_flow(sigma_p=0.02, sigma_q=0.03)
        m_wls.finalize()
        ev = StaticEvaluator(ss, m_wls)
        m_wls.z = ev.h(theta_true, Vm_true).copy()

        # Inject a 20-sigma outlier into one voltage measurement
        m_wls.z[0] += 20 * m_wls.sigma[0]

        # Build identical measurements for LAV
        m_lav = Measurements(ss)
        m_lav.add_bus_voltage(sigma=0.01)
        m_lav.add_bus_angle(sigma=0.01)
        m_lav.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m_lav.add_line_flow(sigma_p=0.02, sigma_q=0.03)
        m_lav.finalize()
        m_lav.z = m_wls.z.copy()
        m_lav.sigma = m_wls.sigma.copy()

        # Run WLS
        ss.SE.run(measurements=m_wls)
        self.assertTrue(ss.SE.converged, "WLS must converge for comparison")
        wls_v_err = np.max(np.abs(ss.SE.v_est - Vm_true))

        # Run LAV
        ss.SE.run(measurements=m_lav, algorithm=lav)
        self.assertTrue(ss.SE.converged, "LAV must converge")
        lav_v_err = np.max(np.abs(ss.SE.v_est - Vm_true))

        # LAV should produce smaller voltage error than WLS
        self.assertLess(lav_v_err, wls_v_err,
                        f"LAV error ({lav_v_err:.6f}) should be less than "
                        f"WLS error ({wls_v_err:.6f}) with bad data")

    def test_returns_correct_keys(self):
        """LAV result dict must have the same keys as WLS."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)

        ss.SE.run(measurements=m, algorithm=lav)

        expected_keys = {'x_est', 'converged', 'n_iter', 'residuals', 'J', 'gain_matrix'}
        self.assertEqual(set(ss.SE.result.keys()), expected_keys)

    def test_chi_squared_rejected_after_lav(self):
        """Chi-squared test must return passed=False after LAV."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)

        ss.SE.run(measurements=m, algorithm=lav)

        passed, J, threshold, dof = ss.SE.chi_squared_test()
        self.assertFalse(passed,
                         "chi_squared_test must reject after non-WLS run")
        self.assertEqual(threshold, float('inf'),
                         "threshold must be inf when chi-squared is inapplicable")


class TestAlgorithmEdgeCases(unittest.TestCase):
    """Edge case tests for SE algorithms."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def test_wls_max_iter_zero(self):
        """WLS with max_iter=0 returns not-converged without crashing."""
        from andes.se.algorithms import wls

        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)
        m.finalize()

        ev = StaticEvaluator(ss, m)
        x0 = np.concatenate([np.array(ss.Bus.a.v, dtype=float),
                             np.array(ss.Bus.v.v, dtype=float)])

        result = wls(ev, x0, max_iter=0)
        self.assertFalse(result['converged'])
        self.assertIsNone(result['gain_matrix'])

    def test_lav_max_iter_zero(self):
        """LAV with max_iter=0 returns not-converged without crashing."""
        ss = self.ss
        m = Measurements(ss)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)
        m.finalize()

        ev = StaticEvaluator(ss, m)
        x0 = np.concatenate([np.array(ss.Bus.a.v, dtype=float),
                             np.array(ss.Bus.v.v, dtype=float)])

        result = lav(ev, x0, max_iter=0)
        self.assertFalse(result['converged'])
        self.assertIsNone(result['gain_matrix'])


if __name__ == '__main__':
    unittest.main()
