"""
Tests for the Continuation Power Flow (CPF) routine.
"""

import unittest

import numpy as np

import andes


class TestCPFConvergence(unittest.TestCase):
    """CPF convergence tests on IEEE 14-bus."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()

    def _fresh_ss(self):
        """Load a fresh system with converged PFlow."""
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        return ss

    def test_pseudo_arclength_converges(self):
        """Default pseudo arc-length finds nose on IEEE 14-bus."""
        ss = self._fresh_ss()
        self.assertTrue(ss.CPF.run(load_scale=2.0))
        self.assertGreater(ss.CPF.max_lam, 2.0)
        self.assertLess(ss.CPF.max_lam, 5.0)

    def test_arclength_converges(self):
        """Arc-length parameterization finds nose."""
        ss = self._fresh_ss()
        ss.CPF.config.parameterization = 'arclength'
        self.assertTrue(ss.CPF.run(load_scale=2.0))
        self.assertGreater(ss.CPF.max_lam, 2.0)

    def test_natural_converges(self):
        """Natural parameterization finds nose."""
        ss = self._fresh_ss()
        ss.CPF.config.parameterization = 'natural'
        self.assertTrue(ss.CPF.run(load_scale=2.0))
        self.assertGreater(ss.CPF.max_lam, 2.0)

    def test_all_methods_agree(self):
        """All three methods agree on max_lam within 1%."""
        results = {}
        for param in ['pseudo_arclength', 'arclength', 'natural']:
            ss = self._fresh_ss()
            ss.CPF.config.parameterization = param
            ss.CPF.run(load_scale=2.0)
            results[param] = ss.CPF.max_lam

        lam_vals = list(results.values())
        spread = max(lam_vals) - min(lam_vals)
        mean = np.mean(lam_vals)
        self.assertLess(spread / mean, 0.01,
                        f"Methods disagree: {results}")


class TestCPFStopAt(unittest.TestCase):
    """Tests for stop_at termination modes."""

    def _fresh_ss(self):
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        return ss

    def test_stop_at_lambda(self):
        """stop_at=1.5 reaches exactly lambda=1.5."""
        ss = self._fresh_ss()
        ss.CPF.config.stop_at = 1.5
        ss.CPF.run(load_scale=2.0)
        np.testing.assert_allclose(ss.CPF.lam[-1], 1.5, atol=1e-3)

    def test_stop_at_nose(self):
        """stop_at='NOSE' terminates at nose point."""
        ss = self._fresh_ss()
        ss.CPF.config.stop_at = 'NOSE'
        ss.CPF.run(load_scale=2.0)
        self.assertGreater(ss.CPF.max_lam, 1.0)
        nose_events = [e for e in ss.CPF.events if e['type'] == 'NOSE']
        self.assertGreater(len(nose_events), 0)

    def test_stop_at_full(self):
        """stop_at='FULL' traces upper and lower branch back to lam=0."""
        ss = self._fresh_ss()
        ss.CPF.config.stop_at = 'FULL'
        ss.CPF.config.max_steps = 500
        self.assertTrue(ss.CPF.run(load_scale=2.0))

        lams = ss.CPF.lam
        nose_idx = np.argmax(lams)

        # Must have lower branch points
        self.assertGreater(len(lams), nose_idx + 1,
                           "No lower branch points traced")

        # Last lambda should be at 0
        np.testing.assert_allclose(lams[-1], 0.0, atol=1e-3)

        # Voltages on lower branch should be below upper branch
        V14_upper = ss.CPF.V[13, 0]   # base case
        V14_lower = ss.CPF.V[13, -1]  # lower branch at lam=0
        self.assertLess(V14_lower, V14_upper - 0.1)


class TestCPFLowerBranch(unittest.TestCase):
    """Tests for lower branch tracking."""

    def _fresh_ss(self):
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        return ss

    def test_lower_branch_voltages_decrease(self):
        """Voltages on lower branch are below upper branch at same lambda."""
        ss = self._fresh_ss()
        ss.CPF.config.stop_at = 'FULL'
        ss.CPF.config.max_steps = 500
        ss.CPF.run(load_scale=2.0)

        lams = ss.CPF.lam
        nose_idx = np.argmax(lams)

        # Pick a lambda on the lower branch and find closest upper branch point
        if len(lams) > nose_idx + 5:
            lam_lower = lams[nose_idx + 5]
            V14_lower = ss.CPF.V[13, nose_idx + 5]

            # Find upper branch point at similar lambda
            upper_diffs = np.abs(lams[:nose_idx] - lam_lower)
            closest = np.argmin(upper_diffs)
            V14_upper = ss.CPF.V[13, closest]

            self.assertLess(V14_lower, V14_upper,
                            "Lower branch V should be below upper branch V")

    def test_nose_is_maximum_lambda(self):
        """max_lam is the true maximum over the curve."""
        ss = self._fresh_ss()
        ss.CPF.config.stop_at = 'FULL'
        ss.CPF.config.max_steps = 500
        ss.CPF.run(load_scale=2.0)

        nose_idx = np.argmax(ss.CPF.lam)
        self.assertAlmostEqual(ss.CPF.max_lam, ss.CPF.lam[nose_idx])

        # All lower branch lambdas should be <= max_lam
        self.assertTrue(np.all(ss.CPF.lam <= ss.CPF.max_lam + 1e-6))


class TestCPFVoltageTrajectory(unittest.TestCase):
    """Tests that would have caught past bugs in voltage behavior."""

    def _fresh_ss(self):
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        return ss

    def test_voltages_decrease_on_upper_branch(self):
        """PQ bus voltages must decrease monotonically on the upper branch.

        Catches the old bug where voltages INCREASED toward the nose.
        """
        ss = self._fresh_ss()
        ss.CPF.run(load_scale=2.0)

        lams = ss.CPF.lam
        nose_idx = np.argmax(lams)

        # Bus 14 is a PQ bus — voltage should decrease as load increases
        V14_upper = ss.CPF.V[13, :nose_idx + 1]
        for i in range(1, len(V14_upper)):
            self.assertLess(V14_upper[i], V14_upper[i - 1] + 1e-6,
                            f"V14 increased at step {i}: "
                            f"{V14_upper[i]:.6f} > {V14_upper[i-1]:.6f}")

    def test_nose_voltage_below_vmin(self):
        """Nose point voltage must be well below PQ.vmin (0.8 default).

        Catches the pq2z bug where the vcmp limiter created a false
        nose at exactly v=vmin by converting loads to constant-Z.
        """
        ss = self._fresh_ss()
        vmin = ss.PQ.vmin.v[0]  # typically 0.8
        ss.CPF.run(load_scale=2.0)

        nose_idx = np.argmax(ss.CPF.lam)
        V_nose = np.min(ss.CPF.V[:, nose_idx])  # weakest bus at nose

        self.assertLess(V_nose, vmin - 0.05,
                        f"Nose voltage {V_nose:.4f} suspiciously close to "
                        f"vmin={vmin} — pq2z may not be disabled")

    def test_cpf_matches_fixed_lambda_pflow(self):
        """CPF voltage at a mid-curve point must match a standalone PFlow.

        Catches bugs where the corrector converges to a wrong solution
        (e.g. stale reference point, wrong loading).
        """
        ss = self._fresh_ss()
        ss.CPF.config.stop_at = 1.0
        ss.CPF.run(load_scale=2.0)

        V_cpf = ss.CPF.V[:, -1]
        lam_cpf = ss.CPF.lam[-1]

        # Run a fresh PFlow at the same loading.
        # Must set both PV.p0 (source) and PV.p (ConstService copy)
        # because PFlow init re-evaluates ConstService from p0.
        ss2 = self._fresh_ss()
        ss2.PQ.p0.v[:] *= (1 + lam_cpf)
        ss2.PQ.q0.v[:] *= (1 + lam_cpf)
        ss2.PV.p0.v[:] *= (1 + lam_cpf)
        ss2.PV.p.v[:] *= (1 + lam_cpf)
        ss2.PFlow.run()

        V_pflow = np.array(ss2.Bus.v.v)
        np.testing.assert_allclose(
            V_cpf, V_pflow, atol=1e-4,
            err_msg="CPF voltages don't match standalone PFlow at same loading")

    def test_run_twice_gives_same_result(self):
        """Running CPF twice on the same system produces identical results.

        Catches state-leak bugs where _restore_base fails to clean up.
        """
        ss = self._fresh_ss()

        ss.CPF.run(load_scale=2.0)
        max_lam_1 = ss.CPF.max_lam
        V_1 = ss.CPF.V.copy()

        ss.CPF.run(load_scale=2.0)
        max_lam_2 = ss.CPF.max_lam
        V_2 = ss.CPF.V.copy()

        np.testing.assert_allclose(max_lam_1, max_lam_2, rtol=1e-4,
                                   err_msg="max_lam differs between runs")
        np.testing.assert_allclose(V_1, V_2, atol=1e-6,
                                   err_msg="Voltages differ between runs")


class TestCPFResultStructure(unittest.TestCase):
    """Tests for result array shapes and types."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                            default_config=True)
        cls.ss.PFlow.run()
        cls.v_pflow = np.array(cls.ss.Bus.v.v)
        cls.ss.CPF.run(load_scale=2.0)

    def test_lam_is_1d(self):
        self.assertEqual(self.ss.CPF.lam.ndim, 1)
        self.assertGreater(len(self.ss.CPF.lam), 1)

    def test_V_shape(self):
        """V has shape (nbus, npoints)."""
        nbus = self.ss.Bus.n
        npoints = len(self.ss.CPF.lam)
        self.assertEqual(self.ss.CPF.V.shape, (nbus, npoints))

    def test_theta_shape(self):
        """theta has shape (nbus, npoints)."""
        nbus = self.ss.Bus.n
        npoints = len(self.ss.CPF.lam)
        self.assertEqual(self.ss.CPF.theta.shape, (nbus, npoints))

    def test_lam_starts_at_zero(self):
        self.assertAlmostEqual(self.ss.CPF.lam[0], 0.0)

    def test_base_voltage_matches_pflow(self):
        """First column of V should match PFlow bus voltages."""
        np.testing.assert_allclose(self.ss.CPF.V[:, 0], self.v_pflow,
                                   atol=1e-6)


class TestCPFSafety(unittest.TestCase):
    """Edge cases and safety checks."""

    def test_no_pflow_returns_false(self):
        """CPF without converged PFlow returns False."""
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        self.assertFalse(ss.CPF.run(load_scale=1.5))

    def test_base_case_restored(self):
        """PQ.p0, DAE state, and vcmp limiter are restored after CPF run."""
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        p0_orig = ss.PQ.p0.v.copy()
        x_orig = ss.dae.x.copy()
        y_orig = ss.dae.y.copy()
        vcmp_orig = ss.PQ.vcmp.enable
        ss.CPF.run(load_scale=2.0)
        np.testing.assert_allclose(ss.PQ.p0.v, p0_orig)
        np.testing.assert_allclose(ss.dae.x, x_orig, atol=1e-10,
                                   err_msg="dae.x not restored after CPF")
        np.testing.assert_allclose(ss.dae.y, y_orig, atol=1e-10,
                                   err_msg="dae.y not restored after CPF")
        self.assertEqual(ss.PQ.vcmp.enable, vcmp_orig)

    def test_invalid_target_size_raises(self):
        """Wrong-length p0_target raises ValueError."""
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        with self.assertRaises(ValueError):
            ss.CPF.run(p0_target=[1.0, 2.0])

    def test_invalid_q0_target_size_raises(self):
        """Wrong-length q0_target raises ValueError."""
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        with self.assertRaises(ValueError):
            ss.CPF.run(q0_target=[1.0])

    def test_invalid_pg_target_size_raises(self):
        """Wrong-length pg_target raises ValueError."""
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        with self.assertRaises(ValueError):
            ss.CPF.run(pg_target=[1.0])


class TestCPFCustomTargets(unittest.TestCase):
    """Tests for custom target arrays (non-uniform loading)."""

    def _fresh_ss(self):
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        return ss

    def test_custom_p0_target(self):
        """CPF with explicit p0_target converges."""
        ss = self._fresh_ss()
        p0_target = ss.PQ.p0.v * 2.0
        self.assertTrue(ss.CPF.run(p0_target=p0_target))
        self.assertGreater(ss.CPF.max_lam, 0.5)

    def test_custom_q0_target(self):
        """CPF with explicit q0_target converges."""
        ss = self._fresh_ss()
        p0_target = ss.PQ.p0.v * 2.0
        q0_target = ss.PQ.q0.v * 2.0
        self.assertTrue(ss.CPF.run(p0_target=p0_target, q0_target=q0_target))
        self.assertGreater(ss.CPF.max_lam, 0.5)

    def test_custom_pg_target(self):
        """CPF with explicit pg_target converges."""
        ss = self._fresh_ss()
        p0_target = ss.PQ.p0.v * 2.0
        pg_target = ss.PV.p.v * 2.0
        self.assertTrue(ss.CPF.run(p0_target=p0_target, pg_target=pg_target))
        self.assertGreater(ss.CPF.max_lam, 0.5)

    def test_custom_targets_match_load_scale(self):
        """Custom targets equal to 2x base should match load_scale=2.0."""
        ss1 = self._fresh_ss()
        ss1.CPF.run(load_scale=2.0)

        ss2 = self._fresh_ss()
        ss2.CPF.run(p0_target=ss2.PQ.p0.v * 2.0,
                    q0_target=ss2.PQ.q0.v * 2.0,
                    pg_target=ss2.PV.p.v * 2.0)

        np.testing.assert_allclose(ss1.CPF.max_lam, ss2.CPF.max_lam,
                                   rtol=1e-3)

    def test_zero_direction_does_not_crash(self):
        """Target equal to base (zero direction) should not crash."""
        ss = self._fresh_ss()
        ss.CPF.config.max_steps = 3
        p0_target = ss.PQ.p0.v.copy()
        ss.CPF.run(p0_target=p0_target)
        # Zero direction means loading never changes;
        # lambda grows but system state is unchanged.
        self.assertGreater(len(ss.CPF.lam), 1)

    def test_max_steps_terminates(self):
        """CPF with very few max_steps terminates gracefully."""
        ss = self._fresh_ss()
        ss.CPF.config.max_steps = 3
        ss.CPF.run(load_scale=2.0)
        # Should have exactly 3+1 points (base + 3 steps)
        self.assertLessEqual(len(ss.CPF.lam), 4)
        self.assertIn('max steps', ss.CPF.done_msg.lower())


class TestCPFQLimits(unittest.TestCase):
    """Tests for reactive power limit enforcement via PV.config.pv2pq."""

    def test_pv2pq_reduces_max_lam(self):
        """Enabling pv2pq at load time reduces the loadability limit."""
        # Without Q limits
        ss1 = andes.load(andes.get_case('ieee14/ieee14.raw'),
                         default_config=True)
        ss1.PFlow.run()
        ss1.CPF.run(load_scale=2.0)

        # With Q limits — must use config_option so SortedLimiter.enable=1
        ss2 = andes.load(andes.get_case('ieee14/ieee14.raw'),
                         default_config=True,
                         config_option=["PV.pv2pq=1"])
        ss2.PFlow.run()
        ss2.CPF.run(load_scale=2.0)

        self.assertGreater(ss1.CPF.max_lam, ss2.CPF.max_lam,
                           "Q limits should reduce max_lam")
        self.assertGreater(ss2.CPF.max_lam, 0.0,
                           "CPF with Q limits should still converge")


class TestCPFFailurePaths(unittest.TestCase):
    """Tests that failure paths report failure, not silent success."""

    def _fresh_ss(self):
        ss = andes.load(andes.get_case('ieee14/ieee14.raw'),
                        default_config=True)
        ss.PFlow.run()
        return ss

    def test_max_steps_returns_false(self):
        """Max-steps exhaustion must return False and set converged=False.

        Catches the old bug where len(lam) > 1 was the sole success
        criterion, making max-steps termination look like success.
        """
        ss = self._fresh_ss()
        ss.CPF.config.max_steps = 3
        result = ss.CPF.run(load_scale=2.0)
        self.assertFalse(result)
        self.assertFalse(ss.CPF.converged)
        self.assertEqual(ss.exit_code, 1)

    def test_state_restored_after_build_targets_error(self):
        """vcmp limiter and base-case state must be restored even when
        _build_targets raises ValueError (try/finally guard).
        """
        ss = self._fresh_ss()
        vcmp_orig = ss.PQ.vcmp.enable
        p0_orig = ss.PQ.p0.v.copy()
        y_orig = ss.dae.y.copy()

        with self.assertRaises(ValueError):
            ss.CPF.run(p0_target=[1.0, 2.0])

        self.assertEqual(ss.PQ.vcmp.enable, vcmp_orig,
                         "vcmp limiter not restored after ValueError")
        np.testing.assert_allclose(
            ss.PQ.p0.v, p0_orig,
            err_msg="PQ.p0.v not restored after ValueError")
        np.testing.assert_allclose(
            ss.dae.y, y_orig, atol=1e-10,
            err_msg="dae.y not restored after ValueError")

    def test_conflicting_inputs_warns(self):
        """Passing load_scale together with explicit targets logs a warning."""
        import logging

        ss = self._fresh_ss()
        with self.assertLogs('andes.routines.cpf', level=logging.WARNING) as cm:
            ss.CPF.run(load_scale=2.0,
                       p0_target=ss.PQ.p0.v * 3.0)

        found = any('load_scale' in msg and 'ignored' in msg
                     for msg in cm.output)
        self.assertTrue(found, f"Expected warning about ignored targets, "
                                f"got: {cm.output}")

    def test_corrector_failure_returns_false(self):
        """Corrector failure on lower branch must return False.

        Uses natural parameterization with tiny step limits to force
        failure on the lower branch after the nose.
        """
        ss = self._fresh_ss()
        ss.CPF.config.parameterization = 'natural'
        ss.CPF.config.step_min = 1e-2
        ss.CPF.config.step_max = 0.1
        ss.CPF.config.stop_at = 'FULL'
        ss.CPF.config.max_steps = 200
        result = ss.CPF.run(load_scale=2.0)

        if 'Corrector failed' in ss.CPF.done_msg:
            self.assertFalse(result,
                             "Corrector failure must return False")
            self.assertFalse(ss.CPF.converged)
            self.assertEqual(ss.exit_code, 1)


if __name__ == '__main__':
    unittest.main()
