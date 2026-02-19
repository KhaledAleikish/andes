"""
Tests for QNDF adaptive step-size integration method.
"""

import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


class TestQNDFSmoke(unittest.TestCase):
    """Basic smoke tests for QNDF method."""

    def test_kundur_qndf(self):
        """Kundur full case with QNDF, no fault, tf=2.0."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.config.tf = 2.0
        ss.TDS.init()
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)

    def test_ieee14_fault_qndf(self):
        """IEEE 14-bus with fault event, QNDF method."""
        ss = andes.load(
            get_case('ieee14/ieee14_fault.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.config.tf = 2.0
        ss.TDS.init()
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)


class TestQNDFConfig(unittest.TestCase):
    """Config and wiring tests."""

    def test_fixt_override(self):
        """method='qndf' should force fixt=0."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.TDS.config.fixt = 1
        ss.TDS.set_method('qndf')
        self.assertEqual(ss.TDS.config.fixt, 0)

    def test_cache_created(self):
        """QNDF cache should be created during init."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.init()

        self.assertIsNotNone(ss.TDS.qndf_cache)
        self.assertEqual(ss.TDS.qndf_cache.n, ss.dae.n)
        self.assertEqual(ss.TDS.qndf_cache.order, 1)


class TestQNDFBehavior(unittest.TestCase):
    """Behavioral tests for QNDF method."""

    def test_order_selection(self):
        """QNDF order controller should work (order >= 1, steps succeed)."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.config.tf = 1.9
        ss.TDS.init()
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)
        cache = ss.TDS.qndf_cache
        # Order should be in valid range
        self.assertGreaterEqual(cache.order, 1)
        self.assertLessEqual(cache.order, 5)
        # Should have accumulated some successful steps
        self.assertGreater(cache.consec_steps, 0)

    def test_event_resets_cache(self):
        """After fault event, cache history should be cleared."""
        ss = andes.load(
            get_case('ieee14/ieee14_fault.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.config.tf = 2.0
        ss.TDS.init()
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)
        # After fault on+off, simulation should still complete successfully
        # and consec_steps should be small (reset by events at t=1.0 and t=1.1)
        self.assertLess(ss.TDS.qndf_cache.consec_steps, 100)

    def test_time_monotonic(self):
        """Stored time series should be strictly monotonically increasing."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.config.tf = 2.0
        ss.TDS.init()
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)
        ts = np.array(ss.dae.ts.t)
        diffs = np.diff(ts)
        self.assertTrue(np.all(diffs > 0),
                        f"Time series not monotonic: min diff = {diffs.min()}")

    def test_accuracy_vs_trapezoid(self):
        """QNDF final bus voltages should match trapezoid within tolerance."""
        # Run with trapezoid (fine step)
        ss_trap = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss_trap.PFlow.run()
        ss_trap.TDS.config.tf = 2.0
        ss_trap.TDS.config.tstep = 1 / 120
        ss_trap.TDS.init()
        ss_trap.TDS.run()

        # Run with QNDF
        ss_qndf = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss_qndf.PFlow.run()
        ss_qndf.TDS.config.method = 'qndf'
        ss_qndf.TDS.config.tf = 2.0
        ss_qndf.TDS.init()
        ss_qndf.TDS.run()

        self.assertEqual(ss_trap.exit_code, 0)
        self.assertEqual(ss_qndf.exit_code, 0)

        # Compare final bus voltages (quiescent case, should be very close)
        v_trap = ss_trap.dae.y[ss_trap.Bus.v.a]
        v_qndf = ss_qndf.dae.y[ss_qndf.Bus.v.a]
        np.testing.assert_allclose(v_qndf, v_trap, rtol=0.01,
                                   err_msg="QNDF bus voltages diverge from trapezoid")


class TestQNDFEIG(unittest.TestCase):
    """EIG compatibility — single-step itm_step()."""

    def test_eig_single_step(self):
        """TDS.itm_step() should work for QNDF (used by EIG)."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'qndf'
        ss.TDS.init()

        # Single step should succeed
        result = ss.TDS.itm_step()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
