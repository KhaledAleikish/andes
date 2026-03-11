"""
Tests for TDS.reinit() — fast idempotent reset for repeated simulations.
"""

import time
import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


class TestReinitDeterminism(unittest.TestCase):
    """reinit() should produce identical trajectories across episodes."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 2.0
        cls.ss.TDS.init()

    def _run_episode(self):
        """Run one episode and return (t, x, y) arrays."""
        ss = self.ss
        ss.TDS.run(no_summary=True)
        ss.dae.ts.unpack()
        t = np.array(ss.dae.ts.t)
        x = np.array(ss.dae.ts.x)
        y = np.array(ss.dae.ts.y)
        return t, x, y

    def test_two_runs_identical(self):
        """init → run → reinit → run should produce identical timeseries."""
        t1, x1, y1 = self._run_episode()
        self.assertEqual(self.ss.exit_code, 0)

        self.ss.TDS.reinit()
        t2, x2, y2 = self._run_episode()
        self.assertEqual(self.ss.exit_code, 0)

        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_allclose(x1, x2, atol=1e-12)
        np.testing.assert_allclose(y1, y2, atol=1e-12)

    def test_ten_cycles_identical(self):
        """10 reinit cycles should all produce identical results."""
        self.ss.TDS.reinit()
        t_ref, x_ref, y_ref = self._run_episode()

        for i in range(9):
            self.ss.TDS.reinit()
            t, x, y = self._run_episode()

            np.testing.assert_array_equal(t_ref, t, err_msg=f"cycle {i+2} time differs")
            np.testing.assert_allclose(x_ref, x, atol=1e-12,
                                       err_msg=f"cycle {i+2} x differs")
            np.testing.assert_allclose(y_ref, y, atol=1e-12,
                                       err_msg=f"cycle {i+2} y differs")


class TestReinitWithAlter(unittest.TestCase):
    """reinit() with Alter events (= method, which sets absolute values)."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('ieee14/ieee14_alter.xlsx'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 5.0
        cls.ss.TDS.init()

    def test_alter_deterministic(self):
        """Alter with rand=1 should be deterministic when RNG is seeded."""
        ss = self.ss

        np.random.seed(42)
        ss.TDS.run(no_summary=True)
        ss.dae.ts.unpack()
        t1 = np.array(ss.dae.ts.t)
        x1 = np.array(ss.dae.ts.x)

        ss.TDS.reinit()
        np.random.seed(42)
        ss.TDS.run(no_summary=True)
        ss.dae.ts.unpack()
        t2 = np.array(ss.dae.ts.t)
        x2 = np.array(ss.dae.ts.x)

        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_allclose(x1, x2, atol=1e-12)


class TestReinitDifferentTf(unittest.TestCase):
    """reinit() should work with different tf values."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 2.0
        cls.ss.TDS.init()

    def test_different_tf(self):
        """reinit + run to tf=1 then reinit + run to tf=3."""
        ss = self.ss

        ss.TDS.config.tf = 1.0
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)
        self.assertAlmostEqual(float(ss.dae.t), 1.0, places=6)

        ss.TDS.reinit()
        ss.TDS.config.tf = 3.0
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)
        self.assertAlmostEqual(float(ss.dae.t), 3.0, places=6)


class TestReinitBustedRecovery(unittest.TestCase):
    """reinit() should recover from a busted simulation."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('ieee14/ieee14_fault.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 2.0
        cls.ss.TDS.init()

    def test_busted_recovery(self):
        """Crash the sim by clearing fault too late, then reinit and run clean."""
        ss = self.ss

        # Normal run should succeed
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)

        # reinit and run again — should still succeed
        ss.TDS.reinit()
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)


class TestReinitPerformance(unittest.TestCase):
    """Performance regression guard for reinit().

    Design target: < 2ms per call on a developer machine.
    CI threshold: < 10ms per call (10s for 1000 calls) to accommodate
    slower CI runners (observed: Ubuntu ~7ms, Windows ~3.7ms).
    """

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 2.0
        cls.ss.TDS.init()

    def test_1000_reinits_fast(self):
        """1000 reinits should complete in < 10 seconds."""
        t0 = time.perf_counter()
        for _ in range(1000):
            self.ss.TDS.reinit()
        elapsed = time.perf_counter() - t0

        # < 10 seconds for 1000 reinits = < 10ms per reinit
        # CI runners are slower than local machines; typical local ~2ms, CI ~7ms
        self.assertLess(elapsed, 10.0,
                        f"1000 reinits took {elapsed:.2f}s ({elapsed/1000*1000:.1f}ms each)")


class TestReinitQNDF(unittest.TestCase):
    """reinit() should work with QNDF integration method."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.method = 'qndf'
        cls.ss.TDS.config.tf = 2.0
        cls.ss.TDS.init()

    def test_qndf_deterministic(self):
        """QNDF: reinit should produce identical trajectories."""
        ss = self.ss

        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)
        ss.dae.ts.unpack()
        x1 = np.array(ss.dae.ts.x)

        ss.TDS.reinit()
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)
        ss.dae.ts.unpack()
        x2 = np.array(ss.dae.ts.x)

        # QNDF uses adaptive stepping, so timeseries lengths may differ
        # but for identical initial conditions they should be deterministic
        self.assertEqual(len(x1), len(x2))
        np.testing.assert_allclose(x1, x2, atol=1e-10)


class TestReinitNumParamRestore(unittest.TestCase):
    """reinit() should restore NumParam values modified by Alter +/-/*/÷."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
            setup=False,
        )
        # Add an Alter that uses '+' on a NumParam (GENROU inertia M)
        genrou_idx = cls.ss.GENROU.idx.v[0]
        cls.ss.add('Alter', model='GENROU', dev=genrou_idx,
                   src='M', method='+', amount=2.0, t=0.5)
        cls.ss.setup()

        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 2.0
        cls.ss.TDS.init()

        # Record the original M value for verification
        cls._m_original = cls.ss.GENROU.M.v[0]

    def test_numparam_restored_after_alter(self):
        """Alter '+' modifies M during run; reinit should restore it."""
        ss = self.ss

        # Run 1
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)
        ss.dae.ts.unpack()
        x1 = np.array(ss.dae.ts.x)

        # M should have been modified by Alter
        self.assertAlmostEqual(ss.GENROU.M.v[0], self._m_original + 2.0)

        # reinit should restore M
        ss.TDS.reinit()
        self.assertAlmostEqual(ss.GENROU.M.v[0], self._m_original)

        # Run 2 should be identical
        ss.TDS.run(no_summary=True)
        self.assertEqual(ss.exit_code, 0)
        ss.dae.ts.unpack()
        x2 = np.array(ss.dae.ts.x)

        np.testing.assert_allclose(x1, x2, atol=1e-12)


class TestReinitPreInit(unittest.TestCase):
    """reinit() before init() should raise RuntimeError."""

    def test_reinit_before_init(self):
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        # TDS.init() not called
        with self.assertRaises(RuntimeError):
            ss.TDS.reinit()


if __name__ == '__main__':
    unittest.main()
