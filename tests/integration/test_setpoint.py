"""
Tests for group-level setpoint API:
- SynGen.set_pref() / set_vref() / get_pref() / get_vref()
- RenGen.set_pref() / set_qref() / get_pref() / get_qref()
- GroupBase.set_setpoint() / get_setpoint()
"""

import logging
import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


class TestSynGenSetpoint(unittest.TestCase):
    """Test SynGen.set_pref() and set_vref() on Kundur 4-machine system."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()

    def test_set_pref_with_governor(self):
        """set_pref routes to TGOV1.pref0 when governor is connected."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        old_pref0 = ss.TGOV1.pref0.v[0].copy()

        ss.SynGen.set_pref(ss, gen_idx, 0.8)

        self.assertAlmostEqual(ss.TGOV1.pref0.v[0], 0.8)

        # Restore
        ss.TGOV1.pref0.v[0] = old_pref0

    def test_set_vref_with_exciter(self):
        """set_vref routes to EXDC2.vref0 when exciter is connected."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        old_vref0 = ss.EXDC2.vref0.v[0].copy()

        ss.SynGen.set_vref(ss, gen_idx, 1.05)

        self.assertAlmostEqual(ss.EXDC2.vref0.v[0], 1.05)

        # Restore
        ss.EXDC2.vref0.v[0] = old_vref0

    def test_set_pref_all_generators(self):
        """set_pref works for every generator in the system."""
        ss = self.ss
        for i, gen_idx in enumerate(ss.GENROU.idx.v):
            old = ss.TGOV1.pref0.v[i].copy()
            ss.SynGen.set_pref(ss, gen_idx, 0.5)
            self.assertAlmostEqual(ss.TGOV1.pref0.v[i], 0.5)
            ss.TGOV1.pref0.v[i] = old

    def test_set_pref_no_governor(self):
        """set_pref falls back to GENROU.tm0 when no governor exists."""
        # Load a case without governors
        ss2 = andes.load(
            get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )
        ss2.PFlow.run()

        # ieee14.json has GENCLS generators without turbine governors
        if ss2.GENCLS.n > 0:
            gen_idx = ss2.GENCLS.idx.v[0]
            old_tm0 = ss2.GENCLS.tm0.v[0].copy()

            ss2.SynGen.set_pref(ss2, gen_idx, 0.7)

            self.assertAlmostEqual(ss2.GENCLS.tm0.v[0], 0.7)

            # Restore
            ss2.GENCLS.tm0.v[0] = old_tm0

    def test_set_vref_no_exciter(self):
        """set_vref falls back to GENROU.vf0 when no exciter exists."""
        ss2 = andes.load(
            get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )
        ss2.PFlow.run()

        if ss2.GENCLS.n > 0:
            gen_idx = ss2.GENCLS.idx.v[0]
            old_vf0 = ss2.GENCLS.vf0.v[0].copy()

            ss2.SynGen.set_vref(ss2, gen_idx, 1.1)

            self.assertAlmostEqual(ss2.GENCLS.vf0.v[0], 1.1)

            ss2.GENCLS.vf0.v[0] = old_vf0

    def test_get_pref_with_governor(self):
        """get_pref returns TGOV1.pref0 when governor is connected."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        expected = ss.TGOV1.pref0.v[0]
        actual = ss.SynGen.get_pref(ss, gen_idx)

        self.assertAlmostEqual(actual, expected)

    def test_get_vref_with_exciter(self):
        """get_vref returns EXDC2.vref0 when exciter is connected."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        expected = ss.EXDC2.vref0.v[0]
        actual = ss.SynGen.get_vref(ss, gen_idx)

        self.assertAlmostEqual(actual, expected)

    def test_get_then_set_roundtrip(self):
        """get_pref followed by set_pref restores original value."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        original = ss.SynGen.get_pref(ss, gen_idx)
        ss.SynGen.set_pref(ss, gen_idx, original + 0.1)
        self.assertAlmostEqual(ss.SynGen.get_pref(ss, gen_idx), original + 0.1)

        # Restore
        ss.SynGen.set_pref(ss, gen_idx, original)
        self.assertAlmostEqual(ss.SynGen.get_pref(ss, gen_idx), original)

    def test_generic_set_setpoint(self):
        """GroupBase.set_setpoint works with arbitrary name."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        old = ss.SynGen.get_setpoint(ss, gen_idx, 'pref')
        ss.SynGen.set_setpoint(ss, gen_idx, 'pref', 0.9)
        self.assertAlmostEqual(ss.SynGen.get_setpoint(ss, gen_idx, 'pref'), 0.9)

        # Restore
        ss.SynGen.set_setpoint(ss, gen_idx, 'pref', old)


class TestSynGenSetpointTDS(unittest.TestCase):
    """Verify that setpoint changes propagate during TDS."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()

    def test_pref_change_affects_tds(self):
        """Changing pref0 via set_pref and running TDS produces a response."""
        ss = self.ss
        gen_idx = ss.GENROU.idx.v[0]

        # Disable pre-existing Toggle
        if ss.Toggle.n > 0:
            for i in range(ss.Toggle.n):
                ss.Toggle.u.v[i] = 0

        # Run to steady state
        ss.TDS.config.tf = 1.0
        ss.TDS.run()

        # Record omega at t=1
        omega_before = ss.GENROU.omega.v[0].copy()

        # Step change in power reference using get then set
        old_pref = ss.SynGen.get_pref(ss, gen_idx)
        ss.SynGen.set_pref(ss, gen_idx, old_pref + 0.05)

        # Continue TDS
        ss.TDS.config.tf = 3.0
        ss.TDS.run()

        # omega should have changed (transient response)
        omega_after = ss.GENROU.omega.v[0]
        self.assertNotAlmostEqual(omega_before, omega_after, places=6)


class TestRenGenSetpoint(unittest.TestCase):
    """Test RenGen.set_pref() and set_qref()."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('ieee14/ieee14_solar.xlsx'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()

    def test_set_pref_with_ren_exciter(self):
        """set_pref routes to REECA1.Pref0 when RenExciter is connected."""
        ss = self.ss

        # Get the RenGen idx that REECA1 controls
        reg_idx = ss.REECA1.reg.v[0]

        old_pref0 = ss.REECA1.Pref0.v[0].copy()

        ss.RenGen.set_pref(ss, reg_idx, 0.5)

        self.assertAlmostEqual(ss.REECA1.Pref0.v[0], 0.5)

        # Restore
        ss.REECA1.Pref0.v[0] = old_pref0

    def test_set_qref_with_ren_exciter(self):
        """set_qref routes to REECA1.qref0 when RenExciter is connected."""
        ss = self.ss

        reg_idx = ss.REECA1.reg.v[0]

        old_qref0 = ss.REECA1.qref0.v[0].copy()

        ss.RenGen.set_qref(ss, reg_idx, 0.1)

        self.assertAlmostEqual(ss.REECA1.qref0.v[0], 0.1)

        ss.REECA1.qref0.v[0] = old_qref0

    def test_get_pref_with_ren_exciter(self):
        """get_pref returns REECA1.Pref0 when RenExciter is connected."""
        ss = self.ss
        reg_idx = ss.REECA1.reg.v[0]

        expected = ss.REECA1.Pref0.v[0]
        actual = ss.RenGen.get_pref(ss, reg_idx)

        self.assertAlmostEqual(actual, expected)

    def test_get_qref_with_ren_exciter(self):
        """get_qref returns REECA1.qref0 when RenExciter is connected."""
        ss = self.ss
        reg_idx = ss.REECA1.reg.v[0]

        expected = ss.REECA1.qref0.v[0]
        actual = ss.RenGen.get_qref(ss, reg_idx)

        self.assertAlmostEqual(actual, expected)


class TestSetpointErrors(unittest.TestCase):
    """Test error handling for setpoint API."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()

    def test_invalid_idx_raises(self):
        """Passing a nonexistent idx raises KeyError."""
        with self.assertRaises(KeyError):
            self.ss.SynGen.set_pref(self.ss, 'nonexistent', 0.5)


class TestFallbackWarning(unittest.TestCase):
    """Test that fallback to device emits a warning."""

    def test_fallback_logs_warning(self):
        """set_pref without governor logs a warning about fallback."""
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()

        if ss.GENCLS.n > 0:
            gen_idx = ss.GENCLS.idx.v[0]
            old_tm0 = ss.GENCLS.tm0.v[0].copy()

            with self.assertLogs('andes.models.group', level='WARNING') as cm:
                ss.SynGen.set_pref(ss, gen_idx, 0.7)

            self.assertTrue(any('No TurbineGov controller found' in msg
                                for msg in cm.output))

            # Restore
            ss.GENCLS.tm0.v[0] = old_tm0


class TestCheckSetpoints(unittest.TestCase):
    """Test that _check_setpoints catches bad declarations at setup time."""

    def test_bad_attr_name_raises(self):
        """A typo in _setpoints target raises AttributeError during setup."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
            setup=False,
        )
        # Inject a bad _setpoints on one model class
        ss.TGOV1._setpoints = {'pref': 'nonexistent_attr'}

        with self.assertRaises(AttributeError):
            ss.setup()


if __name__ == '__main__':
    unittest.main()
