"""
Tests for adaptive trapezoid integration method.
"""

import unittest

import andes
from andes.utils.paths import get_case


class TestTrapAdaptConfig(unittest.TestCase):
    """Config and method wiring tests."""

    def test_fixt_override(self):
        """Adaptive trapezoid should force variable stepping."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.TDS.config.fixt = 1
        ss.TDS.set_method('trap_adapt')
        self.assertEqual(ss.TDS.config.fixt, 0)


class TestTrapAdaptSmoke(unittest.TestCase):
    """Basic smoke tests for adaptive trapezoid."""

    def test_kundur_trap_adapt(self):
        """Kundur full case with adaptive trapezoid."""
        ss = andes.load(
            get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'trap_adapt'
        ss.TDS.config.tf = 1.0
        ss.TDS.init()
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)

    def test_ieee14_shaft5_trap_adapt(self):
        """IEEE 14-bus with shaft model — exercises post-event LTE recovery."""
        ss = andes.load(
            get_case('ieee14/ieee14_shaft5.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        ss.TDS.config.method = 'trap_adapt'
        ss.TDS.config.tf = 3
        ss.TDS.config.tstep = 0.001
        ok = ss.TDS.run()

        self.assertTrue(ok)
        self.assertEqual(ss.exit_code, 0)


if __name__ == '__main__':
    unittest.main()
