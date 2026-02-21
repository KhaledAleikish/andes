"""Unit tests for symprocessor helpers."""

import unittest

from andes.core.symprocessor import _warn_missing_e_str


class TestWarnMissingEStr(unittest.TestCase):
    """Tests for _warn_missing_e_str."""

    def test_warns_when_numeric_flag_set(self):
        """Should warn when model has g_numeric but var has no e_str."""
        with self.assertLogs('andes.core.symprocessor', level='WARNING') as cm:
            _warn_missing_e_str('MyModel', 'x', 'g', has_numeric=True)

        self.assertEqual(len(cm.output), 1)
        self.assertIn('MyModel.x', cm.output[0])
        self.assertIn('g_numeric', cm.output[0])
        self.assertIn("e_str='0'", cm.output[0])

    def test_warns_for_f_numeric(self):
        """Should warn for f_numeric with correct equation type."""
        with self.assertLogs('andes.core.symprocessor', level='WARNING') as cm:
            _warn_missing_e_str('GenModel', 'omega', 'f', has_numeric=True)

        self.assertIn('f_numeric', cm.output[0])
        self.assertIn('dae.f', cm.output[0])

    def test_no_warn_without_numeric_flag(self):
        """Should not warn when model has no numeric method."""
        with self.assertNoLogs('andes.core.symprocessor', level='WARNING'):
            _warn_missing_e_str('PureSymbolic', 'v', 'g', has_numeric=False)


if __name__ == '__main__':
    unittest.main()
