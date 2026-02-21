"""
Tests for configurations.
"""

import unittest
from collections import OrderedDict

import andes
from andes.core.common import Config


class TestConfigOption(unittest.TestCase):
    """
    Tests for config_option.
    """

    def test_config_option(self):
        """
        Test single and multiple config_option passed to `andes.run`.
        """

        path = andes.get_case("5bus/pjm5bus.json")
        self.assertRaises(ValueError, andes.load, path, config_option={"TDS = 1"})
        self.assertRaises(ValueError, andes.load, path, config_option={"System.TDS.any = 1"})
        self.assertRaises(ValueError, andes.load, path, config_option={"TDS.tf == 1"})

        ss = andes.load(path, config_option={"PQ.pq2z = 0"}, default_config=True)
        self.assertEqual(ss.PQ.config.pq2z, 0)

        ss = andes.load(path, config_option={"PQ.pq2z=0"}, default_config=True)
        self.assertEqual(ss.PQ.config.pq2z, 0)

        ss = andes.load(path, config_option=["PQ.pq2z=0", "TDS.tf = 1"], default_config=True)
        self.assertEqual(ss.PQ.config.pq2z, 0)
        self.assertEqual(ss.TDS.config.tf, 1)


class TestConfigDeprecated(unittest.TestCase):
    """Tests for the deprecated-field mechanism in Config."""

    def test_deprecated_dict_no_hint(self):
        """Deprecated field with empty hint is blocked."""
        c = Config('Test')
        c._deprecated['old'] = ''
        c.old = 42
        self.assertNotIn('old', c.__dict__)

    def test_deprecated_dict_with_hint(self):
        """Deprecated field with hint is blocked and hint is logged."""
        c = Config('Test')
        c._deprecated['old'] = 'Use [Runtime] instead.'
        with self.assertLogs('andes.core.common', level='DEBUG') as cm:
            c.old = 42
        self.assertNotIn('old', c.__dict__)
        self.assertTrue(any('[Runtime]' in msg for msg in cm.output))

    def test_deprecated_get_returns_zero(self):
        """Getting a deprecated field returns 0."""
        c = Config('Test')
        c._deprecated['gone'] = ''
        self.assertEqual(c.gone, 0)

    def test_deprecated_add_blocked(self):
        """_add() skips deprecated fields from rc loading."""
        c = Config('Test')
        c._deprecated['numba'] = 'Use [Runtime].'
        c._add(numba=1, freq=60)
        self.assertNotIn('numba', c.__dict__)
        self.assertEqual(c.freq, 60)

    def test_deprecated_add_warns(self):
        """_add() logs a warning with the hint when loading deprecated key."""
        c = Config('Test')
        c._deprecated['numba'] = 'Use [Runtime].'
        with self.assertLogs('andes.core.common', level='WARNING') as cm:
            c._add(numba=1)
        self.assertTrue(any('[Runtime]' in msg for msg in cm.output))

    def test_deprecated_update_blocked(self):
        """update() skips deprecated fields."""
        c = Config('Test', freq=60)
        c.add_extra('_help', freq='base frequency')
        c.add_extra('_alt', freq='float')
        c._deprecated['numba'] = ''
        c.update(numba=1)
        self.assertNotIn('numba', c.__dict__)
        self.assertEqual(c.freq, 60)

    def test_deprecated_load_from_configparser(self):
        """Config.load() skips deprecated fields from a ConfigParser."""
        import configparser
        cp = configparser.ConfigParser()
        cp['Test'] = {'numba': '1', 'freq': '50'}

        c = Config('Test')
        c._deprecated['numba'] = 'Moved.'
        c.load(cp)
        c.add_extra('_help', freq='base frequency')

        self.assertNotIn('numba', c.__dict__)
        self.assertEqual(c.freq, 50)


class TestConfigUnrecognized(unittest.TestCase):
    """Tests for unrecognized-field detection in Config.check()."""

    def test_check_warns_unrecognized(self):
        """check() warns about fields with no _help entry."""
        c = Config('Test', freq=60)
        c.add_extra('_help', freq='base frequency')
        # Inject an unrecognized field directly
        c.__dict__['typo_field'] = 99
        c._dict = OrderedDict()  # force refresh on next as_dict()

        with self.assertLogs('andes.core.common', level='WARNING') as cm:
            c.check()
        self.assertTrue(any('typo_field' in msg for msg in cm.output))
        self.assertTrue(any('not recognized' in msg for msg in cm.output))

    def test_check_no_warning_for_valid(self):
        """check() does not warn about properly registered fields."""
        c = Config('Test', freq=60)
        c.add_extra('_help', freq='base frequency')
        c.add_extra('_alt', freq='float')
        # Should not raise or warn
        c.check()


class TestConfigSplit(unittest.TestCase):
    """Tests for the system.config / system.runtime split."""

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            no_output=True,
            default_config=True,
        )

    def test_case_fields_on_config(self):
        """system.config has exactly the case-relevant fields."""
        cfg = self.ss.config.as_dict()
        self.assertIn('freq', cfg)
        self.assertIn('mva', cfg)
        self.assertIn('diag_eps', cfg)
        self.assertIn('warn_abnormal', cfg)

    def test_runtime_fields_on_runtime(self):
        """system.runtime has the machine/env fields."""
        rt = self.ss.runtime.as_dict()
        self.assertIn('numba', rt)
        self.assertIn('sparselib', rt)
        self.assertIn('dime_enabled', rt)
        self.assertIn('np_divide', rt)
        self.assertIn('seed', rt)
        self.assertIn('ipadd', rt)

    def test_runtime_fields_not_on_config(self):
        """system.config does NOT have runtime fields."""
        cfg = self.ss.config.as_dict()
        for key in ('numba', 'sparselib', 'dime_enabled', 'np_divide', 'ipadd'):
            self.assertNotIn(key, cfg)

    def test_case_fields_not_on_runtime(self):
        """system.runtime does NOT have case fields."""
        rt = self.ss.runtime.as_dict()
        for key in ('freq', 'mva', 'diag_eps', 'warn_abnormal'):
            self.assertNotIn(key, rt)

    def test_config_deprecated_returns_zero(self):
        """Accessing moved fields on system.config returns 0."""
        self.assertEqual(self.ss.config.numba, 0)
        self.assertEqual(self.ss.config.sparselib, 0)

    def test_cli_override_runtime(self):
        """CLI config_option can set Runtime fields."""
        ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            no_output=True,
            default_config=True,
            config_option=['Runtime.numba=1'],
        )
        self.assertEqual(ss.runtime.numba, 1)

    def test_collect_config_rows_excludes_runtime(self):
        """collect_config_rows() does not include runtime fields."""
        rows = self.ss.config_runtime.collect_config_rows()
        sections = {r['section'] for r in rows}
        self.assertNotIn('Runtime', sections)

        keys = {r['key'] for r in rows}
        for key in ('numba', 'sparselib', 'dime_enabled'):
            self.assertNotIn(key, keys)

    def test_collect_config_includes_runtime(self):
        """collect_config() (for rc files) includes [Runtime]."""
        cp = self.ss.config_runtime.collect_config()
        self.assertIn('Runtime', cp)
        self.assertIn('numba', cp['Runtime'])
