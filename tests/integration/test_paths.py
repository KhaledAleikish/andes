import os
import unittest

import andes
from andes.utils.paths import list_cases


class TestPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.kundur = 'kundur/'
        self.matpower = 'matpower/'
        self.ieee14 = andes.get_case("ieee14/ieee14.raw")

    def test_tree(self):
        list_cases(self.kundur, no_print=True)
        list_cases(self.matpower, no_print=True)

    def test_addfile_path(self):
        """Test single addfile as string (backward compat)."""
        path, case = os.path.split(self.ieee14)
        ss = andes.load('ieee14.raw', addfile='ieee14.dyr',
                        input_path=path, default_config=True,
                        )
        self.assertNotEqual(ss, None)

        ss = andes.run('ieee14.raw', addfile='ieee14.dyr',
                       input_path=path,
                       no_output=True, default_config=True,
                       )
        self.assertNotEqual(ss, None)

    def test_addfile_list(self):
        """Test single addfile passed as a list."""
        path, _ = os.path.split(self.ieee14)
        ss = andes.load('ieee14.raw', addfile=['ieee14.dyr'],
                        input_path=path, default_config=True,
                        )
        self.assertIsNotNone(ss)
        self.assertGreater(ss.GENROU.n, 0)

    def test_addfile_cross_format_xlsx(self):
        """Test RAW base + xlsx addfile (uses read fallback)."""
        path, _ = os.path.split(self.ieee14)
        ss = andes.load('ieee14.raw', addfile=['ieee14_dyn_only.xlsx'],
                        input_path=path, default_config=True,
                        )
        self.assertIsNotNone(ss)
        self.assertGreater(ss.GENROU.n, 0)

    def test_addfile_no_addfile(self):
        """Test that no addfile works (regression)."""
        ss = andes.load(self.ieee14, default_config=True, no_output=True)
        self.assertIsNotNone(ss)
        self.assertEqual(ss.GENROU.n, 0)

    def test_relative_path(self):
        ss = andes.run('ieee14.raw',
                       input_path=andes.get_case('ieee14/', check=False),
                       no_output=True, default_config=True,
                       )
        self.assertNotEqual(ss, None)

    def test_pert_file(self):
        """Test path of pert file"""
        path, case = os.path.split(self.ieee14)

        # --- with pert file ---
        ss = andes.run('ieee14.raw', pert='pert.py',
                       input_path=path, no_output=True, default_config=True,
                       )
        ss.TDS.init()
        self.assertIsNotNone(ss.TDS.callpert)

        # --- without pert file ---
        ss = andes.run('ieee14.raw',
                       input_path=path, no_output=True, default_config=True,
                       )
        ss.TDS.init()
        self.assertIsNone(ss.TDS.callpert)
