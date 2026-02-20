import unittest

from andes.routines.adaptive import accept_reject, candidate_h


class TestAdaptiveController(unittest.TestCase):

    def test_candidate_h_zero_error_grows(self):
        h_new = candidate_h(err_est=0.0, h=0.1, order=2, max_factor=2.0)
        self.assertEqual(h_new, 0.2)

    def test_accept_reject_accepts(self):
        accepted, h_new, fail = accept_reject(
            err_est=0.5,
            h=0.1,
            deltatmax=1.0,
            order=2,
        )
        self.assertTrue(accepted)
        self.assertGreater(h_new, 0.1)
        self.assertEqual(fail, 0)

    def test_accept_reject_rejects(self):
        accepted, h_new, fail = accept_reject(
            err_est=2.0,
            h=0.1,
            deltatmax=1.0,
            order=2,
            fail_count=1,
            reject_max_factor=1.0,
            repeat_reject_after=1,
            repeat_reject_factor=0.5,
        )
        self.assertFalse(accepted)
        self.assertLessEqual(h_new, 0.05)
        self.assertEqual(fail, 2)


if __name__ == '__main__':
    unittest.main()
