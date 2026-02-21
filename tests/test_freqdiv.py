"""
Tests for FreqDiv (Frequency Divider) model.
"""

import unittest

import numpy as np

import andes


class TestFreqDiv(unittest.TestCase):

    def test_freqdiv_vs_busfreq(self):
        """FreqDiv and BusFreq should agree on IEEE 14-bus with line trip."""

        ss = andes.load(andes.get_case('ieee14/ieee14.json'),
                        setup=False, no_output=True, default_config=True)

        for bus_idx in ss.Bus.idx.v:
            ss.add('FreqDiv', bus=bus_idx)

        existing_bf_buses = set(ss.BusFreq.bus.v)
        for bus_idx in ss.Bus.idx.v:
            if bus_idx not in existing_bf_buses:
                ss.add('BusFreq', bus=bus_idx)

        ss.setup()
        ss.PFlow.run()
        ss.TDS.config.tf = 5.0
        ss.TDS.run()

        self.assertTrue(ss.TDS.converged)

        # Steady-state: all frequencies must be 1.0 pu
        np.testing.assert_allclose(ss.FreqDiv.f.v, 1.0, atol=1e-6)

        # Align timeseries by bus
        ts_fd = ss.TDS.get_timeseries(ss.FreqDiv.f)
        ts_bf = ss.TDS.get_timeseries(ss.BusFreq.f)
        time = ts_fd.index.values

        fd_map = dict(zip(ss.FreqDiv.bus.v, ss.FreqDiv.idx.v))
        bf_map = dict(zip(ss.BusFreq.bus.v, ss.BusFreq.idx.v))
        common = sorted(set(fd_map) & set(bf_map))

        fd_arr = np.column_stack([ts_fd[fd_map[b]].values for b in common])
        bf_arr = np.column_stack([ts_bf[bf_map[b]].values for b in common])

        # Skip initial washout-filter transient
        mask = time > 0.5
        max_err = np.max(np.abs(fd_arr[mask] - bf_arr[mask]))

        self.assertLess(max_err, 0.05,
                        f'Max error {max_err:.6f} exceeds 0.05 pu')


if __name__ == '__main__':
    unittest.main()
