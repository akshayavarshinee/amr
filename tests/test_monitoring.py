"""Unit tests for monitoring and PSI drift detection."""

import unittest
import numpy as np
import pandas as pd

from amr_guard.monitoring.drift import _psi, compute_drift_report


class TestDriftMonitoring(unittest.TestCase):

    def test_psi_zero_for_identical_distributions(self):
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(0, 1, 1000))
        score = _psi(data, data, bins=10)
        self.assertLess(score, 0.01)

    def test_psi_high_for_drifted_distributions(self):
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 1000))
        cur = pd.Series(rng.normal(5, 1, 1000))
        score = _psi(ref, cur, bins=10)
        self.assertGreater(score, 0.5)

    def test_psi_handles_numeric_codes(self):
        ref = pd.Series([0, 0, 0, 1, 1, 2] * 100)
        cur_no_drift = pd.Series([0, 0, 0, 1, 1, 2] * 100)
        cur_drifted = pd.Series([2, 2, 2, 2, 2, 2] * 100)

        self.assertLess(_psi(ref, cur_no_drift, bins=5), 0.05)
        self.assertGreater(_psi(ref, cur_drifted, bins=5), 0.3)


if __name__ == "__main__":
    unittest.main()
