import unittest
from src import signals
import numpy as np


class TestSignals(unittest.TestCase):
    def test_resolution(self) -> None:
        sampling_frequency = 100
        expected_result = 0.01
        result = signals.resolution(sampling_frequency)
        self.assertEqual(result, expected_result)
    
    def test_root_mean_square(self) -> None:
        signal = np.array([1, 2])
        result = signals.root_mean_square(signal)
        expected_result = np.sqrt(2.5)        
        self.assertEqual(result, expected_result)

    def test_calculate_absolute_peak(self) -> None:
        signal = np.array([-3, 1, 2, -4, 0])
        result = signals.absolute_peak(signal)
        expected_result = 4
        self.assertEqual(result, expected_result)

    def test_calculate_absolute_peak_empty_array(self) -> None:
        with self.assertRaises(ValueError):
            signals.absolute_peak(np.array([]))        
    
    def test_crest_factor(self) -> None:
        signal = np.array([1, -2, 3, -4, 5])
        result = signals.crest_factor(signal)
        expected_peak = 5
        expected_rms = signals.root_mean_square(y=signal)
        expected_crest_factor = expected_peak / expected_rms
        self.assertAlmostEqual(result, expected_crest_factor)

    def test_calculate_average_rectified_value(self) -> None:
        signal = np.array([-1, 2, -3, 4, -5])
        result = signals.average_rectified_value(signal)
        expected_result = 3.0
        self.assertEqual(result, expected_result)

    def test_calculate_shape_factor(self) -> None:
        signal = np.array([1, -2, 3, -4, 5])
        result = signals.shape_factor(signal)
        expected_rms = signals.root_mean_square(y=signal)
        expected_arv = signals.average_rectified_value(y=signal)
        expected_shape_factor = expected_rms / expected_arv
        self.assertAlmostEqual(result, expected_shape_factor)