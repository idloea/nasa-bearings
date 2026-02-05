import unittest
from src.signals import calculations
import numpy as np


class TestCalculations(unittest.TestCase):
    def setUp(self) -> None:
        """Set up basic signal parameters for testing."""
        self.sampling_frequency = 1000.0  # 1 kHz sampling
        self.duration = 1.0  # 1 second
        self.t = np.linspace(0, self.duration, int(self.sampling_frequency * self.duration), endpoint=False)
        # Create a 50Hz sine wave
        self.freq_target = 50.0
        self.waveform = np.sin(2 * np.pi * self.freq_target * self.t)    

    def test_resolution(self) -> None:
        sampling_frequency = 100
        expected_result = 0.01
        result = calculations.resolution(sampling_frequency)
        self.assertEqual(result, expected_result)
    
    def test_root_mean_square(self) -> None:
        signal = np.array([1, 2])
        result = calculations.root_mean_square(signal)
        expected_result = np.sqrt(2.5)        
        self.assertEqual(result, expected_result)

    def test_calculate_absolute_peak(self) -> None:
        signal = np.array([-3, 1, 2, -4, 0])
        result = calculations.absolute_peak(signal)
        expected_result = 4
        self.assertEqual(result, expected_result)

    def test_calculate_absolute_peak_empty_array(self) -> None:
        with self.assertRaises(ValueError):
            calculations.absolute_peak(np.array([]))        
    
    def test_crest_factor(self) -> None:
        signal = np.array([1, -2, 3, -4, 5])
        result = calculations.crest_factor(signal)
        expected_peak = 5
        expected_rms = calculations.root_mean_square(y=signal)
        expected_crest_factor = expected_peak / expected_rms
        self.assertAlmostEqual(result, expected_crest_factor)

    def test_calculate_average_rectified_value(self) -> None:
        signal = np.array([-1, 2, -3, 4, -5])
        result = calculations.average_rectified_value(signal)
        expected_result = 3.0
        self.assertEqual(result, expected_result)

    def test_calculate_shape_factor(self) -> None:
        signal = np.array([1, -2, 3, -4, 5])
        result = calculations.shape_factor(signal)
        expected_rms = calculations.root_mean_square(y=signal)
        expected_arv = calculations.average_rectified_value(y=signal)
        expected_shape_factor = expected_rms / expected_arv
        self.assertAlmostEqual(result, expected_shape_factor)
