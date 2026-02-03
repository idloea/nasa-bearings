import unittest
from src import signals
import numpy as np


class TestSignals(unittest.TestCase):
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

    def test_power_spectrum_when_number_of_points_per_segment_is_none(self) -> None:
        number_of_points_per_segment = 512
        frequencies, amplitudes = signals.power_spectrum(waveform=self.waveform, 
                                                         sampling_frequency=self.sampling_frequency, 
                                                         number_of_points_per_segment=number_of_points_per_segment)
        
        expected_frequencies = np.array([0., 1.953125, 3.90625, 5.859375, 7.8125])        
        expected_amplitudes = np.array([8.33031602e-05, 1.03965024e-04, 2.43945161e-05, 1.90974684e-06, 3.82374134e-08])
        
        self.assertIsInstance(frequencies, np.ndarray)
        self.assertIsInstance(amplitudes, np.ndarray)
        self.assertEqual(len(frequencies), len(amplitudes))
        self.assertEqual(len(frequencies), (number_of_points_per_segment // 2) + 1)
        np.testing.assert_almost_equal(frequencies[:5], expected_frequencies)
        np.testing.assert_almost_equal(amplitudes[:5], expected_amplitudes)

    def test_power_spectrum_when_number_of_points_per_segment_is_not_none(self) -> None:
        number_of_points_per_segment = 512
        number_of_overlapping_points_between_segments = 100
        frequencies, amplitudes = signals.power_spectrum(waveform=self.waveform, 
                                                         sampling_frequency=self.sampling_frequency, 
                                                         number_of_points_per_segment=number_of_points_per_segment,
                                                         number_of_overlapping_points_between_segments=number_of_overlapping_points_between_segments)
        
        expected_frequencies = np.array([0., 1.953125, 3.90625, 5.859375, 7.8125])        
        expected_amplitudes = np.array([1.04022052e-04, 1.29822876e-04, 3.04618397e-05, 2.38473358e-06, 4.77476092e-08])
        
        np.testing.assert_almost_equal(frequencies[:5], expected_frequencies)
        np.testing.assert_almost_equal(amplitudes[:5], expected_amplitudes)
        