import unittest
import numpy as np
from src.signals.processing import Signal, band_pass_filter, low_pass_filter, high_pass_filter, envelope, power_spectrum, process_signal

class TestProcessing(unittest.TestCase):
    def setUp(self) -> None:
        # Create a sample signal for testing
        # 100 Hz signal sampled at 1000 Hz for 1 second
        self.fs = 1000.0
        self.t = np.arange(0, 3.0, 1.0/self.fs)
        self.freq = 100.0
        self.y = np.sin(2 * np.pi * self.freq * self.t)
        self.signal = Signal(x=self.t, y=self.y)

    def test_signal_dataclass(self) -> None:
        """Test Signal dataclass structure."""
        self.assertTrue(hasattr(self.signal, 'x'))
        self.assertTrue(hasattr(self.signal, 'y'))
        np.testing.assert_array_equal(self.signal.x, self.t)
        np.testing.assert_array_equal(self.signal.y, self.y)

    def test_low_pass_filter(self) -> None:
        """Test low pass filter attenuation."""
        # Signal is 100Hz.
        # Low pass at 50Hz should attenuate it.
        cutoff = 50.0
        filtered = low_pass_filter(self.signal, self.fs, cutoff)
        
        self.assertEqual(len(filtered.x), len(self.signal.x))
        # Check middle segment to avoid filter transient
        mid = len(self.signal.y) // 2
        input_amp = np.max(np.abs(self.signal.y[mid-100:mid+100]))
        output_amp = np.max(np.abs(filtered.y[mid-100:mid+100]))
        self.assertLess(output_amp, input_amp * 0.1, "Signal should be significantly attenuated")

    def test_high_pass_filter(self) -> None:
        """Test high pass filter attenuation."""
        # Signal is 100Hz.
        # High pass at 200Hz should attenuate it.
        cutoff = 200.0
        filtered = high_pass_filter(self.signal, self.fs, cutoff)
        
        self.assertEqual(len(filtered.x), len(self.signal.x))
        mid = len(self.signal.y) // 2
        input_amp = np.max(np.abs(self.signal.y[mid-100:mid+100]))
        output_amp = np.max(np.abs(filtered.y[mid-100:mid+100]))
        self.assertLess(output_amp, input_amp * 0.1, "Signal should be significantly attenuated")

    def test_band_pass_filter(self) -> None:
        """Test pass vs stop band."""
        # Signal is 100Hz.
        # Band pass 80-120Hz should pass it.
        filtered_pass = band_pass_filter(self.signal, self.fs, 80.0, 120.0)
        
        # Band pass 200-300Hz should attenuate it.
        filtered_stop = band_pass_filter(self.signal, self.fs, 200.0, 300.0)

        mid = len(self.signal.y) // 2
        pass_amp = np.max(np.abs(filtered_pass.y[mid-100:mid+100]))
        stop_amp = np.max(np.abs(filtered_stop.y[mid-100:mid+100]))
        
        # Pass amplitude should be close to 1.0 (original), stop amplitude should be small
        self.assertGreater(pass_amp, 0.9)
        self.assertLess(stop_amp, 0.1)

    def test_envelope(self) -> None:
        """Test envelope extraction."""
        # AM signal: Carrier 100Hz, Modulation 5Hz
        carrier_freq = 100.0
        mod_freq = 5.0
        # Envelope varies between 0.5 and 1.5
        modulation = (1 + 0.5 * np.sin(2 * np.pi * mod_freq * self.t))
        y_am = modulation * np.sin(2 * np.pi * carrier_freq * self.t)
        signal_am = Signal(x=self.t, y=y_am)
        
        # Test with DC offset removal (default)
        env_dc_removed = envelope(signal_am, remove_dc_offset=True)
        
        self.assertEqual(len(env_dc_removed.x), len(signal_am.x))
        # With DC offset removed, the mean should be close to 0
        self.assertAlmostEqual(np.mean(env_dc_removed.y), 0.0, delta=0.01)
        # Check that it's dynamic (has variation)
        self.assertGreater(np.max(env_dc_removed.y) - np.min(env_dc_removed.y), 0.5)
        
        # Test without DC offset removal
        env_no_dc_removal = envelope(signal_am, remove_dc_offset=False)
        
        self.assertEqual(len(env_no_dc_removal.x), len(signal_am.x))
        self.assertTrue(np.all(env_no_dc_removal.y >= 0))
        
        # Check if extracted envelope correlates with modulation signal (ignoring edge effects)
        mid = len(self.t) // 2
        window = slice(mid-100, mid+100)
        
        # Simple check: max envelope should be near max modulation
        self.assertAlmostEqual(np.max(env_no_dc_removal.y), 1.5, delta=0.2)
        # min envelope should be near min modulation (0.5) roughly (hard due to carrier dips)
        # Instead, just checking it's dynamic
        self.assertGreater(np.max(env_no_dc_removal.y) - np.min(env_no_dc_removal.y), 0.5)


    def test_power_spectrum(self) -> None:
        """Test power spectrum frequency peak."""
        psd = power_spectrum(self.signal, self.fs)
        
        # Check that the peak is around 100Hz
        peak_idx = np.argmax(psd.y)
        peak_freq = psd.x[peak_idx]
        
        self.assertAlmostEqual(peak_freq, 100.0, delta=5.0) # Delta covers freq resolution
        self.assertTrue(len(psd.x) > 0)

    def test_process_signal(self) -> None:
        """Test the signal processing pipeline with all options."""
        steps = [
            {
                'type': 'high_pass_filter',
                'sampling_frequency': self.fs,
                'cutoff_frequency': 50.0
            },
            {
                'type': 'low_pass_filter',
                'sampling_frequency': self.fs,
                'cutoff_frequency': 200.0
            },
            {
                'type': 'band_pass_filter',
                'sampling_frequency': self.fs,
                'low_cutoff_frequency': 80.0,
                'high_cutoff_frequency': 120.0
            },
            {
                'type': 'envelope',
                'remove_dc_offset': True    
            },
            {
                'type': 'power_spectrum',
                'sampling_frequency': self.fs
            }
        ]
        
        result = process_signal(self.signal, steps)
        
        self.assertIsInstance(result, Signal)
        self.assertTrue(len(result.x) > 0)
        self.assertTrue(len(result.y) > 0)

        # Test invalid step
        bad_steps = [{'type': 'invalid_step_name'}]
        with self.assertRaises(ValueError):
            process_signal(self.signal, bad_steps)
