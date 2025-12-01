import unittest
from src.signals import get_resolution


class TestSignals(unittest.TestCase):
    def test_get_resolution(self) -> None:
        sampling_frequency = 100
        expected_result = 0.01
        result = get_resolution(sampling_frequency)
        self.assertEqual(result, expected_result)
