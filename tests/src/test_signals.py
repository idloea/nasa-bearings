import unittest
from src.signals import resolution


class TestSignals(unittest.TestCase):
    def test_resolution(self) -> None:
        sampling_frequency = 100
        expected_result = 0.01
        result = resolution(sampling_frequency)
        self.assertEqual(result, expected_result)
