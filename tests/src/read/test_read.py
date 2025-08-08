import unittest
from pathlib import Path
from src.read import read_nasa_vibration_file


class TestRead(unittest.TestCase):
    def test_read_nasa_vibration_file(self) -> None:
        path = Path('tests/src/read/mock_data/2003.10.22.12.06.24')
        column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                        'channel_5', 'channel_6', 'channel_7', 'channel_8']
        df = read_nasa_vibration_file(path=path, column_names=column_names)
        shape = df.shape
        expected_shape = (20480, 8)
        self.assertEqual(expected_shape, shape)
