import unittest
from pathlib import Path
from src.read import read_nasa_vibration_file, read_nasa_vibration_files_in_directory

class TestRead(unittest.TestCase):
    def setUp(self) -> None:
        sampling_frequency = 20000  # Hz
        self.signal_resolution = 1 /sampling_frequency

    def test_read_nasa_vibration_file(self) -> None:
        file_path = Path('tests/src/read/mock_data/2003.10.22.12.06.24')
        column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                        'channel_5', 'channel_6', 'channel_7', 'channel_8']
        df = read_nasa_vibration_file(file_path=file_path, column_names=column_names,
                                      signal_resolution=self.signal_resolution)
        shape = df.shape
        expected_shape = (20480, 9)
        expected_column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5',
                                 'channel_6', 'channel_7', 'channel_8', 'measurement_time_in_seconds']
        self.assertEqual(expected_shape, shape)
        self.assertEqual(expected_column_names, df.columns.tolist())

    def test_read_nasa_vibration_files_in_directory(self) -> None:
        files_path = Path('tests/src/read/mock_data/mock_folder')
        column_names =  ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        df = read_nasa_vibration_files_in_directory(files_path=files_path, column_names=column_names,
                                                    signal_resolution=self.signal_resolution)
        self.assertEqual(len(df), 2)

    def test_raises_on_empty_directory(self) -> None:
        files_path = Path('tests/src/read/mock_data/empty_folder')
        column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        with self.assertRaises(ValueError) as context:
            read_nasa_vibration_files_in_directory(files_path=files_path, column_names=column_names,
                                                   signal_resolution=self.signal_resolution)
        self.assertIn("No files found in the directory", str(context.exception))

    