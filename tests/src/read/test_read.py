import unittest
from pathlib import Path
from src.read import read_nasa_vibration_file, read_nasa_vibration_files_in_directory

class TestRead(unittest.TestCase):
    def test_read_nasa_vibration_file(self) -> None:
        file_path = Path('tests/src/read/mock_data/2003.10.22.12.06.24')
        column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                        'channel_5', 'channel_6', 'channel_7', 'channel_8']
        df = read_nasa_vibration_file(file_path=file_path, column_names=column_names)
        shape = df.shape
        expected_shape = (20480, 8)
        self.assertEqual(expected_shape, shape)

    def test_read_nasa_vibration_files_in_directory(self) -> None:
        files_path = Path('tests/src/read/mock_data/mock_folder')
        column_names =  ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        df = read_nasa_vibration_files_in_directory(files_path=files_path, column_names=column_names)
        self.assertEqual(set(df.columns), set(['file_name'] + column_names))
        self.assertEqual(df.shape, (40960, 5))

    def test_raises_on_empty_directory(self) -> None:
        files_path = Path('tests/src/read/mock_data/empty_folder')
        column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        with self.assertRaises(ValueError) as context:
            read_nasa_vibration_files_in_directory(files_path=files_path, column_names=column_names)
        self.assertIn("No files found in the directory", str(context.exception))

    