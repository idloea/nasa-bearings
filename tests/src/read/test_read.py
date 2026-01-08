import unittest
from pathlib import Path
from src.read import read_nasa_vibration_file, read_nasa_vibration_files_in_directory
import shutil

class TestRead(unittest.TestCase):
    def setUp(self) -> None:
        sampling_frequency = 20000  # Hz
        self.signal_resolution = 1 /sampling_frequency
        self.empty_folder_path = Path('tests/src/read/mock_data/empty_folder')
        self.empty_folder_path.mkdir(parents=True, exist_ok=True)  
        
    def tearDown(self):
        if self.empty_folder_path.exists():
            shutil.rmtree(self.empty_folder_path)              

    def test_read_nasa_vibration_file(self) -> None:
        file_path = Path('tests/src/read/mock_data/2003.10.22.12.06.24')
        sensors = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                        'channel_5', 'channel_6', 'channel_7', 'channel_8']
        df = read_nasa_vibration_file(file_path=file_path, sensors=sensors,
                                      signal_resolution=self.signal_resolution)
        shape = df.shape
        expected_shape = (20480, 9)
        expected_column_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5',
                                 'channel_6', 'channel_7', 'channel_8', 'measurement_time_in_seconds']
        self.assertEqual(expected_shape, shape)
        self.assertEqual(expected_column_names, df.columns)

    def test_read_nasa_vibration_file_raises_on_missing_file(self) -> None:
        file_path = Path('tests/src/read/mock_data/non_existent_file')
        sensors = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                   'channel_5', 'channel_6', 'channel_7', 'channel_8']
        with self.assertRaises(FileNotFoundError) as context:
            read_nasa_vibration_file(file_path=file_path, sensors=sensors,
                                     signal_resolution=self.signal_resolution)
        self.assertIn("File not found", str(context.exception))

    def test_read_nasa_vibration_file_with_faulty_sensors(self) -> None:
        file_path = Path('tests/src/read/mock_data/2004.04.18.02.42.55')
        sensors = ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        acceptable_sensor_range = 0.01
        df = read_nasa_vibration_file(file_path=file_path, sensors=sensors,
                                      signal_resolution=self.signal_resolution,
                                      acceptable_sensor_range=acceptable_sensor_range)
        shape = df.shape
        expected_shape = (0, 0)
        self.assertEqual(expected_shape, shape)


    def test_read_nasa_vibration_files_in_directory(self) -> None:
        files_path = Path('tests/src/read/mock_data/mock_folder')
        sensors =  ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        acceptable_sensor_range = 0.01
        df_list = read_nasa_vibration_files_in_directory(files_path=files_path, sensors=sensors,
                                                         signal_resolution=self.signal_resolution,
                                                         acceptable_sensor_range=acceptable_sensor_range)
        self.assertEqual(len(df_list), 2)

    def test_raises_on_empty_directory(self) -> None:
        sensors = ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        with self.assertRaises(ValueError) as context:
            read_nasa_vibration_files_in_directory(files_path=self.empty_folder_path, sensors=sensors,
                                                   signal_resolution=self.signal_resolution)
        self.assertIn("No files found in the directory", str(context.exception))

    def test_read_nasa_vibration_file_with_time(self) -> None:
        """Test if the single file reader returns the duration when requested."""
        file_path = Path('tests/src/read/mock_data/2003.10.22.12.06.24')
        sensors = ['channel_1', 'channel_2', 'channel_3', 'channel_4',
                   'channel_5', 'channel_6', 'channel_7', 'channel_8']
        
        df, duration = read_nasa_vibration_file(
            file_path=file_path, 
            sensors=sensors,
            signal_resolution=self.signal_resolution,
            return_time=True
        )
        
        self.assertIsInstance(duration, float)
        self.assertGreater(duration, 0)
        self.assertEqual(df.shape[1], 9)

    def test_read_nasa_vibration_files_in_directory_with_time(self) -> None:
        """Test if the directory reader returns a list of durations."""
        files_path = Path('tests/src/read/mock_data/mock_folder')
        sensors = ['channel_1', 'channel_2', 'channel_3', 'channel_4']
        
        dfs, durations = read_nasa_vibration_files_in_directory(
            files_path=files_path, 
            sensors=sensors,
            signal_resolution=self.signal_resolution,
            return_time=True
        )
        
        self.assertEqual(len(dfs), len(durations))
        self.assertTrue(all(isinstance(t, float) for t in durations))

    def test_read_nasa_vibration_files_column_structure(self) -> None:
        """Verify that 'file_name' is the first column in the directory reader output."""
        files_path = Path('tests/src/read/mock_data/mock_folder')
        sensors = ['channel_1', 'channel_2']
        
        df_list = read_nasa_vibration_files_in_directory(
            files_path=files_path, 
            sensors=sensors,
            signal_resolution=self.signal_resolution
        )
        
        for df in df_list:
            # Check if file_name is the first column
            self.assertEqual(df.columns[0], 'file_name')
            # Check if it's populated
            self.assertFalse(df['file_name'].is_empty())

    def test_read_nasa_vibration_files_skips_faulty(self) -> None:
        """
        Verify that files where all sensors are faulty are 
        skipped (not included in the returned list).
        """
        # Note: This assumes 'mock_folder' contains at least one file 
        # that would be cleared by a high acceptable_sensor_range
        files_path = Path('tests/src/read/mock_data/mock_folder')
        sensors = ['channel_1', 'channel_2']
        
        # Using an extremely high range to force files to be 'empty' after drop_faulty_sensor_data
        extreme_range = 9999.0 
        
        # We expect a logger warning and the list to be empty if all files are 'faulty'
        df_list = read_nasa_vibration_files_in_directory(
            files_path=files_path, 
            sensors=sensors,
            signal_resolution=self.signal_resolution,
            acceptable_sensor_range=extreme_range
        )
        
        self.assertEqual(len(df_list), 0)