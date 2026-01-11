import unittest
from src.measurements import drop_faulty_sensor_data
import polars as pl
from polars.testing import assert_frame_equal


class TestMeasurements(unittest.TestCase):
    def test_drop_faulty_sensor_data(self) -> None:
        df = pl.DataFrame({
            'measurement_time_in_seconds': [0, 1, 2, 3, 4],
            'channel_1': [0.001, -0.001, -0.003, 0.004, -0.005],
            'channel_2': [0.5, 0.4, 0.3, 0.2, 0.1]
        })
        expected_df = pl.DataFrame({
            'measurement_time_in_seconds': [0, 1, 2, 3, 4],
            'channel_2': [0.5, 0.4, 0.3, 0.2, 0.1]
        })
        acceptable_range = 0.01
        result_df = drop_faulty_sensor_data(df=df,
                                            sensors=['channel_1', 'channel_2'],
                                            acceptable_range=acceptable_range)
        assert_frame_equal(result_df, expected_df)

    def test_drop_unorganized_faulty_sensor_data(self) -> None:
        df = pl.DataFrame({
            'measurement_time_in_seconds': [0, 1, 2, 3, 4],
            'channel_1': [0.001, -0.001, -0.003, 0.004, -0.005],
            'channel_3': [0.5, 0.4, 0.3, 0.2, 0.1],
            'channel_2': [0.5, 0.4, 0.3, 0.2, 0.1],
            'channel_5': [0.001, -0.001, -0.003, 0.004, -0.005],
            'channel_4': [0.5, 0.4, 0.3, 0.2, 0.1]
        })
        expected_df = pl.DataFrame({
            'measurement_time_in_seconds': [0, 1, 2, 3, 4],
            'channel_3': [0.5, 0.4, 0.3, 0.2, 0.1],            
            'channel_2': [0.5, 0.4, 0.3, 0.2, 0.1],
            'channel_4': [0.5, 0.4, 0.3, 0.2, 0.1]            
        })
        acceptable_range = 0.01
        result_df = drop_faulty_sensor_data(df=df,
                                            sensors=['channel_1', 'channel_4', 'channel_3', 'channel_5', 'channel_2'],
                                            acceptable_range=acceptable_range)
        assert_frame_equal(result_df, expected_df)

    def test_drop_faulty_sensor_data_all_faulty_sensors(self) -> None:
        df = pl.DataFrame({
            'measurement_time_in_seconds': [0, 1, 2, 3, 4],
            'channel_1': [0.001, -0.001, -0.003, 0.004, -0.005],
            'channel_2': [0.001, -0.001, -0.003, 0.004, -0.005]
        })
        expected_df = pl.DataFrame()
        acceptable_range = 0.01
        result_df = drop_faulty_sensor_data(df=df,
                                            sensors=['channel_1', 'channel_2'],
                                            acceptable_range=acceptable_range)
        assert_frame_equal(result_df, expected_df)
