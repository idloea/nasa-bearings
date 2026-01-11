from typing import List
import polars as pl

def drop_faulty_sensor_data(df: pl.DataFrame, 
                            sensors: List[str],
                            acceptable_range: float) -> pl.DataFrame:
    """
    Drop sensor columns if their value range is below the acceptable threshold.

    :param df: input DataFrame containing sensor data
    :param sensors: list of sensor column names to check
    :param acceptable_range: minimum required range for sensor values
    :return: DataFrame without faulty sensor columns or empty DataFrame if all are faulty
    """
    faulty_sensors = [sensor for sensor in sensors if df[sensor].max() - df[sensor].min() < acceptable_range]

    if set(faulty_sensors) == set(sensors):       
        return pl.DataFrame()
         
    return df.drop(faulty_sensors)
