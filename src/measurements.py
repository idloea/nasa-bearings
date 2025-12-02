from typing import List
import polars as pl

def drop_faulty_sensor_data(df: pl.DataFrame, sensors: List[str],
                            acceptable_range: float) -> pl.DataFrame:
    """
    Set sensor columns to None if their value range is below the acceptable threshold.

    :param df: input DataFrame containing sensor data
    :param sensors: list of sensor column names to check
    :param acceptable_range: minimum required range for sensor values
    :return: DataFrame with faulty sensor columns set to None
    """

    result_df = df
    faulty_sensors = [sensor for sensor in sensors if df[sensor].max() - df[sensor].min() < acceptable_range]
    if faulty_sensors == sensors:
        result_df = pl.DataFrame()
    else:
        for faulty_sensor in faulty_sensors:
            result_df = result_df.with_columns(pl.lit(None).alias(faulty_sensor))

    return result_df