from typing import List
import pandas as pd

def drop_faulty_sensor_data(df: pd.DataFrame, sensors: List[str],
                            acceptable_range: float) -> pd.DataFrame:
    """
    Set sensor columns to pd.NA if their value range is below the acceptable threshold.

    :param df: input DataFrame containing sensor data
    :param sensors: list of sensor column names to check
    :param acceptable_range: minimum required range for sensor values
    :return: DataFrame with faulty sensor columns set to pd.NA
    """

    result_df = df.copy()
    faulty_sensors = [sensor for sensor in sensors if df[sensor].max() - df[sensor].min() < acceptable_range]
    result_df[faulty_sensors] = pd.NA
    return result_df