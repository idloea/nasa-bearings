from typing import List
import polars as pl

def drop_faulty_sensor_data(df: pl.DataFrame, 
                            sensors: List[str],
                            acceptable_range: float) -> pl.DataFrame:
    """
    Drop sensor columns if their value range is below the acceptable threshold.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing sensor data.
    sensors : list[str]
        List of sensor column names to check.
    acceptable_range : float
        Minimum required range for sensor values. Sensors with (max - min)
        below this threshold are considered faulty and removed.

    Returns
    -------
    pl.DataFrame
        DataFrame without faulty sensor columns. Returns an empty DataFrame
        if all provided sensors are considered faulty.
    """
    faulty_sensors = [sensor for sensor in sensors if df[sensor].max() - df[sensor].min() < acceptable_range]

    if set(faulty_sensors) == set(sensors):       
        return pl.DataFrame()
         
    return df.drop(faulty_sensors)
