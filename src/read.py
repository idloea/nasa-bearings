from pathlib import Path
import pandas as pd
from typing import List

def read_nasa_vibration_file(path: Path, column_names: List[str]) -> pd.DataFrame:
    """
    Read one vibration file from the IMS Bearing dataset obtained from NASAs acoustics and vibrations datasets.
    According to its documentation, the channels belong to the following bearings:
    - Bearing 1: Channel 1 and Channel 2
    - Bearing 2: Channel 3 and Channel 4
    - Bearing 3: Channel 5 and Channel 6
    - Bearing 4: Channel 7 and Channel 8

    The file Recording Interval is every 10 minutes (except the first 43 files were taken every 5 minutes for the
    1st test)

    :param path: path to the location of the vibration file
    :param column_names: name of the columns to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :return: Pandas DataFrame containing the vibration data for different channels or sensors
    """
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = column_names
    return df
