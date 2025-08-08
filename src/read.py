from pathlib import Path
import pandas as pd
from typing import List

def read_nasa_vibration_file(path: Path, column_names: List[str]) -> pd.DataFrame:
    """
    Read one vibration file from the IMS Bearing dataset obtained from NASAs acoustics and vibrations datasets.
    According to its documentation, the channels belong to the following bearings:
    Check the "Readme Document for IMS Bearing Data.pdf" before loading the data since the channel or sensor
    settings are different depending on the test (1st, 2nd, or 3rd).

    :param path: path to the location of the vibration file
    :param column_names: name of the columns to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :return: Pandas DataFrame containing the vibration data for different channels or sensors
    """
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = column_names
    return df
