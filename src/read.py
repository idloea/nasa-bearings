from pathlib import Path
import pandas as pd
from typing import List, Union
import os
from src.measurements import drop_faulty_sensor_data
from loguru import logger


def read_nasa_vibration_file(file_path: Path, sensors: List[str],
                             signal_resolution: Union[int, float],
                             acceptable_sensor_range: Union[float, None]=None) -> pd.DataFrame:
    """
    Read one vibration file from the IMS Bearing dataset obtained from NASAs acoustics and vibrations datasets.
    According to its documentation, the channels belong to the following bearings:
    Check the "Readme Document for IMS Bearing Data.pdf" before loading the data since the channel or sensor
    settings are different depending on the test (1st, 2nd, or 3rd).

    :param file_path: path to the location of the vibration file
    :param sensors: name of the channels or sensors to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :param signal_resolution: resolution of the signal in seconds
    :param acceptable_sensor_range: if provided, sensors with a value range below this threshold will be set to pd.NA
    :return: Pandas DataFrame containing the vibration data for different channels or sensors
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep='\t', header=None, names=sensors)
    df['measurement_time_in_seconds'] = df.index * signal_resolution

    if acceptable_sensor_range is not None:
        df = drop_faulty_sensor_data(df=df, sensors=sensors, acceptable_range=acceptable_sensor_range)

    return df

def read_nasa_vibration_files_in_directory(files_path: Path, sensors: List[str],
                                           signal_resolution: Union[int, float],
                                           acceptable_sensor_range: Union[float, None]=None) -> List[pd.DataFrame]:
    """
    Read all vibration files in a directory and return a DataFrames.

    :param files_path: path to the directory containing the vibration files
    :param sensors:  name of the channels or sensors to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :param signal_resolution: resolution of the signal in seconds
    :param acceptable_sensor_range: if provided, sensors with a value range below this threshold will be set to pd.NA
    :return: List of Pandas DataFrames containing the vibration data for different channels or sensors
    """
    list_of_files =  os.listdir(files_path)
    if len(list_of_files) == 0:
        raise ValueError(f"No files found in the directory: {files_path}")
    
    dataframes = []
    for file in list_of_files:
        file_path = files_path.joinpath(file)
        df = read_nasa_vibration_file(file_path=file_path, sensors=sensors,
                                      signal_resolution=signal_resolution,
                                      acceptable_sensor_range=acceptable_sensor_range)
        if df.empty:
            logger.info(f'All sensors in file {file} are faulty for the defined acceptable_sensor_range '
                        f'of {acceptable_sensor_range}. Skipping this file.')
            continue
        df['file_name'] = file
        cols = ['file_name'] + [col for col in df.columns if col != 'file_name']
        df = df[cols]    
        dataframes.append(df)
    return dataframes