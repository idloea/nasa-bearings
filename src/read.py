from pathlib import Path
from typing import List, Union, Tuple
import os
from src.measurements import drop_faulty_sensor_data
from loguru import logger
import polars as pl
import numpy as np
import time

def read_nasa_vibration_file(file_path: Path, sensors: List[str],
                             signal_resolution: Union[int, float],
                             acceptable_sensor_range: Union[float, None]=None,
                             return_time: bool = False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, float]]:
    """
    Read one vibration file from the IMS Bearing dataset obtained from NASAs acoustics and vibrations datasets.
    According to its documentation, the channels belong to the following bearings:
    Check the "Readme Document for IMS Bearing Data.pdf" before loading the data since the channel or sensor
    settings are different depending on the test (1st, 2nd, or 3rd).

    :param file_path: path to the location of the vibration file
    :param sensors: name of the channels or sensors to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :param signal_resolution: resolution of the signal in seconds
    """

    start_time = time.perf_counter()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pl.read_csv(file_path, separator='\t', has_header=False, new_columns=sensors)
    number_of_data_points = df.shape[0]
    measurement_time_in_seconds = np.arange(number_of_data_points) * signal_resolution
    df = df.with_columns(pl.Series("measurement_time_in_seconds", measurement_time_in_seconds))
    
    if acceptable_sensor_range is not None:
        df = drop_faulty_sensor_data(df=df, sensors=sensors, acceptable_range=acceptable_sensor_range)

    end_time = time.perf_counter()
    duration = end_time - start_time

    if return_time:
        return df, duration
    
    return df

def read_nasa_vibration_files_in_directory(files_path: Path, 
                                           sensors: List[str],
                                           signal_resolution: Union[int, float],
                                           acceptable_sensor_range: Union[float, None]=None,
                                           return_file_reading_time: bool = False) -> Union[List[pl.DataFrame], Tuple[List[pl.DataFrame], List[float]]]:
    """
    Read all vibration files in a directory and return a list of DataFrames.

    :param files_path: path to the directory containing the vibration files
    :param sensors:  name of the channels or sensors to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :param signal_resolution: resolution of the signal in seconds
    :param acceptable_sensor_range: if provided, sensors with a value range below this threshold will be set to None
    :param return_time: if True, a list of reading times is returned alongside the list of DataFrames
    :return: List of Polars DataFrames containing the vibration data for different channels or sensors
    """
    list_of_files =  os.listdir(files_path)
    number_of_files = len(list_of_files)
    if number_of_files == 0:
        raise ValueError(f"No files found in the directory: {files_path}")
    
    dataframes = []
    file_reading_time = []
    
    for file in list_of_files:
        file_path = files_path.joinpath(file)
        
        result = read_nasa_vibration_file(
            file_path=file_path, 
            sensors=sensors,
            signal_resolution=signal_resolution,
            acceptable_sensor_range=acceptable_sensor_range,
            return_time=return_file_reading_time
        )

        if return_file_reading_time:
            df, duration = result
        else:
            df = result

        if df.is_empty():
            logger.warning(f'All sensors in file {file} are faulty for the defined acceptable_sensor_range '
                        f'of {acceptable_sensor_range}. Skipping this file.')
            continue
            
        df = df.with_columns(pl.lit(file).alias('file_name'))
        cols = ['file_name'] + [col for col in df.columns if col != 'file_name']
        df = df.select(cols)    
        
        dataframes.append(df)
        if return_file_reading_time:
            file_reading_time.append(duration)

    number_of_read_files = len(dataframes)
    number_of_discarded_files = number_of_files - number_of_read_files
    logger.info(f'{number_of_discarded_files} files were discarded.') 
    logger.info(f'{number_of_read_files} files were read successfully.')
    
    if return_file_reading_time:
        return dataframes, file_reading_time
        
    return dataframes