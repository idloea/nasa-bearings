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
    Read a single vibration file from the NASA IMS bearing dataset.

    The dataset channels correspond to different bearings depending on the
    specific test (1st, 2nd or 3rd). Refer to the repository's IMS Bearing
    README for channel mappings before loading data.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the vibration file to read.
    sensors : list[str]
        Names of channels/sensors to load. Example:
        ['channel_1', 'channel_2', 'channel_3', 'channel_4',
         'channel_5', 'channel_6', 'channel_7', 'channel_8']
    signal_resolution : int or float
        Resolution of the signal in seconds (sampling interval).
    acceptable_sensor_range : float or None, optional
        If provided, sensors with (max - min) below this threshold are
        considered faulty and removed from the returned DataFrame.
    return_time : bool, optional
        If True, return a tuple ``(df, duration)`` where ``duration`` is the
        time (in seconds) it took to read/process the file.

    Returns
    -------
    pl.DataFrame or tuple
        If ``return_time`` is False, returns a ``pl.DataFrame`` with the
        measurement data and an added ``measurement_time_in_seconds`` column.
        If ``return_time`` is True, returns a tuple ``(pl.DataFrame, float)``
        where the float is the read duration in seconds.
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

    Parameters
    ----------
    files_path : pathlib.Path
        Path to the directory containing the vibration files.
    sensors : list[str]
        Names of channels/sensors to load for each file.
    signal_resolution : int or float
        Resolution of the signal in seconds (sampling interval).
    acceptable_sensor_range : float or None, optional
        If provided, sensors with (max - min) below this threshold are
        considered faulty and removed from each file's DataFrame.
    return_file_reading_time : bool, optional
        If True, return alongside the list of DataFrames a list of file
        reading durations (in seconds) corresponding to each read file.

    Returns
    -------
    list[pl.DataFrame] or tuple
        If ``return_file_reading_time`` is False, returns a list of Polars
        DataFrames, one per successfully read file. If True, returns a tuple
        ``(list_of_dataframes, list_of_durations)``.

    Raises
    ------
    ValueError
        If the provided ``files_path`` directory contains no files.
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