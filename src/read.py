from pathlib import Path
import pandas as pd
from typing import List
import os

def read_nasa_vibration_file(file_path: Path, column_names: List[str]) -> pd.DataFrame:
    """
    Read one vibration file from the IMS Bearing dataset obtained from NASAs acoustics and vibrations datasets.
    According to its documentation, the channels belong to the following bearings:
    Check the "Readme Document for IMS Bearing Data.pdf" before loading the data since the channel or sensor
    settings are different depending on the test (1st, 2nd, or 3rd).

    :param file_path: path to the location of the vibration file
    :param column_names: name of the columns to be used. Example:
    ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7', 'channel_8']
    :return: Pandas DataFrame containing the vibration data for different channels or sensors
    """
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = column_names
    return df

def read_nasa_vibration_files_in_directory(files_path: Path, column_names: List[str]) -> List[pd.DataFrame]:
    """
    Read all vibration files in a directory and return a DataFrames.

    :param files_path: path to the directory containing the vibration files
    :param column_names: name of the columns to be used for each DataFrame
    :return: List of Pandas DataFrames containing the vibration data for different channels or sensors
    """
    list_of_files =  os.listdir(files_path)
    if len(list_of_files) == 0:
        raise ValueError(f"No files found in the directory: {files_path}")
    
    dataframes = []
    for file in list_of_files:
        file_path = files_path.joinpath(file)
        df = read_nasa_vibration_file(file_path=file_path, column_names=column_names)
        df['file_name'] = file
        cols = ['file_name'] + [col for col in df.columns if col != 'file_name']
        df = df[cols]    
        dataframes.append(df)
    return dataframes