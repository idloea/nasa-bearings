from typing import Union
import numpy as np

def calculate_resolution(sampling_frequency: Union[int, float]) -> Union[int, float]:
    """
    Calculate the signal resolution in seconds from the sampling frequency.

    :param sampling_frequency: sampling frequency in Hertz
    :return: signal resolution in seconds
    """
    return 1 / sampling_frequency

def calculate_root_mean_square(y: np.ndarray) -> float:
    """
    Calculate the root mean square (RMS) of a signal.

    :param y: Array or iterable of signal values
    :return: RMS value of the signal
    """
    return np.sqrt(np.mean(y**2))

def calculate_absolute_peak(y: np.ndarray) -> float:
    """
    Calculate the peak (maximum absolute value) of a signal.

    :param y: Array or iterable of signal values
    :return: Peak value of the signal
    """
    return np.max(np.abs(y))

def calculate_crest_factor(y: np.ndarray) -> float:
    """
    Calculates the crest factor of a signal.

    :param y: Array or iterable of signal values
    :return: Crest factor value of the signal
    """
    
    peak = calculate_absolute_peak(y=y)
    root_mean_square = calculate_root_mean_square(y=y)
    return peak / root_mean_square

def calculate_average_rectified_value(y: np.ndarray) -> float:
    """
    Calculates the Average Rectified Value (ARV) of a given input signal.
    
    :param y: Array or iterable of signal values
    :return: Average Rectified Value (ARV) of the signal
    """

    return np.mean(np.abs(y))

def calculate_shape_factor(y: np.ndarray) -> float:
    """
    Calculates the shape factor of a given input signal.
    
    Args:
        signal (array-like): The input time-series signal.
        
    Returns:
        float: The shape factor value.
    """
    rms = calculate_root_mean_square(y=y)
    arv = calculate_average_rectified_value(y=y)
    return rms / arv
