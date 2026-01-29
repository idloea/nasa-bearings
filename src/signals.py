from typing import Union
import numpy as np

def resolution(sampling_frequency: Union[int, float]) -> Union[int, float]:
    """
    Calculate the signal resolution in seconds from the sampling frequency.

    Parameters
    ----------
    sampling_frequency : int or float
        Sampling frequency in Hertz.

    Returns
    -------
    float
        Signal resolution in seconds (1 / sampling_frequency).
    """
    return 1 / sampling_frequency

def root_mean_square(y: np.ndarray) -> float:
    """
    Calculate the root mean square (RMS) of a signal.

    Parameters
    ----------
    y : ndarray
        Array or iterable of signal values.

    Returns
    -------
    float
        RMS value of the signal.
    """
    return np.sqrt(np.mean(y**2))

def absolute_peak(y: np.ndarray) -> float:
    """
    Calculate the peak (maximum absolute value) of a signal.

    Parameters
    ----------
    y : ndarray
        Array or iterable of signal values.

    Returns
    -------
    float
        Peak (maximum absolute) value of the signal.
    """
    return np.max(np.abs(y))

def crest_factor(y: np.ndarray) -> float:
    """
    Calculate the crest factor of a signal.

    The crest factor is the ratio of the peak value to the root mean square (RMS).

    Parameters
    ----------
    y : ndarray
        Array or iterable of signal values.

    Returns
    -------
    float
        Crest factor value of the signal.
    """

    peak = absolute_peak(y=y)
    rms = root_mean_square(y=y)
    return peak / rms

def average_rectified_value(y: np.ndarray) -> float:
    """
    Calculate the Average Rectified Value (ARV) of a signal.

    Parameters
    ----------
    y : ndarray
        Array or iterable of signal values.

    Returns
    -------
    float
        Average Rectified Value (ARV) of the signal.
    """

    return np.mean(np.abs(y))

def shape_factor(y: np.ndarray) -> float:
    """
    Calculate the shape factor of a signal.

    The shape factor is the ratio of the RMS to the average rectified value (ARV).

    Parameters
    ----------
    y : ndarray
        Array or iterable of signal values.

    Returns
    -------
    float
        Shape factor value of the signal.
    """

    rms = root_mean_square(y=y)
    arv = average_rectified_value(y=y)
    return rms / arv
