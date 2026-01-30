from typing import Union
import numpy as np
from scipy import signal

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

def power_spectrum(waveform: np.ndarray, 
                   sampling_frequency: float,
                   window=('kaiser', 20),
                   number_of_points_per_segment: int = 2048,
                   number_of_overlapping_points_between_segments: int = None) -> tuple:

    """
    Estimate the power spectral density (PSD) of a time-domain waveform using Welch's method.

    :param waveform: 1-D array of time-domain signal samples.
    :param sampling_frequency: Sampling frequency of the waveform in Hz.
    :param window: Window specification passed to ``scipy.signal.welch`` (e.g.
        ``('kaiser', 20)``). Defaults to ``('kaiser', 20)``.
    :param number_of_points_per_segment: Number of points per segment (``nperseg``)
        used by Welch's method. Defaults to ``2048``.
    :param number_of_overlapping_points_between_segments: Number of points to
        overlap between segments (``noverlap``). If ``None``, defaults to
        ``number_of_points_per_segment // 2``.
    :return: Tuple ``(frequencies, power_spectrum)`` where ``frequencies`` is a
        1-D NumPy array of frequency bin centers (Hz) and ``power_spectrum`` is
        the estimated power spectral density for each frequency bin.
    """

    if number_of_overlapping_points_between_segments is None:
        number_of_overlapping_points_between_segments = number_of_points_per_segment // 2
    
    frequencies, power_spectrum = signal.welch(x=waveform, 
                                               fs=sampling_frequency, 
                                               window=window, 
                                               nperseg=number_of_points_per_segment, 
                                               noverlap=number_of_overlapping_points_between_segments, 
                                               scaling='spectrum')
    
    return frequencies, power_spectrum