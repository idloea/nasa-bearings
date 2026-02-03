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
                   window: Union[None, tuple, str] = ('kaiser', 20),
                   number_of_points_per_segment: int = 2048,
                   number_of_overlapping_points_between_segments: int = None) -> tuple:
    """
    Estimate the power spectral density (PSD) of a time-domain waveform using Welch's method.

    Parameters
    ----------
    waveform : 1-D array_like
        Time-domain signal samples.
    sampling_frequency : float
        Sampling frequency of the waveform in Hertz.
    window : {None, str, tuple}, optional
        Window specification passed to ``scipy.signal.welch`` (for example
        ``('kaiser', 20)``). Defaults to ``('kaiser', 20)``.
    number_of_points_per_segment : int, optional
        Number of points per segment (``nperseg``) used by Welch's method.
        Defaults to 2048.
    number_of_overlapping_points_between_segments : int, optional
        Number of points to overlap between segments (``noverlap``). If
        ``None``, defaults to ``number_of_points_per_segment // 2``.

    Returns
    -------
    frequencies : ndarray
        1-D array of frequency bin centers in Hertz.
    amplitudes : ndarray
        Estimated power spectral density (spectrum) for each frequency bin.
    """

    if number_of_overlapping_points_between_segments is None:
        number_of_overlapping_points_between_segments = number_of_points_per_segment // 2
    
    frequencies, amplitudes = signal.welch(x=waveform, 
                                           fs=sampling_frequency, 
                                           window=window, 
                                           nperseg=number_of_points_per_segment, 
                                           noverlap=number_of_overlapping_points_between_segments, 
                                           scaling='spectrum')
    
    return frequencies, amplitudes

def band_pass_filter(y: np.ndarray,
                     sampling_frequency: float,
                     low_cutoff_frequency: float,
                     high_cutoff_frequency: float,
                     filter_order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to a signal.

    Parameters
    ----------
    y : ndarray
        Input signal to be filtered.
    sampling_frequency : float
        Sampling frequency of the signal in Hz.
    low_cutoff_frequency : float
        Low cutoff frequency of the band-pass filter in Hz.
    high_cutoff_frequency : float
        High cutoff frequency of the band-pass filter in Hz.
    filter_order : int, optional
        Order of the Butterworth filter. Default is 4.

    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    low = low_cutoff_frequency / nyquist_frequency
    high = high_cutoff_frequency / nyquist_frequency

    b, a = signal.butter(filter_order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, y)

    return filtered_signal

def low_pass_filter(y: np.ndarray,
                    sampling_frequency: float,
                    cutoff_frequency: float,
                    filter_order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to a signal.

    Parameters
    ----------
    y : ndarray
        Input signal to be filtered.
    sampling_frequency : float
        Sampling frequency of the signal in Hz.
    cutoff_frequency : float
        Cutoff frequency of the low-pass filter in Hz.
    filter_order : int, optional
        Order of the Butterworth filter. Default is 4.

    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency

    b, a = signal.butter(filter_order, normal_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, y)

    return filtered_signal

def high_pass_filter(y: np.ndarray,
                     sampling_frequency: float,
                     cutoff_frequency: float,
                     filter_order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to a signal.

    Parameters
    ----------
    y : ndarray
        Input signal to be filtered.
    sampling_frequency : float
        Sampling frequency of the signal in Hz.
    cutoff_frequency : float
        Cutoff frequency of the high-pass filter in Hz.
    filter_order : int, optional
        Order of the Butterworth filter. Default is 4.

    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency

    b, a = signal.butter(filter_order, normal_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, y)

    return filtered_signal

def envelope(y: np.ndarray) -> np.ndarray:
    """
    Compute the envelope of a signal using the Hilbert transform.

    Parameters
    ----------
    y : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Envelope of the input signal.
    """
    analytic_signal = signal.hilbert(y)
    envelope_signal = np.abs(analytic_signal)
    return envelope_signal
