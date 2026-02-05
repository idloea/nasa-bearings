from typing import Union
from dataclasses import dataclass
import numpy as np
from scipy import signal as scipy_signal

@dataclass
class Signal:  # TODO: does it make sense to have this class here?
    """
    Data structure holding a signal's domain and amplitude values.

    Attributes
    ----------
    x : np.ndarray
        Domain values (e.g., time, frequency, or quefrency).
    y : np.ndarray
        Signal amplitude values corresponding to `x`.
    """
    x: np.ndarray
    y: np.ndarray

def band_pass_filter(signal: Signal,
                     sampling_frequency: float,
                     low_cutoff_frequency: float,
                     high_cutoff_frequency: float,
                     filter_order: int = 4) -> Signal:
    """
    Apply a Butterworth band-pass filter to a signal.

    Parameters
    ----------
    signal : Signal
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
    Signal
        Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    low = low_cutoff_frequency / nyquist_frequency
    high = high_cutoff_frequency / nyquist_frequency

    b, a = scipy_signal.butter(filter_order, [low, high], btype='band')
    filtered_y = scipy_signal.filtfilt(b, a, signal.y)
    
    return Signal(x=signal.x, y=filtered_y)

def low_pass_filter(signal: Signal,
                    sampling_frequency: float,
                    cutoff_frequency: float,
                    filter_order: int = 4) -> Signal:
    """
    Apply a Butterworth low-pass filter to a signal.

    Parameters
    ----------
    signal : Signal
        Input signal to be filtered.
    sampling_frequency : float
        Sampling frequency of the signal in Hz.
    cutoff_frequency : float
        Cutoff frequency of the low-pass filter in Hz.
    filter_order : int, optional
        Order of the Butterworth filter. Default is 4.

    Returns
    -------
    Signal
        Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency

    b, a = scipy_signal.butter(filter_order, normal_cutoff, btype='low')
    filtered_y = scipy_signal.filtfilt(b, a, signal.y)

    return Signal(x=signal.x, y=filtered_y)

def high_pass_filter(signal: Signal,
                     sampling_frequency: float,
                     cutoff_frequency: float,
                     filter_order: int = 4) -> Signal:
    """
    Apply a Butterworth high-pass filter to a signal.

    Parameters
    ----------
    signal : Signal
        Input signal to be filtered.
    sampling_frequency : float
        Sampling frequency of the signal in Hz.
    cutoff_frequency : float
        Cutoff frequency of the high-pass filter in Hz.
    filter_order : int, optional
        Order of the Butterworth filter. Default is 4.

    Returns
    -------
    Signal
        Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency

    b, a = scipy_signal.butter(filter_order, normal_cutoff, btype='high')
    filtered_y = scipy_signal.filtfilt(b, a, signal.y)

    return Signal(x=signal.x, y=filtered_y)

def envelope(signal: Signal, remove_dc_offset: bool = True) -> Signal:
    """
    Compute the envelope of a signal using the Hilbert transform.

    Parameters
    ----------
    signal : Signal
        Input signal.
    remove_dc_offset : bool, optional
        Whether to remove the DC offset from the envelope. Default is True.

    Returns
    -------
    Signal
        Envelope of the input signal.
    """
    analytic_signal = scipy_signal.hilbert(signal.y)
    envelope_y = np.abs(analytic_signal)
    if remove_dc_offset:
        envelope_y = envelope_y - np.mean(envelope_y)
    return Signal(x=signal.x, y=envelope_y)    

def power_spectrum(signal: Signal, 
                   sampling_frequency: float,
                   window: Union[None, tuple, str] = ('kaiser', 20),
                   number_of_points_per_segment: int = 2048,
                   number_of_overlapping_points_between_segments: int = None) -> Signal:
    """
    Estimate the power spectral density (PSD) of a time-domain signal using Welch's method.

    Parameters
    ----------
    signal : Signal
        Input signal with time-domain samples.
    sampling_frequency : float
        Sampling frequency of the input signal in Hertz.
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
    Signal
        Power spectral density (spectrum) where x is frequencies and y is amplitudes.
    """

    if number_of_overlapping_points_between_segments is None:
        number_of_overlapping_points_between_segments = number_of_points_per_segment // 2
    
    frequencies, amplitudes = scipy_signal.welch(x=signal.y, 
                                           fs=sampling_frequency, 
                                           window=window, 
                                           nperseg=number_of_points_per_segment, 
                                           noverlap=number_of_overlapping_points_between_segments, 
                                           scaling='spectrum')
    
    return Signal(x=frequencies, y=amplitudes)


def process_signal(signal: Signal, steps: list[dict]) -> Signal:
    """
    Process a signal through a pipeline of specified operations.

    Parameters
    ----------
    signal : Signal
        Input signal.
    steps : list[dict]
        List of dictionaries, each defining a processing step.
        Each dictionary must have a 'type' key corresponding to the operation name:
        'band_pass_filter', 'low_pass_filter', 'high_pass_filter', 'envelope', 'power_spectrum'.
        Other keys in the dictionary are passed as arguments to the respective functions.

    Returns
    -------
    Signal
        Processed signal.
    
    Raises
    ------
    ValueError
        If an unknown step type is encountered.
    """
    processed_signal = signal
    for step in steps:
        step_type = step['type']
        
        if step_type == 'band_pass_filter':
            processed_signal = band_pass_filter(signal=processed_signal, 
                                              sampling_frequency=step['sampling_frequency'], 
                                              low_cutoff_frequency=step['low_cutoff_frequency'], 
                                              high_cutoff_frequency=step['high_cutoff_frequency'], 
                                              filter_order=step.get('filter_order', 4))
        elif step_type == 'low_pass_filter':
            processed_signal = low_pass_filter(signal=processed_signal, 
                                             sampling_frequency=step['sampling_frequency'], 
                                             cutoff_frequency=step['cutoff_frequency'], 
                                             filter_order=step.get('filter_order', 4))
        elif step_type == 'high_pass_filter':
            processed_signal = high_pass_filter(signal=processed_signal, 
                                              sampling_frequency=step['sampling_frequency'], 
                                              cutoff_frequency=step['cutoff_frequency'], 
                                              filter_order=step.get('filter_order', 4))
        elif step_type == 'envelope':
            processed_signal = envelope(signal=processed_signal)
        elif step_type == 'power_spectrum':
             processed_signal = power_spectrum(signal=processed_signal,
                                             sampling_frequency=step['sampling_frequency'],
                                             window=step.get('window', ('kaiser', 20)),
                                             number_of_points_per_segment=step.get('number_of_points_per_segment', 2048),
                                             number_of_overlapping_points_between_segments=step.get('number_of_overlapping_points_between_segments', None))
        else:
            raise ValueError(f"Unknown step type: {step_type}")  
    return processed_signal

