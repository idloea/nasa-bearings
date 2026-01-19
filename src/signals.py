from typing import Union

def resolution(sampling_frequency: Union[int, float]) -> Union[int, float]:
    """
    Calculate the signal resolution in seconds from the sampling frequency.

    :param sampling_frequency: sampling frequency in Hertz
    :return: signal resolution in seconds
    """
    return 1 / sampling_frequency
