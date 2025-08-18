from typing import Union

def get_resolution(sampling_frequency: Union[int, float]) -> Union[int, float]:
    return 1 / sampling_frequency
