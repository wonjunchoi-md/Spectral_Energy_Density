import numpy as np


class TimeAverager:
    """Naive special purpose averager class used in dynasor to collect and time-average arrays
    obtained from sliding time-window averaging.

    It assists with keeping track of how many data samples have been added to each slot.

    It will time-average arrays of shape ``(Nq, time_window)`` where ``Ǹq`` is the
    number of q-points and ``time_window`` is the size of the time window.

    Parameters
    ----------
    time_window
        size of the time window in which the time-average happens
    array_length
        length of the array to be averaged for each time-lag, i.e. number of q-points
    """

    def __init__(self, time_window: int, array_length: int):
        assert time_window >= 1
        self._time_window = time_window
        self._array_length = array_length

        self._counts = np.zeros(time_window, dtype=int)
        self._arrays = [np.zeros(array_length) for _ in range(time_window)]

    def add_sample(self, time_lag: int, sample: np.ndarray):
        assert len(sample) == self._array_length
        self._counts[time_lag] += 1
        self._arrays[time_lag] += sample

    def get_average_at_timelag(self, time_lag: int):
        if self._counts[time_lag] == 0:
            array = np.full((self._array_length, ), np.nan)
            return array
        return self._arrays[time_lag] / self._counts[time_lag]

    def get_average_all(self):
        """
        Returns an averaged array of shape ``(array_length, time_window)``
        """
        return np.array([self.get_average_at_timelag(t) for t in range(self._time_window)]).T
