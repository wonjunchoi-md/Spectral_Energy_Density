from abc import ABC, abstractmethod


class AbstractTrajectoryReader(ABC):
    """Provides a way to iterate through a molecular dynamics (MD) trajectory
    file.

    Each frame/time-step is returned as a trajectory_frame.
    """

    # unit conversion tables
    lengthunits_to_nm_table = {
        'Angstrom': 1.0,
        'nm': 10.0,
        'pm': 1e3,
        'fm': 1e6,
    }

    timeunits_to_fs_table = {
        'fs': 1.0,
        'ps': 1000,
        'ns': 1000000,
    }

    @abstractmethod
    def __iter__(self):
        """ Iterates through the trajectory file, frame by frame. """
        pass

    @abstractmethod
    def __next__(self):
        """ Gets next trajectory frame. """
        pass

    @abstractmethod
    def close(self):
        """ Closes down, release resources etc. """
        pass
