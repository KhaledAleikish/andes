"""
Base class for ANDES calculation routines.
"""

from andes.linsolvers.solverbase import Solver
from andes.core import Config
from collections import OrderedDict


class BaseRoutine:
    """
    Base routine class.

    Provides references to system, config, and solver.
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)
        self.config._deprecated['sparselib'] = 'Use [Runtime] section instead.'

        if config is not None:
            self.config.load(config)

        self.config.add(OrderedDict((('linsolve', 0),
                                     )))
        self.config.add_extra("_help",
                              linsolve="solve symbolic factorization each step (enable when KLU segfaults)",
                              )
        self.config.add_extra("_alt",
                              linsolve=(0, 1),
                              )

        sparselib = 'klu'
        if system is not None and hasattr(system, 'runtime'):
            sparselib = system.runtime.sparselib

        self.solver = Solver(sparselib=sparselib)
        self.exec_time = 0.0  # recorded time to execute the routine in seconds

    @property
    def class_name(self):
        return self.__class__.__name__

    def doc(self, max_width=78, export='plain'):
        """
        Routine documentation interface.
        """
        return self.config.doc(max_width, export)

    def init(self):
        """
        Routine initialization interface.
        """
        pass

    def run(self, **kwargs):
        """
        Routine main entry point.
        """
        raise NotImplementedError

    def summary(self, **kwargs):
        """
        Summary interface
        """
        raise NotImplementedError

    def report(self, **kwargs):
        """
        Report interface.
        """
        raise NotImplementedError
