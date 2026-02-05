class BaseSimulator:
    """
    Base class for simulators in the sbi-bax package.
    This class provides a common interface for all simulators, including
    methods for generating measurements and visualizing results.
    """

    def __init__(self, true_thetas=None):
        self.true_thetas = true_thetas

    # Define a function that returns what will be treated as a real measurement.
    def make_measurement(self, design_params):
        # Generate the image using the simulator
        return self.__call__(design_params, self.true_thetas)
