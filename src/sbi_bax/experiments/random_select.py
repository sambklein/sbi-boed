from sbi_bax.experiments.base import BaseAcquisitionExperiment
import torch
import logging

# Set up logging
log = logging.getLogger(__name__)


class RandomAcquisition(BaseAcquisitionExperiment):
    """
    An experiment that randomly selects points from a grid for acquisition.
    This is a simple baseline that does not use any learned model or posterior.
    An SBI model is still trained, but it is not used for acquisition.
    """

    def run_acquisition(self, *args, **kwargs) -> torch.Tensor:
        """
        Run the acquisition function to select the next measurement point.
        Args:
            data_posterior: The posterior distribution p(θ | D_t).
            n_measured (int): The number of measurements taken so far.
            output_dir (Path): Directory to save acquisition results.
        Returns:
            torch.Tensor: The next measurement point.
        """
        log.info("Using RandomAcquisition for acquisition")
        # Randomly select a point from the grid
        return self.x_grid[torch.randint(0, len(self.x_grid), (1,)).item()]
