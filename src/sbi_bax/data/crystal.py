import logging
from pathlib import Path
import torch
import os
from contextlib import redirect_stdout, redirect_stderr
from sbi_bax.calculate.crystal import count_bright_peaks
from sbi_bax.simulators.crystal_diffraction import (
    get_inverse_euler_angles,
)
from sbi_bax.utils.crystal_images import visualize_image
from sbi.utils import BoxUniform
import numpy as np
import matplotlib.pyplot as plt

from sbi_bax.utils.torch_utils import get_device

# Set up logging
log = logging.getLogger(__name__)


class CrystalData:
    def __init__(
        self,
        simulator,
        n_alg_path,
        data_bounds,
        prior_bounds,
        n_angles,
        n_xy,
        true_thetas=None,
        figure_dir=None,
    ):
        self.simulator = simulator
        self.n_alg_path = n_alg_path
        self.prior_bounds = prior_bounds
        # Extract low and high bounds for each parameter in the prior
        self.prior_low = [prior_bounds[0]] * 3 + [prior_bounds[2]] * 2
        self.prior_high = [prior_bounds[1]] * 3 + [prior_bounds[3]] * 2
        # Extract low and high bounds for each parameter in the data
        self.data_low = [data_bounds[0]] * 3 + [data_bounds[2]] * 2
        self.data_high = [data_bounds[1]] * 3 + [data_bounds[3]] * 2
        self.data_low_angles = torch.tensor(self.data_low[:3])
        self.data_high_angles = torch.tensor(self.data_high[:3])
        self.n_xy = n_xy
        self.n_angles = n_angles
        self.true_thetas = true_thetas
        self.prior = self.make_prior()
        self.x_grid = self.make_x_grid(data_bounds, n_angles, n_xy)
        # Define a directory for saving figures from this data object.
        if figure_dir is not None:
            self.figure_dir = Path(figure_dir)
            self.figure_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.figure_dir = None

    def make_prior(self):
        """
        Create a prior distribution over the angles.
        Args:
            prior_bounds (list): List of four elements defining the bounds for the prior.
            First three elements are the bounds for the angles (theta_x, theta_y, theta_z),
            and last two elements are the x-y bounds for the sample position.
        Returns:
            BoxUniform: A uniform prior distribution over the angles.
        """
        return BoxUniform(
            low=torch.tensor(self.prior_low, dtype=torch.float32),
            high=torch.tensor(self.prior_high, dtype=torch.float32),
            device=get_device(),
        )

    def make_x_grid(self, bounds, n_angles, n_xy, n_max=None):
        # Build 1D grids
        th = np.linspace(bounds[0], bounds[1], n_angles)
        xs = np.linspace(bounds[2], bounds[3], n_xy)
        # Stack last axis and flatten
        stacked = np.stack(np.meshgrid(th, th, th, xs, xs, indexing="ij"), axis=-1)
        # Convert to tensor
        full_grid = torch.tensor(stacked.reshape(-1, 5), dtype=torch.float32)
        # Randomly shuffle the grid
        full_grid = full_grid[torch.randperm(full_grid.shape[0])]
        if n_max is not None:
            # Limit the number of angles to n_max
            full_grid = full_grid[:n_max]
        return full_grid

    def make_measurement(self, design_params):
        # Generate the image using the simulator
        image = self.simulator(design_params, self.true_thetas)
        return count_bright_peaks(image), image.squeeze()

    # Define a simulator that returns concatenated angles and n_peaks
    def sub_simulator(self):
        """
        Simulate the data for a given x and theta.
        Returns the number of bright peaks measured at a given angle x
        for a given theta, and the angle x itself.
        """

        def _simulator(x, theta, noise=0, batch_size=None):
            # Generate the image using the simulator
            images = self.simulator(x, theta, noise=noise, batch_size=batch_size)
            # Count bright peaks in the image
            return torch.tensor(
                np.array([count_bright_peaks(image) for image in images]),
                dtype=torch.float32,
            ).squeeze()

        return _simulator

    def build_next_obs(self, image, angle):
        """
        Build the next observation for the SBI posterior.

        Args:
            n_peaks (int): Number of bright peaks in the image.
            image (np.ndarray): The diffraction image.
            angle (np.ndarray): The angle at which the measurement was taken.

        Returns:
            torch.Tensor: The next observation as a tensor.
        """
        # Flatten the image and concatenate with angle
        return torch.cat([image.flatten(), angle]).view(1, -1)

    def split_obs(self, observed_data, n_measured):
        """
        Split the observed data into the number of bright peaks and the angles.
        Args:
            observed_data (torch.Tensor): The observed data tensor.
            n_measured (int): The number of measurements taken.
        Returns:
            tuple: A tuple containing the number of bright peaks and the angles.
        """
        # Get the images from the observed data and reshape them to correct dimensions
        image_dim = int((observed_data.shape[1] - 5) ** 0.5)
        images = observed_data[:, :-5].view(-1, image_dim, image_dim)
        # Get the angles from the last 5 columns
        angles = observed_data[:, -5:]
        # Return the number of bright peaks and the angles
        return torch.tensor(np.array(count_bright_peaks(images))), angles

    def cat_im_ang(self, images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Concatenate images and angles for SBI"""
        return torch.cat(
            [
                images.reshape(*images.shape[:2], -1),
                angles,
            ],
            dim=-1,
        )

    def combine_no_ea(
        self, images: torch.Tensor, angles: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return self.cat_im_ang(images[:, :n_measured], angles[:, :n_measured])

    def combine_ea(
        self, images: torch.Tensor, angles: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return self.cat_im_ang(images, angles)

    def get_inbounds(self, thetas):
        return (thetas >= self.data_low_angles) & (thetas <= self.data_high_angles)

    def get_min(self, thetas):
        optimal_theta = torch.zeros_like(thetas)
        optimal_theta[..., -2:] = -thetas[..., -2:].clone()

        # Get primary inverse solution
        optimal_theta[..., :3] = get_inverse_euler_angles(thetas[..., :3])

        # Check that the inverted angles are within the prior bounds
        if not self.get_inbounds(optimal_theta[..., :3]).all():
            log.warning("Inverted angles are out of prior bounds.")
            # Report the percentage of angles out of bounds
            out_of_bounds = (~self.get_inbounds(optimal_theta[..., :3]).all(-1)).sum()
            log.warning(
                f"Percentage of angles out of bounds: {out_of_bounds.item() / optimal_theta.shape[0] * 100:.2f}%"
            )

        return self.simulator(optimal_theta, thetas, batch_size=1000), optimal_theta

    # Define function that returns samples where full posterior will be evaluated
    def alg_path_gen(self, sbi_posterior, n_mc, data_obs):
        # """Generate the algorithm path for a given theta."""
        # # For each theta generate an image at the already observed angles
        # fake_images = self.simulator(data_obs[:, -5:], theta)
        # # For each of these fake datasets, sample a new theta from the posterior
        # with open(os.devnull, "w") as fnull:
        #     with redirect_stdout(fnull), redirect_stderr(fnull):
        #         if data_obs.shape[0] > 1:
        #             posterior.set_default_x(
        #                 self.cat_im_ang(
        #                     fake_images[:-1].unsqueeze(0),
        #                     data_obs[:-1, -5:].unsqueeze(0),
        #                 ).squeeze(0)
        #             )
        #         # Sample from the posterior
        #         theta_p = posterior.sample((1,)).cpu().squeeze(0)

        # # For each of these thetas make the optimal measurement
        # images = self.simulator(theta_p, theta_p.squeeze())
        # # Concatenate these fakes with the other generated images
        # images = torch.cat((fake_images, images), dim=0)
        # angles = torch.cat(
        #     (
        #         data_obs[:, -5:].repeat(1, 1),
        #         theta_p.unsqueeze(0).cpu(),
        #     ),
        #     dim=0,
        # )
        # if data_obs.shape[0] > 1:
        #     posterior.set_default_x(data_obs)
        # # Append the angles to the images
        # return torch.cat((images.view(angles.shape[0], -1), angles), dim=-1)
        # Sample from the posterior
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                sbi_posterior.set_default_x(data_obs)
                thetas = sbi_posterior.sample((n_mc,)).cpu()
        image_mins, angle_mins = self.get_min(thetas)
        # Concatenate these fakes with the other generated images
        ea_path = self.combine_ea(
            image_mins.unsqueeze(1), angle_mins.unsqueeze(1), data_obs.shape[0]
        )
        # Broadcast the observed data to match the number of samples
        data_obs = data_obs.unsqueeze(0).repeat(thetas.shape[0], 1, 1)
        # Concatenate the observed data with the generated data
        return torch.cat((data_obs, ea_path), dim=1)

    def build_train_set(
        self,
        thetas,
        posterior,
        data_obs,
        n_measured,
    ):
        """
        Build a training set for the SBI posterior.
        This will do something similar to build_alg_path, but will be much faster.
        """
        # Repeat thetas such that every theta will be run n_measured times.
        n_thetas = thetas.shape[0]
        # For each theta generate an image at the already observed angles
        fake_images = self.simulator(
            data_obs[:, -5:].repeat(n_thetas, 1),
            thetas.repeat_interleave(n_measured, 0),
            batch_size=1000,
        )
        # Reshape the images to have the correct shape
        fake_images = fake_images.view(n_thetas, n_measured, *fake_images.shape[1:])
        # For each of these fake datasets, sample a new theta from the posterior
        # theta_p = []
        # # This has to be done sequentially due to structure of the SBI package.
        # for fake_image in fake_images:
        #     with open(os.devnull, "w") as fnull:
        #         with redirect_stdout(fnull), redirect_stderr(fnull):
        #             # Unconditional when n_measured == 1
        #             if n_measured > 1:
        #                 posterior.set_default_x(
        #                     self.cat_im_ang(
        #                         fake_image[:-1].unsqueeze(0),
        #                         data_obs[:-1, -5:].unsqueeze(0),
        #                     ).squeeze(0)
        #                 )
        #             # Sample from the posterior
        #             theta_p += [posterior.sample((1,)).cpu().squeeze(0)]
        # # For each of these thetas make the optimal measurement
        # theta_p = torch.stack(theta_p)
        theta_p = thetas
        # Get the image at the optimal measurement
        images, angle_mins = self.get_min(theta_p)
        # Concatenate these fakes with the other generated images
        images = torch.cat((fake_images, images.unsqueeze(1)), dim=1)
        angles = torch.cat(
            (
                data_obs[:, -5:].unsqueeze(0).repeat(n_thetas, 1, 1),
                angle_mins.cpu().unsqueeze(1),
            ),
            dim=1,
        )
        if data_obs.shape[0] > 1:
            posterior.set_default_x(data_obs)
        if self.figure_dir is not None:
            # Plot some of the training images and angles
            for i in range(min(5, images.shape[0])):
                self._visualize_obs(
                    images[i, -1],
                    angles[i, -1],
                    self.figure_dir / f"training_image_{i}.png",
                )
        return images, angles

    def _visualize_obs(self, image, angle, save_path):
        visualize_image(image, angle, save_path)

    def visualize_obs(self, data_obs, save_path):
        pass

    def check_posterior(self, posterior, sample_path, directory):
        # For the MAP estimate for theta, show the distribution of the number of bright peaks
        posterior.set_default_x(sample_path)
        theta_map = posterior.map().cpu()
        optimal_theta = self.get_min(theta_map)[1]
        # Generate images at randomly sampled angles
        with torch.no_grad():
            angles = self.x_grid[torch.randint(0, self.x_grid.shape[0], (500,))]
            # Add the MAP estimate to the angles
            angles = torch.cat([angles, optimal_theta], dim=0)
            images = self.simulator(angles, theta_map, batch_size=1000)
            # Run a simulation at the true angles if available
            if self.true_thetas is not None:
                # Use the current predicted best angles for the true thetas
                true_images = self.simulator(optimal_theta, self.true_thetas)
                # Get the true optimum for the true angles
                true_optimum = self.get_min(self.true_thetas)[1]
                # Make the optimal measurement
                best_images = self.simulator(true_optimum, self.true_thetas)

        # Count the number of bright peaks in the images
        n_peaks = torch.tensor(
            np.array([count_bright_peaks(image) for image in images]),
            dtype=torch.float32,
        )

        # Calculate the distance of each angle from the optimal angle
        angle_distances = torch.norm(angles - optimal_theta, dim=-1)
        # Make a scatter plot of the number of bright peaks vs the distance from the MAP estimate
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(angle_distances.cpu(), n_peaks.cpu(), alpha=0.5)
        # If true images are available, plot them as well
        if self.true_thetas is not None:
            # Count the number of peaks when making the predicted best angles
            ax.scatter(
                [0],
                count_bright_peaks(true_images),
                color="red",
                label="Predicted best with true angles",
                alpha=0.5,
            )
            # Add a
            ax.scatter(
                torch.norm(optimal_theta - true_optimum, dim=-1),
                count_bright_peaks(best_images),
                color="red",
                label="True optimum",
                alpha=0.5,
            )
        ax.legend()
        ax.set_xlabel("Distance from Optimal angle for MAP Estimate")
        ax.set_ylabel("Number of Bright Peaks")
        ax.set_title("Peak Distribution vs Distance from MAP Estimate")
        # Save the figure
        if directory is not None:
            fig.savefig(
                directory / "peak_distribution.png", dpi=300, bbox_inches="tight"
            )
        plt.close(fig)
