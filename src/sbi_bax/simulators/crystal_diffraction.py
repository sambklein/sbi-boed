# SB+ BAX for crystal diffraction images
import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core.structure import Structure


def rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Generate rotation matrices on GPU for batched angles.

    Args:
        angles: (batch_size, 3) tensor of Euler angles
    Returns:
        (batch_size, 3, 3) tensor of rotation matrices
    """
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)

    angle_x, angle_y, angle_z = angles.unbind(dim=-1)

    # Precompute trig functions
    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)

    # Zero and one tensors
    zeros = torch.zeros_like(cos_x)
    ones = torch.ones_like(cos_x)

    # Rotation matrices - vectorized construction
    Rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_x, -sin_x], dim=-1),
            torch.stack([zeros, sin_x, cos_x], dim=-1),
        ],
        dim=-2,
    )

    Ry = torch.stack(
        [
            torch.stack([cos_y, zeros, sin_y], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_y, zeros, cos_y], dim=-1),
        ],
        dim=-2,
    )

    Rz = torch.stack(
        [
            torch.stack([cos_z, -sin_z, zeros], dim=-1),
            torch.stack([sin_z, cos_z, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    # Combined rotation: R = Rz @ Ry @ Rx
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    return R


def extract_euler_angles(R: torch.Tensor) -> torch.Tensor:
    """
    Extract Euler angles from rotation matrices.
    Assumes rotation order: R = Rz(γ) @ Ry(β) @ Rx(α)

    Args:
        R: (batch_size, 3, 3) rotation matrices
    Returns:
        (batch_size, 3) Euler angles [α, β, γ]
    """
    # Handle numerical precision issues
    R = torch.clamp(R, -1.0, 1.0)

    # Extract angles using the ZYX convention
    # For R = Rz(γ) @ Ry(β) @ Rx(α), the decomposition is:

    # β (y-rotation) from R[2,0] = -sin(β)
    sin_beta = -R[..., 2, 0]
    sin_beta = torch.clamp(sin_beta, -1.0, 1.0)  # Clamp for numerical stability
    beta = torch.asin(sin_beta)

    # Check for gimbal lock (cos(β) ≈ 0)
    cos_beta = torch.cos(beta)
    gimbal_lock = torch.abs(cos_beta) < 1e-6

    # Normal case (no gimbal lock)
    alpha_normal = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    gamma_normal = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    # Gimbal lock case: set α = 0 and solve for γ
    alpha_gimbal = torch.zeros_like(alpha_normal)
    gamma_gimbal = torch.atan2(-R[..., 0, 1], R[..., 1, 1])

    # Choose based on gimbal lock condition
    alpha = torch.where(gimbal_lock, alpha_gimbal, alpha_normal)
    gamma = torch.where(gimbal_lock, gamma_gimbal, gamma_normal)

    return torch.stack([alpha, beta, gamma], dim=-1)


def get_inverse_euler_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    Get inverse Euler angles from given angles.

    Args:
        angles: (batch_size, 3) tensor of Euler angles [α, β, γ]
    Returns:
        (batch_size, 3) tensor of inverse Euler angles [-γ, -β, -α]
    """
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)

    # Build a rotation matrix from the angles
    rotation_matrices = rotation_matrix(angles)
    # Extract inverse angles
    inverse_angles = extract_euler_angles(rotation_matrices.transpose(-1, -2))
    # Return inverse angles with sign flips
    return inverse_angles


class CrystalSimulator:
    def __init__(
        self,
        structure_file,
        wavelength,
        detector_distance,
        detector_size,
        pixel_resolution,
        sample_size,
        device="cuda",
    ):
        self.device = device
        try:
            self.structure = Structure.from_file(structure_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Structure file {structure_file} not found.")
        self.reciprocal_lattice = (
            self.structure.lattice.reciprocal_lattice_crystallographic
        )
        self.k0 = 2 * np.pi / wavelength
        self.detector_distance = detector_distance
        self.detector_size = detector_size
        self.pixel_resolution = pixel_resolution

        # Pixels in full detector with no finite size effects
        self.n_pixels = int(3 * detector_size / pixel_resolution)
        self.center = self.n_pixels // 2

        # Precompute HKL indices and g-vectors on GPU
        max_index = 10
        hkl = []
        for h in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                for ll in range(-max_index, max_index + 1):
                    if not (h == k == ll == 0):
                        hkl.append([h, k, ll])

        self.hkl_indices = torch.tensor(hkl, device=device, dtype=torch.float32)

        # Convert g_vectors to GPU tensor
        g_vectors_np = self.reciprocal_lattice.get_cartesian_coords(hkl)
        self.g_vectors = torch.tensor(g_vectors_np, device=device, dtype=torch.float32)

        # Precompute constants as tensors
        self.k0_vec = torch.tensor([0, 0, self.k0], device=device, dtype=torch.float32)

        # Build finite size effect parameters
        self.sample_size = sample_size
        self.grid_y, self.grid_x = torch.meshgrid(
            torch.arange(self.n_pixels), torch.arange(self.n_pixels), indexing="ij"
        )
        # Slice for detector crop
        start = self.center - int(self.detector_size / (2 * pixel_resolution))
        end = self.center + int(self.detector_size / (2 * pixel_resolution))
        self._crop = slice(start, end)
        # radius of sample in pixels
        self.r_pix = (self.sample_size / 2) / self.pixel_resolution

    def generate_images_gpu(self, Rs: torch.Tensor) -> torch.Tensor:
        """
        Generate diffraction images entirely on GPU.

        Args:
            Rs: (batch_size, 3, 3) rotation matrices on GPU
        Returns:
            (batch_size, n_pixels, n_pixels) images on GPU
        """
        batch_size = Rs.shape[0]
        n_hkl = self.g_vectors.shape[0]

        # Rotate g_vectors: (batch_size, n_hkl, 3)
        g_rotated = torch.einsum("bij,nj->bni", Rs, self.g_vectors)

        # Construct k1 vectors: (batch_size, n_hkl, 3)
        k1_vectors = g_rotated + self.k0_vec

        # Bragg condition (vectorized): (batch_size, n_hkl)
        k1_norms = torch.norm(k1_vectors, dim=-1)
        valid_bragg = torch.abs(k1_norms - self.k0) <= 0.05

        # Forward scattering condition
        k1_z_positive = k1_vectors[..., 2] > 0

        # Detector projection (with safe division)
        k1_z_safe = torch.where(
            k1_z_positive, k1_vectors[..., 2], torch.ones_like(k1_vectors[..., 2])
        )
        scale = self.detector_distance / k1_z_safe
        x_det = k1_vectors[..., 0] * scale
        y_det = k1_vectors[..., 1] * scale

        # Pixel coordinates
        i_coords = (self.center + x_det / self.pixel_resolution).long()
        j_coords = (self.center + y_det / self.pixel_resolution).long()

        # Bounds checking
        valid_bounds = (
            (i_coords >= 0)
            & (i_coords < self.n_pixels)
            & (j_coords >= 0)
            & (j_coords < self.n_pixels)
        )

        # Combined validity mask
        valid_mask = valid_bragg & k1_z_positive & valid_bounds

        # Create images using advanced indexing (GPU-optimized)
        images = torch.zeros(
            batch_size,
            self.n_pixels,
            self.n_pixels,
            device=self.device,
            dtype=torch.float32,
        )

        if valid_mask.any():
            # Vectorized image creation
            batch_indices = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                .expand(-1, n_hkl)
            )

            # Use scatter_add for efficient accumulation
            valid_batch = batch_indices[valid_mask]
            valid_i = i_coords[valid_mask]
            valid_j = j_coords[valid_mask]

            # Flatten indices for scatter_add
            try:
                flat_indices = (
                    valid_batch * (self.n_pixels * self.n_pixels)
                    + valid_j * self.n_pixels
                    + valid_i
                )
            except RuntimeError:
                pass
            images_flat = images.view(-1)

            # Add intensities
            intensities = torch.full_like(flat_indices, 100.0, dtype=torch.float32)
            images_flat.scatter_add_(0, flat_indices, intensities)

            images = images_flat.view(batch_size, self.n_pixels, self.n_pixels)

            # GPU Gaussian filtering (using conv2d)
            images = self.gaussian_filter_gpu(images, sigma=2)

        return images

    def gaussian_filter_gpu(self, images: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian filtering on GPU using conv2d"""
        # Convert sigma to pixel units (factor of 2 for legacy reasons)
        sigma = sigma / (2 * self.pixel_resolution)
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Generate 2D Gaussian kernel
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32)
        x = x - kernel_size // 2
        gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        kernel_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, kernel_size, kernel_size)

        # Apply convolution
        padding = kernel_size // 2
        images_filtered = F.conv2d(
            images.unsqueeze(1),
            kernel_2d,
            padding=padding,  # Add channel dimension
        ).squeeze(1)  # Remove channel dimension

        return images_filtered

    def simulate_full_pattern(
        self, measured_angles, crystal_orientation, noise=0, batch_size=None
    ):
        # Convert inputs to GPU tensors
        if not isinstance(measured_angles, torch.Tensor):
            measured_angles = torch.tensor(
                measured_angles, device=self.device, dtype=torch.float32
            )
        else:
            measured_angles = measured_angles.to(self.device)

        if not isinstance(crystal_orientation, torch.Tensor):
            crystal_orientation = torch.tensor(
                crystal_orientation, device=self.device, dtype=torch.float32
            )
        else:
            crystal_orientation = crystal_orientation.to(self.device)

        # Ensure correct dimensions for angles
        if measured_angles.dim() == 1:
            measured_angles = measured_angles.unsqueeze(0)

        if crystal_orientation.dim() == 1:
            crystal_orientation = crystal_orientation.unsqueeze(0)

        # Generate rotation matrices on GPU
        R_measured = rotation_matrix(measured_angles)
        R_initial = rotation_matrix(crystal_orientation)

        # If there is only one of either measured_angles or crystal_orientation then repeat to match
        if R_measured.shape[0] == 1 and R_initial.shape[0] > 1:
            R_measured = R_measured.expand(R_initial.shape[0], -1, -1)
        elif R_initial.shape[0] == 1 and R_measured.shape[0] > 1:
            R_initial = R_initial.expand(R_measured.shape[0], -1, -1)
        elif R_measured.shape[0] != R_initial.shape[0]:
            raise ValueError(
                "Measured angles and crystal orientation must have the same batch size."
            )

        # Combine rotations
        Rs = torch.bmm(R_measured, R_initial)

        # Batch processing if batch_size is specified
        if batch_size is not None and Rs.shape[0] > batch_size:
            # Create batches and process iteratively
            all_images = []
            for i in range(0, Rs.shape[0], batch_size):
                batch_images = self.generate_images_gpu(Rs[i : i + batch_size])
                # Move to CPU and append to list
                all_images.append(batch_images.cpu())
            # Concatenate all batches
            images = torch.cat(all_images, dim=0)
        else:
            # Process all at once (existing logic)
            images = self.generate_images_gpu(Rs)

        # Add noise if requested
        if noise > 0:
            noise_tensor = torch.randn_like(images) * noise
            images = images + noise_tensor

        return images.cpu()

    def apply_sample_mask(
        self,
        images: torch.Tensor,
        angles: torch.Tensor,
        beam_center_mm: torch.Tensor,
        batch_size: int = None,
    ) -> torch.Tensor:
        """
        Mask by a rotated circular sample footprint (appears as ellipse in x-y).
        Circle radius = sample_size/2 in sample frame, then:
         1) rotate by rot_z about sample center
         2) translate by beam_center_mm (mm) → pixel offset
        """
        # ensure batch dim
        if images.dim() == 2:
            images = images.unsqueeze(0)
        B, H, W = images.shape
        if batch_size is None:
            batch_size = B

        # normalize rot_z and beam_center to batch
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        if angles.numel() == 3 and angles.shape[0] == 1:
            angles = angles.expand(B, -1)
        if angles.shape[0] != B:
            raise ValueError("Angles must have the same batch size as images.")
        beam_center_mm = beam_center_mm.view(-1, 2)
        if beam_center_mm.size(0) == 1:
            beam_center_mm = beam_center_mm.expand(B, 2)

        # compute beam center in pixel coords
        cx = self.center + beam_center_mm[:, 0] / self.pixel_resolution
        cy = self.center + beam_center_mm[:, 1] / self.pixel_resolution

        # get grid on correct device
        xx = self.grid_x.to(images.device).unsqueeze(0).float()
        yy = self.grid_y.to(images.device).unsqueeze(0).float()
        for i in range(0, B, batch_size):
            # Shift the grid to the beam center
            x_rel = xx - cx[i : i + batch_size].view(-1, 1, 1)
            y_rel = yy - cy[i : i + batch_size].view(-1, 1, 1)

            # # Extract the angles, treating the crystal as a 2D object
            # rot_z = angles[:, -1].view(-1, 1, 1)  # Get the rotation around z-axis
            # rot_y = angles[:, -2].view(-1, 1, 1)  # Get the rotation around y-axis
            # scale_x = torch.cos(angles[:, -3].view(-1, 1, 1))  # Get the scale around x-axis
            # mask = ellipse in original frame
            # mask = (
            #     x_rel**2 / (self.r_pix * torch.cos(rot_z) * scale_x) ** 2
            #     + y_rel**2 / (self.r_pix * torch.cos(rot_y) * scale_x) ** 2
            # ) <= 1

            # Create the mask for the circular sample footprint treating sample as a sphere
            mask = (x_rel**2 + y_rel**2) <= self.r_pix**2
            # Do masking inplace to avoid memory overhead
            images[i : i + batch_size] *= mask.float()
        return images

    def __call__(
        self,
        measured_angles_xy: torch.Tensor,
        crystal_orientation_xy: torch.Tensor,
        noise: float = 0,
        batch_size: int = None,
    ) -> torch.Tensor:
        """
        1) Simulate full pattern
        2) Cut out (mask) sample finite‐size
        3) Then cut out (mask) beam finite‐size
        """
        # Get the angles
        measured_angles = measured_angles_xy[..., :3]
        crystal_orientation = crystal_orientation_xy[..., :3]
        # Get the offsets
        measured_xy = measured_angles_xy[..., 3:]
        crystal_xy = crystal_orientation_xy[..., 3:]
        images = self.simulate_full_pattern(
            measured_angles, crystal_orientation, noise=noise, batch_size=batch_size
        )
        images = self.apply_sample_mask(
            images, measured_angles, measured_xy + crystal_xy, batch_size=batch_size
        )
        # Apply beam mask by cropping out the center region
        images = images[..., self._crop, self._crop]
        return images.cpu()
