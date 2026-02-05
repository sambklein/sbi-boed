import torch
import torch.nn.functional as F

from gradoptics.distributions.base_distribution import BaseDistribution


def query_voxel_pdf(volume, coords, bounds):
    """
    Trilinear interpolate values from a voxelized PDF in world coordinates.

    Args:
        volume: (D, H, W) tensor of voxel values
        coords: (N, 3) tensor of query points in world coords (x,y,z)
        bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) world coordinate bounds

    Returns:
        (N,) tensor of interpolated values
    """
    D, H, W = volume.shape
    vol = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    # map world -> normalized grid coords [-1,1]
    norm_x = 2.0 * (coords[:, 0] - xmin) / (xmax - xmin) - 1.0
    norm_y = 2.0 * (coords[:, 1] - ymin) / (ymax - ymin) - 1.0
    norm_z = 2.0 * (coords[:, 2] - zmin) / (zmax - zmin) - 1.0

    grid = torch.stack((norm_x, norm_y, norm_z), dim=-1)  # (N,3)
    grid = grid.view(1, 1, 1, -1, 3)  # reshape for grid_sample

    vals = F.grid_sample(vol, grid, mode="bilinear",
                         align_corners=True)  # (1,1,1,N)
    return vals.view(-1)


class VoxelDistribution(BaseDistribution):
    """
    3D Voxel Distribution.
    """

    def __init__(self, values, voxel_size, voxel_center, device='cuda'):
        super().__init__()
        
        n_voxels = values.shape
        bound_x = (voxel_center[0]-n_voxels[0]/2*voxel_size[0],
                   voxel_center[0]+n_voxels[0]/2*voxel_size[0])
        bound_y = (voxel_center[1]-n_voxels[1]/2*voxel_size[1],
                   voxel_center[1]+n_voxels[1]/2*voxel_size[1])
        bound_z = (voxel_center[2]-n_voxels[2]/2*voxel_size[2],
                   voxel_center[2]+n_voxels[2]/2*voxel_size[2])
        
        self.bounds = (bound_x, bound_y, bound_z)
        self.voxel_center = voxel_center
        self.values = values.to(device)
        
    def sample(self, nb_points, device='cpu'):
        pass

    def pdf(self, x):
        """
        Returns the pdf function evaluated at ``x``

        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)

        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        return query_voxel_pdf(self.values.to(x), x, self.bounds)
    
    def plot(self, ax):
        ax.scatter(self.voxel_center[0], self.voxel_center[1], self.voxel_center[2])