import gradoptics as optics
import numpy as np

import torch
from scipy.spatial.transform import Rotation as R

def point_to_origin(cam_position):
    # Default is point along positive x
    current_dir = torch.tensor([1., 0., 0.]).type(cam_position.dtype)
    
    # Vector pointing towards the origin
    new_dir = -cam_position/torch.norm(cam_position)
    
    # Get a perpendicular axis, handling axis aligned cases
    if torch.allclose(torch.abs(new_dir), torch.abs(current_dir)):
        axis = torch.tensor([0., 0., 1.]).double()
    else:
        axis = torch.linalg.cross(current_dir, new_dir)
     
    # Normalize
    axis *= 1/torch.norm(axis)
    
    # Get rotation angle via dot product
    angle = torch.acos(torch.dot(current_dir,new_dir)/(torch.norm(current_dir)*torch.norm(new_dir)))
    
    # Get Euler angles from rotation vector using scipy (right handed coordinate system)
    theta_x, theta_y, theta_z = R.from_rotvec(axis*angle).as_euler('xyz')
    
    # Gradoptics rotations need left handed coordinate system -- flip sign on y
    return theta_x, -theta_y, theta_z

def calculate_focal_length(m, obj_distance):
    f =  obj_distance / ((1 / m) + 1)

    return f

def setup_scene(scene, thetas=[], phis=[], rs=[]):
    #Make sure the number of cameras matches between arguments
    assert len(thetas) == len(phis), "thetas and phis should have the same length"
    assert len(rs) == len(thetas), "rs and thetas should have the same length"
    
    n_cameras = len(thetas)
    
    # Given distance from object and magnification, calculate focal length for in focus object
    m = 0.1
    f = calculate_focal_length(m, rs[0])
    
    # Numerical aperture (size of camera opening, f-number)
    na = 1/1.4

    # Loop over cameras to add to scene
    for i_cam in range(n_cameras):
        # Avoid singular point with slight offset
        if thetas[i_cam] == 0:
            thetas[i_cam] = 1e-6
            
        # Get cartesian coordinates from spherical
        x_cam = rs[i_cam]*np.sin(thetas[i_cam])*np.cos(phis[i_cam])
        y_cam = rs[i_cam]*np.sin(thetas[i_cam])*np.sin(phis[i_cam])
        z_cam = rs[i_cam]*np.cos(thetas[i_cam])
        
        cam_pos = torch.tensor([x_cam, y_cam, z_cam])

        # Get orientation to point at origin and apply to lens
        angles = point_to_origin(cam_pos)
        transform = optics.simple_transform.SimpleTransform(*angles, cam_pos)
        lens = optics.PerfectLens(f=f, m=m, na=na,
                                  position = cam_pos,
                                  transform = transform)

        # Sensor position from lensmakers equation, rotated to match lens
        rel_position = torch.tensor([-f * (1 + m), 0, 0])                       
        rot_position = torch.matmul(transform.transform.float(), torch.cat((rel_position, torch.tensor([0]))))

        sensor_position = cam_pos + rot_position[:-1]
        viewing_direction = torch.matmul(transform.transform.float(), torch.tensor([1.,0,0,0]))

        sensor = optics.Sensor(position=sensor_position, viewing_direction=tuple(viewing_direction.numpy()),
                               resolution=(200,200), pixel_size=(2.4e-06, 2.4e-06),
                               poisson_noise_mean=2.31, quantum_efficiency=0.72)
        
        # Add sensor and lens to the scene
        scene.add_object(sensor)
        scene.add_object(lens)
        
    return scene, n_cameras


def get_pixel_coords(sensor):
    # Pixel coordinates in camera space
    x = -torch.linspace(-sensor.pixel_size[0]*sensor.resolution[0]/2 + sensor.pixel_size[0]/2,
                         sensor.pixel_size[0]*sensor.resolution[0]/2 - sensor.pixel_size[0]/2, 
                         sensor.resolution[0])

    y = -torch.linspace(-sensor.pixel_size[1]*sensor.resolution[1]/2 + sensor.pixel_size[1]/2,
                         sensor.pixel_size[1]*sensor.resolution[1]/2 - sensor.pixel_size[1]/2, 
                         sensor.resolution[1])
    
    
    pix_x, pix_y = torch.meshgrid(x, y)
    
    pix_z = torch.zeros((sensor.resolution[0], sensor.resolution[1]))
    
    all_coords = torch.stack([pix_x, pix_y, pix_z], dim=-1).reshape((-1, 3)).double()
    
    # Use transforms from above setup to go from pixel space to real (world) space
    return sensor.c2w.apply_transform_(all_coords)

def get_rays_pinhole(sensor, lens, nb_rays=None, ind=None, device='cuda', return_ind=False):
    
    if ind is None:
        #Set up for ray batching -- nb_rays switches between all pinhole rays or random batches
        if nb_rays == None or nb_rays == sensor.resolution[0]*sensor.resolution[1]:
            ind = torch.arange(0, sensor.resolution[0]*sensor.resolution[1])
        else:
            ind = torch.randint(0, sensor.resolution[0]*sensor.resolution[1], (nb_rays,))
    
    # Get origins
    all_pix_coords = get_pixel_coords(sensor)  
    origins = all_pix_coords[ind]
    
    #Get directions to center of lens
    lens_center = lens.transform.transform[:-1, -1]
    
    directions = optics.batch_vector(lens_center[None, 0] - origins[:, 0],
                                     lens_center[None, 1] - origins[:, 1],
                                     lens_center[None, 2] - origins[:, 2]).type(origins.dtype)

    # Set up rays
    rays_sensor_to_lens = optics.Rays(origins, directions, device=device)
    
    if return_ind:
        return rays_sensor_to_lens, ind
    else:
        return rays_sensor_to_lens