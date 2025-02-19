import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import torch
import os
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R

### viewpoint
class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int

def matrix_to_quaternion(rotation_matrix): 
    r = R.from_matrix(rotation_matrix)
    return np.roll(r.as_quat(), 1) ### 3DGS w,x,y,z

def quaternion_to_matrix(quat): ### x,y,z,w from pybullet
    r = R.from_quat(quat)
    return r.as_matrix # w shift to left side

def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def obj2world_transform(pos, rot_q):
    """
    having pybullet obj position and rot_quat, convert to transform_matrix in world frame
    """
    transform_matrix = np.eye(4)
    rot_matrix = quaternion_to_matrix(rot_q)
    pos_trans = np.concatenate([pos, 1], axis=1)
    pos_new = np.linalg.inv(rot_matrix) @ pos_trans
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:, 3] = pos_new.T

    return transform_matrix