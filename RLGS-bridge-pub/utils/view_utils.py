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

def euler_to_quaternion(euler): 
    #print('euler', euler)
    r = R.from_euler('xyz', euler)
    return r.as_quat()### 3DGS w,x,y,z

def quaternion_to_matrix(quat): ### x,y,z,w from pybullet
    r = R.from_quat(quat)
    return r.as_matrix() # w shift to left side

def quaternion_inverse(quat):
    #print('quat:', quat)
    aw, ax, ay, az = quat[0], quat[1], quat[2], quat[3]
    c_quat = np.stack([aw, -ax, -ay, -az], axis=-1)
    l2 = np.linalg.norm(quat, axis=-1)
    quat_inv = c_quat/l2
    return quat_inv

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
    if isinstance(a, torch.Tensor):
        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((ow, ox, oy, oz), -1)
    else:
        ## TODO: 换成批量操作。
        aw, ax, ay, az = a[0], a[1], a[2], a[3]
        bw, bx, by, bz = b[0], b[1], b[2], b[3]
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return np.stack([ow, ox, oy, oz], axis=-1)

def obj2world_transform(pos, rot_q):
    """
    having pybullet obj position and rot_quat, convert to transform_matrix in world frame
    """
    transform_matrix = np.eye(4)
    rot_matrix = quaternion_to_matrix(rot_q)
    #pos_trans = np.append(pos, 1)
    #print("pos:", pos_trans)
    #print("rot:", rot_matrix)
    pos_new = np.linalg.inv(rot_matrix) @ pos
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = np.array(pos).T

    return transform_matrix