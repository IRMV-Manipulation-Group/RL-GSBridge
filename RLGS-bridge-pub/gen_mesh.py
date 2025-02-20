import mcubes
import os
import torch
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.plane_utils import DetectSinglePlanes, ReadPlyPoint, RemoveNoiseStatistical, CalRotMatrix, DetectMultiPlanes
import numpy as np


def create_gaussians(iteration : int, model_list = None):
    gaussians_list = []
    if model_list is None:
        model_list = [  '/data/pybullet_3dgs/output/model_b/model_3'
                        ]
    n_model = len(model_list)
    for i in range(0,n_model):
        gaussians = GaussianModel(sh_degree = 3)

        model_path = model_list[i]
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        else:
            loaded_iter = iteration
        print("loading model:", model_path)
        gaussians.load_ply(os.path.join(model_path,
                                        "point_cloud",
                                        "iteration_" + str(loaded_iter),
                                        "point_cloud.ply"))
        gaussians_list.append(gaussians)
    return gaussians_list




if __name__ == "__main__":
    
    #gaussian = create_gaussians(-1, ['/data/pybullet_3dgs/output/model_g/model_3'])[0]
    gaussian = create_gaussians(10000, ['/data/test_cake/model_1'])[0]
    path = './mesh_models_test/cake.obj'
    path_ply = path.replace(".obj", ".ply")
    #### bg
    density_thresh = 0.12
    bottom_plane = 0.0 
    ####### plane registration #########
    raw_points = ReadPlyPoint('/data/test_cake/input_model_1.ply')

    
    #### obj param
    points = RemoveNoiseStatistical(raw_points)
    plane_param, point = DetectSinglePlanes(points, min_ratio=0.2, threshold=0.15)
    #### bg param
    # points = RemoveNoiseStatistical(raw_points, nb_neighbors=50, std_ratio=0.5)
    # plane_params, points = DetectMultiPlanes(points, min_ratio=0.2, threshold=0.2)
    #plane_param = plane_params[0] ## bg:2
    print('ground num', point.shape)
    origin_vector = -plane_param[:3]
    location_vector = np.array([0, 0, 1])
    R_w2c = CalRotMatrix(origin_vector, location_vector)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_w2c
    transform_matrix[2, 3] = -plane_param[3]
    print('trans_mat', transform_matrix)

    # transform_matrix = np.array([[ 0.9746973,    0.19617441,  -0.10714839,   0.        ], 
    # [ 0.19617441,  -0.52096007,   0.83073233,   0.        ], 
    # [ 0.10714839,  -0.83073233,  -0.54626277,   4.86178136], 
    # [ 0.        ,   0.        ,   0.        ,   1.        ]])

    mesh = gaussian.extract_mesh(path, density_thresh = density_thresh, resolution = 256, trans=transform_matrix,  bottom_plane=bottom_plane)
    # bg： 0.1 128 0.001
    # obj: 0.1 256 0.005
    mesh.write_obj(path)
    mesh.write_ply(path_ply)