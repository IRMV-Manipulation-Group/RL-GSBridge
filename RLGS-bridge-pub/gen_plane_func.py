from utils.plane_utils import DetectSinglePlanes, ReadPlyPoint, RemoveNoiseStatistical, CalRotMatrix, DetectMultiPlanes
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='')
    args = parser.parse_args()
    ####### plane registration #########
    #raw_points = ReadPlyPoint('/data/test_cake/input_model_1.ply')
    #raw_points = ReadPlyPoint('/data/exp_obj_GS/real_mesh_cake_aug_noeval/input.ply')
    raw_points = ReadPlyPoint("/data/exp_obj_data/"+ args.name +"/sparse/points3D.ply")

    #### obj param
    # points = RemoveNoiseStatistical(raw_points)
    # plane_params, point = DetectMultiPlanes(points, min_ratio=0.2, threshold=0.15)
    #### bg param
    points = RemoveNoiseStatistical(raw_points, nb_neighbors=50, std_ratio=0.5)
    plane_params, points = DetectMultiPlanes(points, min_ratio=0.2, threshold=0.2)
    plane_param = plane_params[0] ## bg:2
    #print('ground num', point.shape)
    origin_vector = -plane_param[:3]
    location_vector = np.array([0, 0, 1])
    R_w2c = CalRotMatrix(origin_vector, location_vector)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_w2c
    transform_matrix[2, 3] = -plane_param[3]
    np.set_printoptions(suppress=True)
    print('trans_mat', transform_matrix)
    # np.savetxt('./plane_mat.txt', np.c_[transform_matrix],
    #         fmt='%f',delimiter='\t')
