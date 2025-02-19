import os
import random
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

from gaussian_renderer import render
import torch

import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
#from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.view_utils import matrix_to_quaternion, quaternion_raw_multiply, CameraInfo
from utils.sh_utils import eval_sh
import pdb

from scipy.spatial.transform import Rotation as R


####### 以下是我们要用的函数
def create_gaussians(iteration : int, model_list = None): # TODO: 改成加载多个模型
    gaussians_list = []
    if model_list is None:
        model_list = [  "/data/gaussian-splatting/output/rm_bg_print_mask", 
            #'/data/gaussian-splatting/output/38eea1e2-9', 
                        '/data/gaussian-splatting/output/38eea1e2-9'
                        ]
    for i in range(0,2):
        gaussians = GaussianModel(3)

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


def gaussian_merge(viewpoint_camera, pcs : GaussianModel, transforms, rots, convert_SHs_python, obj_scale=1.0): ### 改变物体高斯的程序
    """
    transform: 4X4 matrix ！注意一直是相对于初始加载模型的！
    rot: relative quaternion

    ### GS模型现在加载模式是以世界为起点，变换要拿当前位姿除以初始位姿才是变换量（且为世界坐标系）
    ### 现实世界搭建时，以世界为起点做SAM，模型就以当前坐标系放到环境中然后调整，用调整量加真值位姿作为变换位姿就可以了
    ### 语义GS方法：同理。
    """

    n_obj = len(transforms)
    means3D_list = []
    opacity_list = []
    scale_list = []
    rotation_list = []
    shs_list = []
    colors_precomp_list = []

    device = pcs[0].get_xyz.device
    for i in range(0, n_obj): 

        pc = pcs[i]
        transform = torch.tensor(transforms[i], dtype=torch.float32, device=device)
        rot = torch.tensor(rots[i], dtype=torch.float32, device=device)
        # 获取高斯分布的三维坐标、屏幕空间坐标和透明度。
        means3D = pc.get_xyz
        opacity = pc.get_opacity
    
        scale = pc.get_scaling
        rotations = pc.get_rotation

        #obj_scale = 0.1
        means3D = obj_scale * means3D
        scale = obj_scale * scale

        ### position transform
        # TODO: directly add transform! 这需要确定旋转的中心点 --> 已知物体位置与物体姿态，可以计算世界到物体系的变换
        rotations = quaternion_raw_multiply(rot, rotations)
        xyz_trans = torch.cat([means3D, torch.ones((means3D.shape[0], 1), device=device)], dim=1).T
        means3D =  (torch.matmul(transform, xyz_trans).T)[:, :3]
        #pdb.set_trace()
        
        means3D_list.append(means3D)
        opacity_list.append(opacity)
        rotation_list.append(rotations)
        scale_list.append(scale)
        
        if convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) #将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) #计算相机中心到每个点的方向向量，并归一化。
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)   #计算相机中心到每个点的方向向量，并归一化。
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) #使用SH特征将方向向量转换为RGB颜色。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) #将RGB颜色的范围限制在0到1之间。
            colors_precomp_list.append(colors_precomp)
        else:
            shs = pc.get_features
            shs_list.append(shs)

    means3D_list = torch.cat(means3D_list, dim=0)
    opacity_list = torch.cat(opacity_list, dim=0)
    scale_list = torch.cat(scale_list, dim=0)
    rotation_list = torch.cat(rotation_list, dim=0)

    #pdb.set_trace()
    if convert_SHs_python:
        colors_precomp_list = torch.cat(colors_precomp_list, dim=0)
        shs_list = None
        return {
        "means3D": means3D_list,   # cuda
        "rotations": rotation_list,
        "scales" : scale_list,
        "opacity": opacity_list,
        "colors_precomp": colors_precomp_list
        }
    else:
        shs_list = torch.cat(shs_list, dim=0)
        colors_precomp_list = None
        return {
        "means3D": means3D_list,   # cuda
        "rotations": rotation_list,
        "scales" : scale_list,
        "opacity": opacity_list,
        "shs": shs_list,
        }


class SimGaussian:
    def __init__(self, params):
        self.gaussians_list = create_gaussians(-1, params['model_list'])
        self.bg_color = [1,1,1] if params['white_background'] else [0, 0, 0]
        self.convert_SHs_python = params['convert_SHs_python']
        self.obj_scale = params['obj_scale']
        self.viewpoint_camera = None
        #self.model_data = None
        
        transform_matrix = np.array([-1., 0., 0.,0.,
                  0., 0., 1., 0.,
                  0., 1., 0., 0.,
                  0., 0., 0., 1,], np.float32).reshape(4, 4)
        self.init_camera(transform_matrix, params)
        #### TODO 使用convert_SHs_python时每次相机变化要更新颜色特征
        self.bg_GS_data = self.init_background()
        
    def init_background(self):

        pc = self.gaussians_list[0]
        
        means3D = pc.get_xyz
        opacity = pc.get_opacity
    
        scale = pc.get_scaling
        rotations = pc.get_rotation
        
        if self.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) #将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
            dir_pp = (pc.get_xyz - self.viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) #计算相机中心到每个点的方向向量，并归一化。
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)   #计算相机中心到每个点的方向向量，并归一化。
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) #使用SH特征将方向向量转换为RGB颜色。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) #将RGB颜色的范围限制在0到1之间。
            return {
            "means3D": means3D,   # cuda
            "rotations": rotations,
            "scales" : scale,
            "opacity": opacity,
            "colors_precomp": colors_precomp
            }
        else:
            shs = pc.get_features
            return {
            "means3D": means3D,   # cuda
            "rotations": rotations,
            "scales" : scale,
            "opacity": opacity,
            "shs": shs,
            }

    def init_camera(self, view_mat, params):
        c2w = np.linalg.inv(np.array(view_mat))
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        Rot = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        Trans = w2c[:3, 3]

        ####### cameraview loading #################################
        cam_info = CameraInfo(R=Rot, T=Trans, FovY=params['camera_setting']['Fov']/180*np.pi, FovX=params['camera_setting']['Fov']/180*np.pi,
                                width=params['camera_setting']['img_W'], height=params['camera_setting']['img_H'])

        self.viewpoint_camera = Camera(R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    W=cam_info.width, H=cam_info.height )#data_device=args.data_device)
        
    def update_camera(self, view_mat):
        """
        view_matrix: pybullet view_matrix
        """
        c2w = np.linalg.inv(np.array(view_mat))
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        Rot = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        Trans = w2c[:3, 3]

        self.viewpoint_camera.update_view(Rot, Trans)

    def update_and_render(self, trans_list, rots_list):
        
        model_data = gaussian_merge(self.viewpoint_camera, self.gaussians_list[1:], trans_list, rots_list, self.convert_SHs_python, obj_scale = self.obj_scale)
        
        model_data['means3D'] = torch.cat([self.bg_GS_data['means3D'], model_data['means3D']], dim=0)
        model_data['rotations'] = torch.cat([self.bg_GS_data['rotations'], model_data['rotations']], dim=0)
        model_data['scales'] = torch.cat([self.bg_GS_data['scales'], model_data['scales']], dim=0)
        model_data['opacity'] = torch.cat([self.bg_GS_data['opacity'], model_data['opacity']], dim=0)
        
        print("merge finished")
        if self.convert_SHs_python:
            model_data['colors_precomp'] = torch.cat([self.bg_GS_data['colors_precomp'], model_data['colors_precomp']], dim=0)
            render_img = render(self.viewpoint_camera, model_data['means3D'], 
                            model_data['rotations'], model_data['scales'], model_data['opacity'], self.bg_color, 
                            shs=None, colors_precomp=model_data['colors_precomp'])['render']
        else:
            model_data['shs'] = torch.cat([self.bg_GS_data['shs'], model_data['shs']], dim=0)
            render_img = render(self.viewpoint_camera, model_data['means3D'], 
                            model_data['rotations'], model_data['scales'], model_data['opacity'], self.bg_color, 
                            shs=model_data['shs'], colors_precomp=None, scaling_modifier=1)['render']
        print("render finished")
        return render_img
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    #model = ModelParams(parser, sentinel=True)
    #pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    args = parser.parse_args()#get_combined_args(parser)
    #print("Rendering " + args.model_path)
    print("Rendering ")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    #######################################################################
    ####### model loading #################################
    params = {
        'model_list':[#"/data/gaussian-splatting/output/rm_bg_print_mask", 
                    "/data/gaussian-splatting/output/rm_bg_print_mask", 
                    '/data/gaussian-splatting/output/38eea1e2-9'  ], 
        'convert_SHs_python':args.convert_SHs_python, 
        'white_background':args.white_background, 
        'obj_scale':0.1, 
        'camera_setting':{
            'Fov':50., 
            'img_H':512, 
            'img_W':512
        }
    }
    simmodel = SimGaussian(params)
    
    #########################################################################
    ####### camera update #################################
    transform_matrix = [
                [
                    6.123234262925839e-17,
                    1.0,
                    0.0,
                    -3.9801019400295505e-17
                ],
                [
                    -0.16439901292324066,
                    1.006653659186413e-17,
                    0.9863939881324768,
                    0.00821995735168457
                ],
                [
                    0.9863939881324768,
                    -6.039921293373988e-17,
                    0.16439901292324066,
                    -0.9617341756820679
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]]
    
    simmodel.update_camera(transform_matrix)
    #########################################################################
    ####### pose transform loading #################################
    euler_angles = [0., 0., 30.]  # 这里以度为单位，如果以弧度为单位，则不需要转换
 
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    
    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()

    transforms_2 = np.array([1., 0., 0.,0.45,
                  0., 1., 0., 0.1,
                  0., 0., 1., 0.1,
                  0., 0., 0., 0.], np.float32).reshape(4, 4)
    transforms_2[:3, :3] = rotation_matrix

    rot_mat_2 = transforms_2[:3, :3]
    quat2 = matrix_to_quaternion(rot_mat_2)

    #rots = np.array([1, 0, 0, 0], np.float32)

    trans_list = []
    rots_list = []
    trans_list.append(transforms_2)
    rots_list.append(quat2)
    
    render_img = simmodel.update_and_render(trans_list, rots_list)
    idx = 6
    torchvision.utils.save_image(render_img, os.path.join('./test_out', '{0:05d}'.format(idx) + ".png"))
